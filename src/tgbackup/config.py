"""Configuration, path, and credential helpers for Telegram backups.

This module deliberately contains no exporter orchestration.  It is safe to
import from the CLI, GUI worker, maintenance commands, and tests without
initialising a Telegram client.
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import tempfile
from pathlib import Path

from .errors import ExportError

PROJECT_ROOT = Path(os.environ.get("TGBACKMAN_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SCRIPT_DIR = PROJECT_ROOT


def database_mtime_ns(path: Path) -> int:
    """Return the newest mtime of SQLite's main file and write-ahead log."""
    candidates = [path]
    wal = Path(f"{path}-wal")
    if wal.is_file() and wal.stat().st_size > 0:
        candidates.append(wal)
    existing = [candidate for candidate in candidates if candidate.exists()]
    return max(candidate.stat().st_mtime_ns for candidate in existing)


def default_database_path() -> Path:
    """Choose one deterministic database path; never infer authority from mtime."""
    configured = os.environ.get("TGBACKMAN_DB", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    local = SCRIPT_DIR / "sqlitedb" / "telegram_backup.db"
    if local.is_file():
        return local
    media_root = Path("/media") / getpass.getuser()
    configured_volume = os.environ.get("TGBACKMAN_REMOVABLE_VOLUME", "").strip("/")
    if configured_volume:
        removable_candidates = [media_root / configured_volume / "sqlitedb" / "telegram_backup.db"]
    else:
        removable_candidates = sorted(media_root.glob("*/sqlitedb/telegram_backup.db"))
    if removable_candidates:
        return removable_candidates[0]
    return local


DEFAULT_DB = default_database_path()
DEFAULT_CONFIG = Path.home() / ".config" / "tgbackman" / "credentials.env"
DEFAULT_SESSION = Path.home() / ".local" / "share" / "tgbackman" / "telegram"
DEFAULT_OUTPUT = Path.home() / "Telegram Backup" / "Telegram API Incremental"
TARGETS_TABLE = "telegram_backup_targets"
EXPORTS_TABLE = "telegram_backup_exports"
RUNS_TABLE = "telegram_backup_runs"
RUN_MESSAGES_TABLE = "telegram_backup_run_messages"
RUN_ARCHIVE_TABLE = "telegram_backup_run_records"
RUN_ATTEMPTS_TABLE = "telegram_backup_run_attempts"
TARGET_CHAT_LINKS_TABLE = "telegram_backup_target_chats"
DIALOGS_TABLE = "telegram_dialogs"
PURGES_TABLE = "telegram_backup_purges"
BLACKLIST_TABLE = "telegram_backup_blacklist"
DIAGNOSTIC_EVENTS_TABLE = "telegram_backup_diagnostic_events"
MEDIA_TYPES = frozenset({"photo", "video", "voice_message", "audio_file", "sticker", "animation", "file"})
MEDIA_ALIASES = {
    "photo": "photo",
    "photos": "photo",
    "video": "video",
    "videos": "video",
    "voice": "voice_message",
    "voice_messages": "voice_message",
    "audio": "audio_file",
    "audio_files": "audio_file",
    "sticker": "sticker",
    "stickers": "sticker",
    "animation": "animation",
    "animations": "animation",
    "file": "file",
    "files": "file",
    "documents": "file",
}


def parse_size(value: str) -> int:
    """Parse a non-negative byte count such as ``500M`` or ``4GiB``."""
    raw = value.strip().upper().replace(" ", "")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(B|K|KB|KIB|M|MB|MIB|G|GB|GIB|T|TB|TIB)?", raw)
    if not match:
        raise argparse.ArgumentTypeError("size must look like 0, 500M, 4G, 4GiB, or 1048576B")
    number = float(match.group(1))
    unit = match.group(2) or "B"
    multiplier = {
        "B": 1,
        "K": 1024,
        "KB": 1000,
        "M": 1024**2,
        "MB": 1000**2,
        "G": 1024**3,
        "GB": 1000**3,
        "T": 1024**4,
        "TB": 1000**4,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "TIB": 1024**4,
    }[unit]
    return int(number * multiplier)


def parse_media_selection(value: str) -> set[str]:
    raw = value.strip().lower()
    if not raw or raw == "all":
        return set(MEDIA_TYPES)
    selected: set[str] = set()
    unknown: list[str] = []
    for item in raw.split(","):
        item = item.strip()
        if item == "all":
            selected.update(MEDIA_TYPES)
        elif item in MEDIA_ALIASES:
            selected.add(MEDIA_ALIASES[item])
        elif item:
            unknown.append(item)
    if unknown:
        aliases = ", ".join(sorted({"all", *MEDIA_ALIASES}))
        raise ExportError(f"Unknown media type(s): {', '.join(unknown)}. Use: {aliases}")
    return selected


def safe_component(value: str, fallback: str = "chat") -> str:
    value = re.sub(r"[\x00-\x1f\x7f]", "", value).strip()
    value = value.replace("/", "_").replace("\\", "_")
    value = re.sub(r"\s+", " ", value).strip(" .")
    if not value or value in {".", ".."}:
        value = fallback
    return value[:120]


def parse_env_file(path: Path) -> dict[str, str]:
    """Read a dotenv-style file without logging secret values."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    try:
        mode = path.stat().st_mode & 0o777
        if mode & 0o077:
            import sys

            print(
                f"warning: credentials file {path} is mode {mode:03o}; run chmod 600 on it", file=sys.stderr
            )
    except OSError:
        pass
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        normalized_key = key.strip()
        values[normalized_key] = value
        if normalized_key.upper() in {"TG_API_ID", "TG_API_HASH", "TELEGRAM_API_ID", "TELEGRAM_API_HASH"}:
            # Keep secrets out of later diagnostic tracebacks even when they
            # came from a dotenv file rather than the process environment.
            from .diagnostics import register_secret

            register_secret(value)
    return values


def credentials(config_path: Path) -> tuple[int, str]:
    values = parse_env_file(config_path)
    api_id_raw = os.environ.get("TG_API_ID", values.get("TG_API_ID", ""))
    api_hash = os.environ.get("TG_API_HASH", values.get("TG_API_HASH", ""))
    from .diagnostics import register_secret

    register_secret(api_id_raw)
    register_secret(api_hash)
    if not api_id_raw or not api_hash:
        raise ExportError(
            f"Missing TG_API_ID/TG_API_HASH. Run `configure --config {config_path}` or set both environment variables."
        )
    try:
        api_id = int(api_id_raw)
    except ValueError as exc:
        raise ExportError("TG_API_ID must be a numeric Telegram API ID") from exc
    if api_id <= 0 or not re.fullmatch(r"[0-9a-fA-F]{16,}", api_hash):
        raise ExportError("TG_API_ID/TG_API_HASH do not look valid")
    return api_id, api_hash


def ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass


def telethon_session_file(path: Path) -> Path:
    """Return the actual SQLite session filename used by Telethon."""
    return path if path.name.endswith(".session") else Path(f"{path}.session")


def secure_session_file(path: Path) -> None:
    session_file = telethon_session_file(path)
    for candidate in (session_file, Path(f"{session_file}-journal")):
        if candidate.exists():
            try:
                candidate.chmod(0o600)
            except OSError:
                pass


def write_credentials(config_path: Path, api_id: str, api_hash: str) -> None:
    if not api_id.isdigit() or not re.fullmatch(r"[0-9a-fA-F]{16,}", api_hash):
        raise ExportError("The API ID must be numeric and the API hash must be hexadecimal")
    ensure_private_dir(config_path.parent)
    existing_values = parse_env_file(config_path) if config_path.exists() else {}
    if config_path.exists() and input(f"Overwrite {config_path}? [y/N] ").strip().lower() != "y":
        raise ExportError("Refusing to overwrite existing credentials")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{config_path.name}.", dir=config_path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"TG_API_ID={api_id}\nTG_API_HASH={api_hash}\n")
            for key in ("TG_MEDIA", "TG_MAX_FILE_SIZE"):
                if key in existing_values:
                    handle.write(f"{key}={existing_values[key]}\n")
        os.replace(tmp_name, config_path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    print(f"Saved credentials to {config_path} with mode 600.")


__all__ = [
    "PROJECT_ROOT",
    "SCRIPT_DIR",
    "DEFAULT_DB",
    "DEFAULT_CONFIG",
    "DEFAULT_SESSION",
    "DEFAULT_OUTPUT",
    "TARGETS_TABLE",
    "EXPORTS_TABLE",
    "RUNS_TABLE",
    "RUN_MESSAGES_TABLE",
    "RUN_ARCHIVE_TABLE",
    "RUN_ATTEMPTS_TABLE",
    "TARGET_CHAT_LINKS_TABLE",
    "DIALOGS_TABLE",
    "PURGES_TABLE",
    "BLACKLIST_TABLE",
    "DIAGNOSTIC_EVENTS_TABLE",
    "MEDIA_TYPES",
    "MEDIA_ALIASES",
    "database_mtime_ns",
    "default_database_path",
    "parse_size",
    "parse_media_selection",
    "safe_component",
    "parse_env_file",
    "credentials",
    "ensure_private_dir",
    "telethon_session_file",
    "secure_session_file",
    "write_credentials",
]
