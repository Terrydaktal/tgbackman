"""Bounded, privacy-aware diagnostics for the backup services.

Diagnostics are deliberately separate from user-facing progress output.  The
default path records only lifecycle/failure information and never message
bodies or credentials.  A small JSON snapshot is written atomically so it can
be collected while a long-running export is still active.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    import resource
except ImportError:  # pragma: no cover - Windows has no resource module.
    resource = None  # type: ignore[assignment]

DIAGNOSTIC_SCHEMA_VERSION = 1
MAX_FIELD_LENGTH = 512
MAX_TRACEBACK_LENGTH = 12_000
MAX_EVENT_FILE_BYTES = 4 * 1024 * 1024
MAX_TERMINAL_OPERATION_FILES = 32
_BUILD_ID: Optional[dict[str, Any]] = None
_SECRET_VALUES: set[str] = set()


def _state_dir() -> Path:
    root = os.environ.get("XDG_STATE_HOME", "").strip()
    return (Path(root).expanduser() if root else Path.home() / ".local" / "state") / "tgbackman"


def build_identity() -> dict[str, Any]:
    """Return a cached identity suitable for joining logs to a shipped build."""
    global _BUILD_ID
    if _BUILD_ID is not None:
        return dict(_BUILD_ID)
    revision = os.environ.get("TGBACKMAN_BUILD_REVISION", "").strip()
    repository: Optional[Path] = None
    if not revision:
        repository = Path(__file__).resolve().parents[2]
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "--short=12", "HEAD"],
                cwd=repository,
                capture_output=True,
                text=True,
                timeout=0.25,
                check=False,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            revision = "unknown"
    if repository is None:
        repository = Path(__file__).resolve().parents[2]
    dirty = False
    status_digest = "unknown"
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository,
            capture_output=True,
            text=True,
            timeout=0.5,
            check=False,
        )
        status_text = status.stdout if status.returncode == 0 else ""
        dirty = bool(status_text.strip())
        if status.returncode == 0:
            status_digest = hashlib.sha256(status_text.encode("utf-8")).hexdigest()[:16]
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        version = importlib.metadata.version("tgbackman")
    except importlib.metadata.PackageNotFoundError:
        version = "0.1.0"
    try:
        telethon_version = importlib.metadata.version("Telethon")
    except importlib.metadata.PackageNotFoundError:
        telethon_version = "unknown"
    _BUILD_ID = {
        "service": "tgbackup",
        "version": version,
        "revision": revision or "unknown",
        "dirty": dirty,
        "working_tree_status": status_digest,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(aliased=True),
        "telethon": telethon_version,
    }
    return dict(_BUILD_ID)


def _safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _safe_value(item)
            for key, item in list(value.items())[:40]
            if not _sensitive_key(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in list(value)[:40]]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        text = str(value) if not isinstance(value, (str, type(None))) else value
        return text if text is None or len(text) <= MAX_FIELD_LENGTH else text[:MAX_FIELD_LENGTH] + "…"
    return str(value)[:MAX_FIELD_LENGTH]


def _redact_text(value: str) -> str:
    """Remove credential values that may have leaked through an exception."""
    redacted = value
    secrets = set(_SECRET_VALUES)
    for key in ("TG_API_HASH", "TG_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_API_ID"):
        secret = os.environ.get(key, "")
        if secret and len(secret) >= 4:
            secrets.add(secret)
    for secret in sorted(secrets, key=len, reverse=True):
        if len(secret) >= 4:
            redacted = redacted.replace(secret, "<redacted>")
    return redacted[:MAX_TRACEBACK_LENGTH]


def _sensitive_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    return normalized in {
        "api_hash",
        "api_id",
        "password",
        "token",
        "secret",
        "phone",
        "message",
        "text",
        "body",
        "credential",
        "credentials",
    } or any(part in normalized for part in ("api_hash", "password", "access_token", "session_string"))


def register_secret(value: Any) -> None:
    """Register a configuration secret for later traceback/event redaction."""
    text = str(value or "")
    if len(text) >= 4:
        _SECRET_VALUES.add(text)


def _process_resources() -> dict[str, int]:
    if resource is None:
        return {"max_rss_kib": 0, "threads": threading.active_count()}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports KiB; macOS reports bytes. Keep the field explicitly named
    # and normalize the common non-Linux case for portable snapshots.
    rss = int(usage.ru_maxrss)
    if sys.platform == "darwin":
        rss //= 1024
    return {"max_rss_kib": rss, "threads": threading.active_count()}


def exception_details(
    exc: BaseException, *, phase: str, fields: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    """Serialize a bounded exception chain without exposing credential values."""
    chain: list[dict[str, str]] = []
    current: Optional[BaseException] = exc
    while current is not None and len(chain) < 8:
        chain.append(
            {"type": type(current).__name__, "message": _redact_text(str(current)[:MAX_FIELD_LENGTH])}
        )
        current = current.__cause__ or current.__context__
    return {
        "schema": DIAGNOSTIC_SCHEMA_VERSION,
        "phase": phase,
        "error_code": str(getattr(exc, "code", type(exc).__name__)).replace(" ", "_").lower(),
        "exception_type": type(exc).__name__,
        "message": _redact_text(str(exc)[:MAX_FIELD_LENGTH]),
        "chain": chain,
        "traceback": _redact_text("".join(traceback.format_exception(exc))[-MAX_TRACEBACK_LENGTH:]),
        "fields": _safe_value(fields or {}),
        "build": build_identity(),
    }


@dataclass
class DiagnosticEvent:
    event: str
    component: str
    level: str = "info"
    operation_id: Optional[str] = None
    run_key: Optional[str] = None
    attempt_key: Optional[str] = None
    fields: dict[str, Any] = field(default_factory=dict)
    wall_unix: int = field(default_factory=lambda: int(time.time()))
    monotonic_ms: int = field(default_factory=lambda: int(time.monotonic() * 1000))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": DIAGNOSTIC_SCHEMA_VERSION,
            "event": self.event,
            "component": self.component,
            "level": self.level,
            "operation_id": self.operation_id,
            "run_key": self.run_key,
            "attempt_key": self.attempt_key,
            "wall_unix": self.wall_unix,
            "monotonic_ms": self.monotonic_ms,
            "build": build_identity(),
            "fields": _safe_value(self.fields),
        }


class DiagnosticRecorder:
    """Bounded JSONL event sink used by default for failure evidence.

    ``path`` remains injectable for tests and deployments that want a separate
    stream.  The default is a private, rotating state file so a handled CLI
    failure is not lost merely because the caller did not request verbose
    diagnostics.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        configured = os.environ.get("TGBACKMAN_DIAGNOSTICS_FILE", "").strip()
        self.path = path or (Path(configured).expanduser() if configured else _state_dir() / "events.jsonl")
        self._lock = threading.Lock()

    def emit(self, event: DiagnosticEvent) -> None:
        if self.path is None:
            return
        line = json.dumps(event.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with contextlib.suppress(OSError):
                    self.path.parent.chmod(0o700)
                if self.path.exists() and self.path.stat().st_size >= MAX_EVENT_FILE_BYTES:
                    rotated = self.path.with_name(self.path.name + ".1")
                    with contextlib.suppress(OSError):
                        rotated.unlink()
                    os.replace(self.path, rotated)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                with contextlib.suppress(OSError):
                    self.path.chmod(0o600)
            except OSError:
                # Diagnostics must never break or delay the backup operation.
                return


class OperationRegistry:
    """Small atomic lifecycle snapshot for an active CLI process."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or (_state_dir() / "active-operation.json")
        self.operation_id = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "schema": DIAGNOSTIC_SCHEMA_VERSION,
            "operation_id": self.operation_id,
            "service": "tgbackup",
            "pid": os.getpid(),
            "started_unix": int(time.time()),
            "status": "starting",
            "build": build_identity(),
        }

    def update(self, status: str, **fields: Any) -> None:
        with self._lock:
            self._state.update(
                {
                    "status": status,
                    "updated_unix": int(time.time()),
                    "resources": _process_resources(),
                    **_safe_value(fields),
                }
            )
            self._write()

    def finish(self, status: str, **fields: Any) -> None:
        self.update(status, **fields)
        # Keep a bounded terminal record.  The active snapshot is intentionally
        # removed so it remains a truthful indication of a live process, while
        # the terminal copy gives post-mortem tooling a durable operation status.
        terminal_dir = self.path.parent / "operations"
        terminal = terminal_dir / f"{self.operation_id}.json"
        with self._lock:
            try:
                terminal_dir.mkdir(parents=True, exist_ok=True)
                with contextlib.suppress(OSError):
                    terminal_dir.chmod(0o700)
                self._write_to(terminal)
                for candidate in sorted(
                    terminal_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True
                )[MAX_TERMINAL_OPERATION_FILES:]:
                    with contextlib.suppress(OSError):
                        candidate.unlink()
            except OSError:
                pass
        with contextlib.suppress(OSError):
            self.path.unlink()

    def _write(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                self.path.parent.chmod(0o700)
            self._write_to(self.path)
        except OSError:
            return

    def _write_to(self, destination: Path) -> None:
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(self._state, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            os.replace(temporary, destination)
        finally:
            with contextlib.suppress(OSError):
                temporary.unlink()


def snapshot(path: Optional[Path] = None) -> dict[str, Any]:
    target = path or (_state_dir() / "active-operation.json")
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema": DIAGNOSTIC_SCHEMA_VERSION, "status": "idle", "build": build_identity()}


def _crash_path() -> Path:
    return _state_dir() / "crashes" / f"{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex[:12]}.json"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one diagnostic JSON document using same-directory rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


def install_runtime_hooks() -> None:
    """Install bounded uncaught-exception evidence at CLI startup."""
    try:
        import faulthandler

        faulthandler.enable(file=sys.stderr, all_threads=True)
    except (OSError, RuntimeError):
        pass
    previous = sys.excepthook

    def write_crash(exc: BaseException, *, phase: str, thread: Optional[str] = None) -> None:
        details = exception_details(exc, phase=phase, fields={"pid": os.getpid(), "thread": thread})
        path = _crash_path()
        try:
            _atomic_write_json(path, details)
        except (OSError, TypeError, ValueError):
            pass

    def hook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        write_crash(exc, phase="uncaught")
        previous(exc_type, exc, tb)

    sys.excepthook = hook
    previous_thread = threading.excepthook

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        write_crash(args.exc_value, phase="uncaught-thread", thread=getattr(args.thread, "name", None))
        previous_thread(args)

    threading.excepthook = thread_hook


__all__ = [
    "DIAGNOSTIC_SCHEMA_VERSION",
    "DiagnosticEvent",
    "DiagnosticRecorder",
    "OperationRegistry",
    "build_identity",
    "exception_details",
    "_atomic_write_json",
    "install_runtime_hooks",
    "register_secret",
    "snapshot",
]
