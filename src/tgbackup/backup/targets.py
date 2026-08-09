"""Stable target identity and filesystem placement helpers.

These helpers are intentionally independent of the exporter orchestration so
the GUI, scheduled jobs, and tests can reason about a target without opening a
Telegram connection.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

from ..config import ensure_private_dir, safe_component
from ..errors import ExportError
from ..models import Target


def path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def target_output_dir(output_root: Path, target: Target) -> Path:
    """Resolve a stable per-peer directory and write its identity marker."""
    ensure_private_dir(output_root)
    configured = Path(target.output_dir).expanduser().resolve() if target.output_dir else None
    if configured is not None and path_is_under(configured, output_root):
        base = configured
    else:
        base = output_root / safe_component(target.title)
    marker = base / ".tgbackman_target.json"
    if marker.is_file():
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
            if str(existing.get("chat_id")) != target.chat_id:
                base = output_root / f"{safe_component(target.title)}__{safe_component(target.chat_id)}"
        except (OSError, ValueError, TypeError):
            base = output_root / f"{safe_component(target.title)}__{safe_component(target.chat_id)}"
    elif base.is_dir() and any(base.iterdir()):
        matches = False
        for state in base.glob(".partial-*/.partial_state.json"):
            try:
                matches = json.loads(state.read_text(encoding="utf-8")).get("target_key") == target.target_key
            except (OSError, ValueError, TypeError):
                continue
            if matches:
                break
        if not matches:
            base = output_root / f"{safe_component(target.title)}__{safe_component(target.chat_id)}"
    ensure_private_dir(base)
    # The directory may have changed above after the configured/title-based
    # directory was found to be occupied.  Recompute the marker path only
    # after the final directory has been selected; otherwise a marker can be
    # written into an unrelated directory and later authorize its purge.
    marker = base / ".tgbackman_target.json"
    if not marker.exists():
        payload = json.dumps(
            {"chat_id": target.chat_id, "target_key": target.target_key, "title": target.title},
            indent=2,
        ) + "\n"
        fd, temporary = tempfile.mkstemp(prefix=".tgbackman_target.", dir=base)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, marker)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)
    return base


def direct_target_output_dir(path: Path, target: Target, *, write_marker: bool) -> Path:
    if not path.is_dir():
        raise ExportError(f"--chat-output-dir must be an existing directory: {path}")
    marker = path / ".tgbackman_target.json"
    if marker.exists():
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise ExportError(f"invalid target marker in {path}: {exc}") from exc
        if str(existing.get("chat_id")) != target.chat_id:
            raise ExportError(f"{path} is already marked for chat_id={existing.get('chat_id')!r}, not {target.chat_id!r}")
    elif write_marker:
        payload = json.dumps(
            {"chat_id": target.chat_id, "target_key": target.target_key, "title": target.title},
            indent=2,
        ) + "\n"
        fd, temporary = tempfile.mkstemp(prefix=".tgbackman_target.", dir=path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, marker)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)
    return path


def target_key(source_name: str, peer_kind: str, peer_id: int) -> str:
    digest = hashlib.sha1(f"{source_name.strip().casefold()}\0{peer_kind}\0{peer_id}".encode()).hexdigest()[:12]
    return f"{safe_component(source_name).lower().replace(' ', '_')}-{digest}"


def normalized_chat_name(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def generated_chat_id(peer_kind: str, peer_id: int) -> str:
    prefix = "dialog" if peer_kind == "user" else "group" if peer_kind == "group" else "channel"
    return f"{prefix}_{peer_id}"


def database_peer_hint(chat_id: str) -> Optional[tuple[frozenset[str], int]]:
    match = re.fullmatch(r"(dialog|user|group|channel)_(-?\d+)", chat_id.strip(), re.IGNORECASE)
    if not match:
        return None
    prefix, raw_id = match.groups()
    kinds = {"dialog": frozenset({"user"}), "user": frozenset({"user"}), "group": frozenset({"group", "channel"}), "channel": frozenset({"channel"})}
    return kinds[prefix.casefold()], abs(int(raw_id))


def entity_description(entity: Any) -> tuple[str, str, Optional[str], int, Optional[int], str]:
    from telethon.tl.types import Channel, Chat, User

    if isinstance(entity, User):
        kind = "user"
        title = " ".join(x for x in (entity.first_name, entity.last_name) if x).strip() or entity.username or str(entity.id)
        access_hash = int(entity.access_hash) if entity.access_hash is not None else None
    elif isinstance(entity, Channel):
        kind, title = "channel", entity.title or str(entity.id)
        access_hash = int(entity.access_hash) if entity.access_hash is not None else None
    elif isinstance(entity, Chat):
        kind, title, access_hash = "group", entity.title or str(entity.id), None
    else:
        raise ExportError(f"Unsupported Telegram entity type: {type(entity).__name__}")
    return kind, title, getattr(entity, "username", None), int(entity.id), access_hash, entity.__class__.__name__
