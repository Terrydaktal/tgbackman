"""Reorganise legacy media into one stable, identity-named directory per chat.

The canonical database is kept separate from the filesystem layout: the
database remains the source of truth for messages, while each chat gets a
stable directory containing ``media/<type>/...``.  The apply path uses
``cp --reflink=always`` and strict size/SHA-256 verification.  It never falls
back to an ordinary copy, never deletes the source tree, and only updates the
database after every planned file has been copied and verified.

The command is deliberately dry-run by default.  A manifest records every
source and destination path, making interrupted copies resumable without
touching Telegram or downloading anything again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .config import safe_component
from .db.schema import setup_database

LAYOUT_DESCRIPTION = """Current layouts accepted:
  * canonical API output: one stable chat directory with media/<type>/ files;
  * legacy exports: chat/date-range directories containing JSON/HTML and media/.
  The database may still contain relative paths from an older range directory.

New layout:
  <destination-root>/<safe title>__<peer kind>_<peer id>/
    .tgbackman_chat.json
    media/<type>/<filename>

Each visible per-chat media file is a Btrfs reflink.  Equal content in two
chats therefore remains identifiable by its path while sharing physical
extents.  The SQLite database stores paths relative to the new chat directory.
"""


class ReorganisationError(RuntimeError):
    """A migration cannot be made safely."""


@dataclass
class ChatPlan:
    chat_id: str
    title: str
    peer_kind: str | None
    peer_id: int | None
    destination: str
    target_key: str | None = None
    messages: int = 0


@dataclass
class MediaPlan:
    chat_id: str
    message_id: int
    source: str
    relative: str
    size: int
    sha256: str
    media_type: str
    status: str = "pending"


@dataclass
class MigrationPlan:
    db: str
    source_root: str
    destination_root: str
    chats: list[ChatPlan] = field(default_factory=list)
    media: list[MediaPlan] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unsafe: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        return {
            "version": 1,
            "created_unix": int(time.time()),
            "db": self.db,
            "source_root": self.source_root,
            "destination_root": self.destination_root,
            "db_committed": False,
            "chats": [chat.__dict__ for chat in self.chats],
            "media": [item.__dict__ for item in self.media],
            "missing": self.missing,
            "unsafe": self.unsafe,
            "mismatched": self.mismatched,
        }


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _has_symlink_component(path: Path, root: Path) -> bool:
    if not _within(path, root):
        return True
    current = root
    if current.is_symlink():
        return True
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _filesystem(path: Path) -> str:
    return subprocess.run(
        ["stat", "-f", "-c", "%T", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()


def _identity_dir(title: str, peer_kind: str | None, peer_id: int | None, chat_id: str) -> str:
    title_part = safe_component(title or chat_id)
    if peer_kind and peer_id is not None:
        identity = f"{safe_component(peer_kind)}_{peer_id}"
    else:
        identity = f"chat_{hashlib.sha1(chat_id.encode()).hexdigest()[:12]}"
    return f"{title_part}__{identity}"


def _path_parts(value: str) -> tuple[str, ...]:
    return tuple(part for part in PurePosixPath(value.replace("\\", "/")).parts if part not in {".", "..", "/"})


def _source_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        index.setdefault(path.name, []).append(path)
    return index


def _matches_suffix(path: Path, root: Path, parts: tuple[str, ...]) -> bool:
    try:
        relative = path.relative_to(root).parts
    except ValueError:
        return False
    return len(relative) >= len(parts) and relative[-len(parts) :] == parts


def _resolve_source(
    media_path: str,
    backup_path: str | None,
    source_root: Path,
    index: dict[str, list[Path]],
    expected_size: int | None,
    expected_hash: str | None,
) -> tuple[Path | None, str | None]:
    value = media_path.strip()
    if not value or "://" in value:
        return None, "non-local media path"
    raw = Path(value).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        if backup_path:
            candidates.append(Path(backup_path).expanduser() / raw)
        candidates.append(source_root / raw)
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if not resolved.is_file() or not _within(resolved, source_root):
            continue
        if _has_symlink_component(candidate, source_root):
            return None, f"symbolic link in media path: {candidate}"
        return resolved, None

    parts = _path_parts(value)
    if parts:
        matches = [path for path in index.get(parts[-1], []) if _matches_suffix(path, source_root, parts)]
        if expected_size is not None:
            matches = [path for path in matches if path.stat().st_size == int(expected_size)]
        if len(matches) == 1:
            return matches[0].resolve(), None
        if len(matches) > 1 and expected_hash:
            hashed = [path for path in matches if _hash_file(path)[1] == expected_hash]
            if len(hashed) == 1:
                return hashed[0].resolve(), None
        if len(matches) > 1:
            return None, f"ambiguous media path ({len(matches)} matches): {value}"
    return None, "media file not found"


def _unique_relative(
    used: dict[tuple[str, str], tuple[int, str]],
    chat_id: str,
    media_type: str,
    basename: str,
    message_id: int,
    digest: str,
) -> str:
    directory = f"media/{safe_component(media_type, 'file')}"
    clean = safe_component(Path(basename).name, f"{message_id}.bin")
    candidate = f"{directory}/{clean}"
    key = (chat_id, candidate)
    previous = used.get(key)
    if previous is None or previous == (message_id, digest):
        used[key] = (message_id, digest)
        return candidate
    stem = Path(clean).stem or str(message_id)
    suffix = Path(clean).suffix
    for prefix in (str(message_id), digest[:12]):
        candidate = f"{directory}/{safe_component(prefix + '_' + stem)}{suffix}"
        key = (chat_id, candidate)
        previous = used.get(key)
        if previous is None or previous == (message_id, digest):
            used[key] = (message_id, digest)
            return candidate
    raise ReorganisationError(f"unable to choose a unique destination for {chat_id}:{message_id}")


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def build_plan(db: Path, source_root: Path, destination_root: Path, chat_ids: set[str] | None = None) -> MigrationPlan:
    source_root = source_root.expanduser().resolve(strict=True)
    if not source_root.is_dir():
        raise ReorganisationError(f"source root is not a directory: {source_root}")
    destination_root = destination_root.expanduser().resolve()
    if destination_root == source_root or _within(destination_root, source_root):
        raise ReorganisationError("destination must be separate from and outside source root")
    if destination_root.exists() and not destination_root.is_dir():
        raise ReorganisationError(f"destination is not a directory: {destination_root}")
    db = db.expanduser().resolve(strict=True)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        target_join = "LEFT JOIN telegram_backup_targets AS t ON t.chat_id = c.chat_id" if _table_exists(conn, "telegram_backup_targets") else ""
        target_fields = (
            "t.peer_kind, t.peer_id, t.target_key"
            if target_join
            else "NULL AS peer_kind, NULL AS peer_id, NULL AS target_key"
        )
        rows = conn.execute(
            f"""SELECT c.chat_id, c.chat_name, c.backup_path,
                       {target_fields}
                FROM chats AS c {target_join}
                ORDER BY c.chat_id"""
        ).fetchall()
        if chat_ids is not None:
            rows = [row for row in rows if str(row["chat_id"]) in chat_ids]
        plan = MigrationPlan(str(db), str(source_root), str(destination_root))
        for row in rows:
            destination = destination_root / _identity_dir(
                str(row["chat_name"] or row["chat_id"]),
                str(row["peer_kind"]) if row["peer_kind"] is not None else None,
                int(row["peer_id"]) if row["peer_id"] is not None else None,
                str(row["chat_id"]),
            )
            plan.chats.append(ChatPlan(
                chat_id=str(row["chat_id"]), title=str(row["chat_name"] or row["chat_id"]),
                peer_kind=str(row["peer_kind"]) if row["peer_kind"] is not None else None,
                peer_id=int(row["peer_id"]) if row["peer_id"] is not None else None,
                destination=str(destination),
                target_key=str(row["target_key"]) if row["target_key"] is not None else None,
            ))
        if not rows:
            return plan
        chats_by_id = {chat.chat_id: chat for chat in plan.chats}
        source_index = _source_index(source_root)
        used: dict[tuple[str, str], tuple[int, str]] = {}
        for row in rows:
            chat_id = str(row["chat_id"])
            media_rows = conn.execute(
                """SELECT message_id, media_type, media_path, media_size, media_sha256
                   FROM messages
                   WHERE chat_id=? AND COALESCE(is_deleted, 0)=0
                     AND media_path IS NOT NULL AND media_path != ''
                   ORDER BY message_id""", (chat_id,)
            ).fetchall()
            chat = chats_by_id[chat_id]
            chat.messages = len(media_rows)
            for media in media_rows:
                message_id = int(media["message_id"])
                source, error = _resolve_source(
                    str(media["media_path"]),
                    str(row["backup_path"]) if row["backup_path"] else None,
                    source_root, source_index,
                    int(media["media_size"]) if media["media_size"] is not None else None,
                    str(media["media_sha256"]) if media["media_sha256"] else None,
                )
                if source is None:
                    plan.missing.append(f"{chat_id}:{message_id}: {error}")
                    continue
                size, digest = _hash_file(source)
                expected_size = int(media["media_size"]) if media["media_size"] is not None else None
                expected_hash = str(media["media_sha256"]) if media["media_sha256"] else None
                if expected_size is not None and expected_size != size:
                    plan.mismatched.append(f"{chat_id}:{message_id}: expected size {expected_size}, got {size}")
                    continue
                if expected_hash and expected_hash.casefold() != digest.casefold():
                    plan.mismatched.append(f"{chat_id}:{message_id}: expected SHA-256 {expected_hash}, got {digest}")
                    continue
                media_type = safe_component(str(media["media_type"] or "file"), "file")
                relative = _unique_relative(used, chat_id, media_type, source.name, message_id, digest)
                plan.media.append(MediaPlan(chat_id, message_id, str(source), relative, size, digest, media_type))
    finally:
        conn.close()
    return plan


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _write_chat_marker(chat: ChatPlan) -> None:
    destination = Path(chat.destination)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1, "chat_id": chat.chat_id, "target_key": chat.target_key,
        "title": chat.title, "peer_kind": chat.peer_kind, "peer_id": chat.peer_id,
    }
    for marker in (destination / ".tgbackman_chat.json", destination / ".tgbackman_target.json"):
        if marker.is_symlink():
            raise ReorganisationError(f"refusing to follow symbolic-link marker: {marker}")
        if marker.exists():
            try:
                existing = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ReorganisationError(f"invalid chat marker: {marker}: {exc}") from exc
            if str(existing.get("chat_id")) != chat.chat_id:
                raise ReorganisationError(f"destination is marked for another chat: {marker}")
        else:
            _write_json(marker, payload)


def _copy_reflink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    os.unlink(temporary)
    try:
        subprocess.run(
            ["cp", "--reflink=always", "--preserve=mode,timestamps", str(source), temporary],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _plan_from_manifest(path: Path) -> MigrationPlan:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReorganisationError(f"cannot read manifest {path}: {exc}") from exc
    if payload.get("version") != 1:
        raise ReorganisationError(f"unsupported reorganisation manifest version: {payload.get('version')!r}")
    plan = MigrationPlan(
        str(payload["db"]), str(payload["source_root"]), str(payload["destination_root"]),
        chats=[ChatPlan(**chat) for chat in payload.get("chats", [])],
        media=[MediaPlan(**media) for media in payload.get("media", [])],
        missing=list(payload.get("missing", [])),
        unsafe=list(payload.get("unsafe", [])),
        mismatched=list(payload.get("mismatched", [])),
    )
    return plan


def apply_plan(plan: MigrationPlan, manifest_path: Path, *, resume: bool = False, mount_point: Path | None = None) -> None:
    source_root = Path(plan.source_root)
    destination_root = Path(plan.destination_root)
    if mount_point is not None and not os.path.ismount(mount_point):
        raise ReorganisationError(f"required mount point is not mounted: {mount_point}")
    if mount_point is None and destination_root.parts[:2] == ("/", "media"):
        raise ReorganisationError("--apply for a /media destination requires --mount-point")
    if not destination_root.exists() and not destination_root.parent.is_dir():
        raise ReorganisationError(f"destination parent does not exist: {destination_root.parent}")
    if destination_root.exists() and any(destination_root.iterdir()) and not resume:
        raise ReorganisationError("destination is non-empty; pass --resume with the original manifest")
    source_fs = _filesystem(source_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    destination_fs = _filesystem(destination_root)
    if source_fs != "btrfs" or destination_fs != "btrfs":
        raise ReorganisationError(
            f"--apply requires Btrfs at both roots (source={source_fs}, destination={destination_fs})"
        )
    payload = plan.as_json()
    _write_json(manifest_path, payload)
    destination_by_chat = {chat.chat_id: Path(chat.destination) for chat in plan.chats}
    canonical_by_content: dict[tuple[int, str], Path] = {}
    for chat in plan.chats:
        if _has_symlink_component(destination_by_chat[chat.chat_id], destination_root):
            raise ReorganisationError(f"refusing symbolic-link chat destination: {destination_by_chat[chat.chat_id]}")
        _write_chat_marker(chat)
    for index, item in enumerate(plan.media, start=1):
        source = Path(item.source)
        destination = destination_by_chat[item.chat_id] / item.relative
        if _has_symlink_component(destination, destination_root) or destination.is_symlink():
            raise ReorganisationError(f"refusing symbolic-link destination: {destination}")
        content_key = (item.size, item.sha256)
        if destination.exists():
            size, digest = _hash_file(destination)
            if (size, digest) != (item.size, item.sha256):
                raise ReorganisationError(f"destination collision with different content: {destination}")
            item.status = "reused"
            canonical_by_content.setdefault(content_key, destination)
        else:
            _copy_reflink(canonical_by_content.get(content_key, source), destination)
            size, digest = _hash_file(destination)
            if (size, digest) != (item.size, item.sha256):
                raise ReorganisationError(f"reflink verification failed: {destination}")
            item.status = "copied"
            canonical_by_content.setdefault(content_key, destination)
        if index % 100 == 0 or index == len(plan.media):
            payload["media"] = [media.__dict__ for media in plan.media]
            _write_json(manifest_path, payload)
            print(f"media: {index:,}/{len(plan.media):,} verified")

    conn = setup_database(plan.db)
    try:
        conn.execute("BEGIN IMMEDIATE")
        chats_by_id = {chat.chat_id: chat for chat in plan.chats}
        for chat in plan.chats:
            conn.execute("UPDATE chats SET backup_path=? WHERE chat_id=?", (chat.destination, chat.chat_id))
            if chat.target_key:
                conn.execute(
                    """UPDATE telegram_backup_targets
                       SET output_dir=?, updated_unix=? WHERE target_key=?""",
                    (chat.destination, int(time.time()), chat.target_key),
                )
        for item in plan.media:
            if item.chat_id not in chats_by_id:
                raise ReorganisationError(f"media record has no planned chat: {item.chat_id}")
            relative = item.relative
            updated = conn.execute(
                """UPDATE messages SET media_path=?, media_size=?, media_sha256=?, media_status='downloaded'
                   WHERE chat_id=? AND message_id=?""",
                (relative, item.size, item.sha256, item.chat_id, item.message_id),
            )
            if updated.rowcount != 1:
                raise ReorganisationError(f"message disappeared while updating database: {item.chat_id}:{item.message_id}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    payload["media"] = [media.__dict__ for media in plan.media]
    payload["db_committed"] = True
    _write_json(manifest_path, payload)


def _print_summary(plan: MigrationPlan) -> None:
    print(LAYOUT_DESCRIPTION.strip())
    print(f"Chats: {len(plan.chats):,}; media records: {len(plan.media):,}")
    print(f"Missing: {len(plan.missing):,}; metadata mismatches: {len(plan.mismatched):,}; unsafe: {len(plan.unsafe):,}")
    for error in (*plan.missing[:20], *plan.mismatched[:20], *plan.unsafe[:20]):
        print(f"ERROR: {error}")
    if len(plan.missing) + len(plan.mismatched) + len(plan.unsafe) > 40:
        print("(only the first 20 entries of each error category are shown)")
    for chat in plan.chats:
        print(f"  {chat.chat_id}: {chat.title} -> {chat.destination} ({chat.messages:,} media records)")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan or apply a safe per-chat Btrfs-reflink media reorganisation")
    parser.add_argument("--describe", action="store_true", help="print accepted and new layouts without opening a database")
    parser.add_argument("--db", type=Path, help="canonical SQLite database")
    parser.add_argument("--source-root", type=Path, help="existing legacy/canonical backup root")
    parser.add_argument("--destination-root", type=Path, help="new per-chat root; must be outside source-root")
    parser.add_argument("--chat-id", action="append", dest="chat_ids", help="limit migration; may be repeated")
    parser.add_argument("--manifest", type=Path, help="manifest path (required for explicit resume)")
    parser.add_argument("--apply", action="store_true", help="copy reflinks and update SQLite; default is report-only")
    parser.add_argument("--resume", action="store_true", help="resume a previous apply using its manifest")
    parser.add_argument("--mount-point", type=Path, help="refuse apply unless this mount point is currently mounted")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.describe:
        print(LAYOUT_DESCRIPTION.strip())
        return 0
    if not args.db or not args.source_root or not args.destination_root:
        parser.error("--db, --source-root, and --destination-root are required unless --describe is used")
    if args.resume and not args.manifest:
        parser.error("--resume requires --manifest")
    try:
        if args.mount_point is not None and not os.path.ismount(args.mount_point):
            raise ReorganisationError(f"required mount point is not mounted: {args.mount_point}")
        if args.resume:
            plan = _plan_from_manifest(args.manifest)
            if plan.db != str(args.db.expanduser().resolve()) or plan.source_root != str(args.source_root.expanduser().resolve()) or plan.destination_root != str(args.destination_root.expanduser().resolve()):
                raise ReorganisationError("manifest database/source/destination does not match the command line")
        else:
            plan = build_plan(args.db, args.source_root, args.destination_root, set(args.chat_ids) if args.chat_ids else None)
        _print_summary(plan)
        if plan.missing or plan.mismatched or plan.unsafe:
            print("No files were changed; resolve the reported integrity/path errors first.")
            return 2
        if not args.apply:
            if args.manifest:
                _write_json(args.manifest, plan.as_json())
                print(f"Plan manifest written: {args.manifest}")
            print("Dry run only: no files or database rows were changed.")
            return 0
        manifest = args.manifest or (args.destination_root / ".tgbackman-reorganize.json")
        if manifest.exists() and not args.resume:
            raise ReorganisationError(f"manifest already exists; choose a new path or use --resume: {manifest}")
        apply_plan(plan, manifest, resume=args.resume, mount_point=args.mount_point)
        print(f"Reorganisation complete; manifest: {manifest}")
        print("The source tree was retained. Verify the database and media before deleting old files.")
        return 0
    except (OSError, sqlite3.Error, ReorganisationError, subprocess.CalledProcessError) as exc:
        print(f"Reorganisation failed: {exc}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
