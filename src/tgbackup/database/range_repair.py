#!/usr/bin/env python3
"""
tgbackman-db-range-repair

Rename per-chat subfolders in a split multi-chat HTML/JSON/SQLite export output.

This fixes cases where the folder names are unknown/non-standard (e.g. ChatName/unknown__unknown/)
even though the backup data contains messages that allow deriving a real range.

Safety:
- Never deletes content.
- Default is dry-run; pass --apply to actually rename folders.
- Flat backup roots are wrapped only when --wrap-flat is explicitly supplied.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

VALID_RANGE_DIR_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z__\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$"
)

# Regexes for timestamp parsing (imported from rename_backup.py)
HTML_MSG_TS_RE = re.compile(
    r"(\d{2})\.(\d{2})\.(\d{4}) (\d{2}):(\d{2}):(\d{2})(?:\s+UTC([+-]\d{2}):(\d{2}))?"
)
HTML_TITLE_RE = re.compile(r'title="(\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}(?:\s+UTC[+-]\d{2}:\d{2})?)"')
JSON_DATE_RE = re.compile(r'"date":\s*"([^"]+)"')
JSON_DATE_UNIX_RE = re.compile(r'"date_unixtime":\s*(\d+)')

def parse_html_timestamp(ts_str: str) -> Optional[datetime]:
    try:
        m = HTML_MSG_TS_RE.match(ts_str.strip())
        if m:
            dd, mm, yyyy, hh, mi, ss, tz_h_s, tz_m_s = m.groups()
            tz_h = int(tz_h_s) if tz_h_s else 0
            tz_m = int(tz_m_s) if tz_m_s else 0
            offset = timezone(timedelta(hours=tz_h, minutes=(tz_m if tz_h >= 0 else -tz_m)))
            dt = datetime(int(yyyy), int(mm), int(dd), int(hh), int(mi), int(ss), tzinfo=offset)
            return dt.astimezone(timezone.utc)
    except Exception:
        pass
    return None

def format_utc_iso(dt: datetime) -> str:
    """Format datetime to YYYY-MM-DDTHH-MM-SSZ filesystem safe format."""
    return dt.strftime("%Y-%m-%dT%H-%M-%SZ")


def _matching_files(dir_path: str, predicate, *, recursive: bool) -> List[str]:
    matches: List[str] = []
    if recursive:
        for root, _dirs, files in os.walk(dir_path):
            for filename in files:
                if predicate(filename.lower()):
                    matches.append(os.path.join(root, filename))
        return matches
    for filename in os.listdir(dir_path):
        path = os.path.join(dir_path, filename)
        if os.path.isfile(path) and predicate(filename.lower()):
            matches.append(path)
    return matches


def scan_html_files(
    dir_path: str, *, recursive: bool = True
) -> Tuple[Optional[datetime], Optional[datetime]]:
    html_files = _matching_files(
        dir_path,
        lambda name: name.startswith("messages") and name.endswith(".html"),
        recursive=recursive,
    )

    if not html_files:
        return None, None

    min_dt = None
    max_dt = None

    for path in sorted(html_files):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                carry = ""
                for chunk in iter(lambda: f.read(1024 * 1024), ""):
                    data = carry + chunk
                    if len(data) <= 256:
                        carry = data
                        continue
                    scan, carry = data[:-256], data[-256:]
                    for ts_raw in HTML_TITLE_RE.findall(scan):
                        dt = parse_html_timestamp(ts_raw)
                        if dt:
                            min_dt = dt if min_dt is None else min(min_dt, dt)
                            max_dt = dt if max_dt is None else max(max_dt, dt)
                for ts_raw in HTML_TITLE_RE.findall(carry):
                    dt = parse_html_timestamp(ts_raw)
                    if dt:
                        min_dt = dt if min_dt is None else min(min_dt, dt)
                        max_dt = dt if max_dt is None else max(max_dt, dt)
        except Exception:
            pass

    return min_dt, max_dt

def scan_json_files(
    dir_path: str, *, recursive: bool = True
) -> Tuple[Optional[datetime], Optional[datetime]]:
    json_files = _matching_files(
        dir_path,
        lambda name: name in ("result.json", "results.json", "export_results.json"),
        recursive=recursive,
    )

    if not json_files:
        return None, None

    min_dt = None
    max_dt = None

    for path in json_files:
        carry = ""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                while True:
                    chunk = f.read(16 * 1024 * 1024) # 16MB chunks
                    if not chunk:
                        break
                    data = carry + chunk
                    if len(data) <= 256:
                        carry = data
                        continue
                    chunk, carry = data[:-256], data[-256:]

                    unix_matches = JSON_DATE_UNIX_RE.findall(chunk)
                    for ts_str in unix_matches:
                        try:
                            dt = datetime.fromtimestamp(int(ts_str), tz=timezone.utc)
                            if min_dt is None or dt < min_dt:
                                min_dt = dt
                            if max_dt is None or dt > max_dt:
                                max_dt = dt
                        except Exception:
                            pass

                    iso_matches = JSON_DATE_RE.findall(chunk)
                    for iso_str in iso_matches:
                        try:
                            dt = datetime.fromisoformat(iso_str)
                            dt = (
                                dt.replace(tzinfo=timezone.utc)
                                if dt.tzinfo is None
                                else dt.astimezone(timezone.utc)
                            )
                            if min_dt is None or dt < min_dt:
                                min_dt = dt
                            if max_dt is None or dt > max_dt:
                                max_dt = dt
                        except Exception:
                            pass
            for ts_str in JSON_DATE_UNIX_RE.findall(carry):
                try:
                    dt = datetime.fromtimestamp(int(ts_str), tz=timezone.utc)
                    min_dt = dt if min_dt is None else min(min_dt, dt)
                    max_dt = dt if max_dt is None else max(max_dt, dt)
                except Exception:
                    pass
            for iso_str in JSON_DATE_RE.findall(carry):
                try:
                    dt = datetime.fromisoformat(iso_str)
                    dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
                    min_dt = dt if min_dt is None else min(min_dt, dt)
                    max_dt = dt if max_dt is None else max(max_dt, dt)
                except Exception:
                    pass
        except Exception:
            pass

    return min_dt, max_dt

def scan_sqlite_file(
    dir_path: str, *, recursive: bool = True
) -> Tuple[Optional[datetime], Optional[datetime]]:
    sqlite_files = _matching_files(
        dir_path,
        lambda name: name == "database.sqlite",
        recursive=recursive,
    )

    if not sqlite_files:
        return None, None

    min_dt = None
    max_dt = None
    for path in sqlite_files:
        try:
            conn = sqlite3.connect(path)
            cursor = conn.cursor()
            cursor.execute("SELECT MIN(time), MAX(time) FROM messages WHERE time IS NOT NULL AND time > 0;")
            min_epoch, max_epoch = cursor.fetchone()
            conn.close()
            if min_epoch:
                value = datetime.fromtimestamp(int(min_epoch), tz=timezone.utc)
                min_dt = value if min_dt is None else min(min_dt, value)
            if max_epoch:
                value = datetime.fromtimestamp(int(max_epoch), tz=timezone.utc)
                max_dt = value if max_dt is None else max(max_dt, value)
        except Exception:
            continue
    return min_dt, max_dt

def detect_dates(
    dir_path: str, *, recursive: bool = True
) -> Tuple[Optional[datetime], Optional[datetime]]:
    # Combine all available formats; mixed legacy folders must not let one
    # format hide an earlier or later range found in another.
    sq_min, sq_max = scan_sqlite_file(dir_path, recursive=recursive)
    js_min, js_max = scan_json_files(dir_path, recursive=recursive)
    ht_min, ht_max = scan_html_files(dir_path, recursive=recursive)
    values_min = [value for value in (sq_min, js_min, ht_min) if value is not None]
    values_max = [value for value in (sq_max, js_max, ht_max) if value is not None]
    return (min(values_min) if values_min else None, max(values_max) if values_max else None)

def is_backup_root(dir_path: str) -> bool:
    try:
        names = [n.lower() for n in os.listdir(dir_path)]
    except Exception:
        return False
    if any(n.startswith("messages") and n.endswith(".html") for n in names):
        return True
    if any(n in ("result.json", "results.json", "export_results.json") for n in names):
        return True
    if "database.sqlite" in names:
        return True
    return False

def _compute_range_dir(subdir: str, *, recursive: bool = True) -> Optional[str]:
    min_dt, max_dt = detect_dates(subdir, recursive=recursive)
    if min_dt and max_dt:
        return f"{format_utc_iso(min_dt)}__{format_utc_iso(max_dt)}"
    return None

def _should_consider(subdir_name: str, *, all_dirs: bool) -> bool:
    if all_dirs:
        return True
    if "unknown" in subdir_name:
        return True
    if not VALID_RANGE_DIR_RE.match(subdir_name):
        return True
    return False

def _rename_dir(src: str, dst: str, *, apply: bool) -> None:
    if not apply:
        print(f"DRY-RUN rename: {src} -> {dst}")
        return
    os.rename(src, dst)
    print(f"renamed: {src} -> {dst}")


def _flat_entries_to_move(chat_dir: str, range_name: str) -> List[str]:
    """Return direct flat-export entries while preserving chat-level state and ranges."""
    movable: List[str] = []
    for name in sorted(os.listdir(chat_dir)):
        path = os.path.join(chat_dir, name)
        if name == range_name or name == ".tgbackman_target.json":
            continue
        if name.startswith((".partial-", ".dry-run-")):
            continue
        if os.path.isdir(path):
            if VALID_RANGE_DIR_RE.fullmatch(name) or is_backup_root(path):
                continue
        movable.append(name)
    return movable


def _wrap_flat_backup(
    chat_dir: str, range_name: str, *, apply: bool, announce: bool = True
) -> List[str]:
    """Move one flat export into a new range directory, rolling back on failure."""
    destination = os.path.join(chat_dir, range_name)
    if os.path.exists(destination):
        raise FileExistsError(f"destination already exists: {destination}")
    entries = _flat_entries_to_move(chat_dir, range_name)
    if not entries:
        raise RuntimeError(f"no flat-export entries found in {chat_dir}")
    if not apply:
        preview = ", ".join(entries[:8])
        suffix = f", ... and {len(entries) - 8} more" if len(entries) > 8 else ""
        print(
            f"DRY-RUN wrap: {chat_dir} -> {destination} "
            f"({len(entries)} top-level entries: {preview}{suffix})"
        )
        return entries

    os.mkdir(destination)
    moved: List[str] = []
    try:
        for name in entries:
            os.rename(os.path.join(chat_dir, name), os.path.join(destination, name))
            moved.append(name)
    except Exception as exc:
        rollback_errors: List[str] = []
        for name in reversed(moved):
            try:
                os.rename(os.path.join(destination, name), os.path.join(chat_dir, name))
            except Exception as rollback_exc:
                rollback_errors.append(f"{name}: {rollback_exc}")
        if not rollback_errors:
            os.rmdir(destination)
            raise RuntimeError(f"wrap failed and was rolled back: {exc}") from exc
        raise RuntimeError(
            f"wrap failed: {exc}; rollback also failed for {', '.join(rollback_errors)}"
        ) from exc
    if announce:
        print(f"wrapped: {chat_dir} -> {destination} ({len(moved)} top-level entries)")
    return moved


def _rollback_flat_backup(chat_dir: str, destination: str, entries: List[str]) -> None:
    """Undo a completed wrap without overwriting anything at the chat root."""
    errors: List[str] = []
    for name in reversed(entries):
        source = os.path.join(destination, name)
        target = os.path.join(chat_dir, name)
        if not os.path.lexists(source):
            errors.append(f"missing wrapped entry: {source}")
            continue
        if os.path.lexists(target):
            errors.append(f"rollback target already exists: {target}")
            continue
        try:
            os.rename(source, target)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    if not errors:
        try:
            os.rmdir(destination)
        except Exception as exc:
            errors.append(f"could not remove empty destination: {exc}")
    if errors:
        raise RuntimeError("; ".join(errors))


def _path_under(path: str, parent: str) -> bool:
    try:
        return os.path.commonpath((os.path.abspath(path), os.path.abspath(parent))) == os.path.abspath(parent)
    except ValueError:
        return False


def _migrated_media_path(media_path: str, chat_dir: str, destination: str) -> Optional[str]:
    """Return the new absolute path for media formerly resolved below chat_dir."""
    value = media_path.strip()
    if not value or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
        return None
    old_path = value if os.path.isabs(value) else os.path.join(chat_dir, value)
    old_path = os.path.abspath(old_path)
    if not _path_under(old_path, chat_dir):
        return None
    relative = os.path.relpath(old_path, chat_dir)
    return os.path.abspath(os.path.join(destination, relative))


def _matching_db_chat_ids(conn: sqlite3.Connection, chat_dir: str) -> List[str]:
    expected = os.path.normpath(os.path.abspath(chat_dir))
    rows = conn.execute(
        "SELECT chat_id, backup_path FROM chats WHERE backup_path IS NOT NULL"
    ).fetchall()
    return [
        str(chat_id)
        for chat_id, backup_path in rows
        if os.path.normpath(os.path.abspath(str(backup_path))) == expected
    ]


def _plan_db_migration(
    conn: sqlite3.Connection, chat_dir: str, destination: str
) -> Tuple[List[str], List[Tuple[str, int]], int]:
    """Return chat IDs, media row updates, and missing-on-disk media count."""
    chat_ids = _matching_db_chat_ids(conn, chat_dir)
    updates: List[Tuple[str, int]] = []
    missing = 0
    for chat_id in chat_ids:
        rows = conn.execute(
            "SELECT id, media_path FROM messages "
            "WHERE chat_id = ? AND media_path IS NOT NULL AND media_path != ''",
            (chat_id,),
        )
        for row_id, media_path in rows:
            new_path = _migrated_media_path(str(media_path), chat_dir, destination)
            if new_path is None or new_path == media_path:
                continue
            updates.append((new_path, int(row_id)))
            relative = os.path.relpath(new_path, destination)
            if not os.path.exists(os.path.join(chat_dir, relative)):
                missing += 1
    return chat_ids, updates, missing


def _apply_db_migration(
    conn: sqlite3.Connection,
    chat_ids: List[str],
    updates: List[Tuple[str, int]],
    destination: str,
) -> None:
    if updates:
        conn.executemany("UPDATE messages SET media_path = ? WHERE id = ?", updates)
    if chat_ids:
        conn.executemany(
            "UPDATE chats SET backup_path = ? WHERE chat_id = ?",
            [(destination, chat_id) for chat_id in chat_ids],
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Split output root (contains per-chat folders)")
    ap.add_argument("--apply", action="store_true", help="Apply renames (default: dry-run)")
    ap.add_argument(
        "--wrap-flat",
        action="store_true",
        help="Wrap flat per-chat export roots in a detected UTC date-range folder",
    )
    ap.add_argument(
        "--chat",
        action="append",
        default=[],
        metavar="NAME",
        help="Limit work to an exact chat-folder name; may be supplied more than once",
    )
    ap.add_argument(
        "--db",
        metavar="PATH",
        help=(
            "SQLite index whose chat/media paths must follow wrapped flat backups; "
            "required with --wrap-flat --apply"
        ),
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Consider all subfolders (default: only unknown/non-standard names).",
    )
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2
    if args.wrap_flat and args.apply and not args.db:
        print("--db is required with --wrap-flat --apply", file=sys.stderr)
        return 2
    db_path = os.path.abspath(args.db) if args.db else None
    if db_path and not os.path.isfile(db_path):
        print(f"Database does not exist: {db_path}", file=sys.stderr)
        return 2

    chats: List[str]
    try:
        chats = sorted(os.listdir(root))
    except Exception as e:
        print(f"Failed to list root: {e}", file=sys.stderr)
        return 2

    renamed = 0
    wrapped = 0
    planned = 0
    skipped = 0
    errors = 0

    for chat in chats:
        if args.chat and chat not in set(args.chat):
            continue
        chat_dir = os.path.join(root, chat)
        if not os.path.isdir(chat_dir) or os.path.islink(chat_dir):
            continue
        if args.wrap_flat and is_backup_root(chat_dir):
            try:
                new_name = _compute_range_dir(chat_dir, recursive=False)
                if not new_name:
                    skipped += 1
                    print(f"skip (no detectable date range): {chat_dir}", file=sys.stderr)
                    continue
                if os.path.exists(os.path.join(chat_dir, new_name)):
                    skipped += 1
                    print(
                        f"skip (dest exists): {chat_dir} -> {os.path.join(chat_dir, new_name)}",
                        file=sys.stderr,
                    )
                    continue
                destination = os.path.join(chat_dir, new_name)
                db_chat_ids: List[str] = []
                db_updates: List[Tuple[str, int]] = []
                missing_media = 0
                if db_path:
                    with sqlite3.connect(db_path) as preview_conn:
                        db_chat_ids, db_updates, missing_media = _plan_db_migration(
                            preview_conn, chat_dir, destination
                        )
                if not args.apply:
                    planned += 1
                    _wrap_flat_backup(chat_dir, new_name, apply=False)
                    if db_path:
                        print(
                            f"  DB plan: {len(db_chat_ids)} chat row(s), "
                            f"{len(db_updates)} media path(s) made absolute"
                            + (f", {missing_media} already missing on disk" if missing_media else "")
                        )
                    continue

                moved: List[str] = []
                conn = sqlite3.connect(db_path)
                try:
                    conn.execute("PRAGMA busy_timeout = 30000")
                    conn.execute("BEGIN IMMEDIATE")
                    db_chat_ids, db_updates, missing_media = _plan_db_migration(
                        conn, chat_dir, destination
                    )
                    moved = _wrap_flat_backup(
                        chat_dir, new_name, apply=True, announce=False
                    )
                    _apply_db_migration(conn, db_chat_ids, db_updates, destination)
                    conn.commit()
                except Exception:
                    conn.rollback()
                    if moved:
                        try:
                            _rollback_flat_backup(chat_dir, destination, moved)
                        except Exception as rollback_exc:
                            raise RuntimeError(
                                f"database migration failed and filesystem rollback also failed: "
                                f"{rollback_exc}"
                            ) from rollback_exc
                    raise
                finally:
                    conn.close()
                print(
                    f"wrapped: {chat_dir} -> {destination} ({len(moved)} top-level entries); "
                    f"updated {len(db_chat_ids)} DB chat row(s) and {len(db_updates)} media path(s)"
                    + (f"; warning: {missing_media} media path(s) were already missing" if missing_media else "")
                )
                if args.apply:
                    wrapped += 1
            except Exception as e:
                errors += 1
                print(f"error: flat wrap failed: {chat_dir}: {e}", file=sys.stderr)
            continue
        try:
            subdirs = sorted(os.listdir(chat_dir))
        except Exception:
            continue
        for sub in subdirs:
            subdir = os.path.join(chat_dir, sub)
            if not os.path.isdir(subdir):
                continue
            if not _should_consider(sub, all_dirs=args.all):
                continue
            # Only touch dirs that look like an export root
            if not is_backup_root(subdir):
                continue
            try:
                new_name = _compute_range_dir(subdir)
            except Exception as e:
                errors += 1
                print(f"error: {subdir}: {e}", file=sys.stderr)
                continue
            if not new_name or "unknown__unknown" == new_name:
                skipped += 1
                continue
            if new_name == sub:
                skipped += 1
                continue
            dst = os.path.join(chat_dir, new_name)
            if os.path.exists(dst):
                skipped += 1
                print(f"skip (dest exists): {subdir} -> {dst}", file=sys.stderr)
                continue
            try:
                if not args.apply:
                    planned += 1
                _rename_dir(subdir, dst, apply=args.apply)
                if args.apply:
                    renamed += 1
            except Exception as e:
                errors += 1
                print(f"error: rename failed: {subdir} -> {dst}: {e}", file=sys.stderr)

    if args.apply:
        print(f"done: wrapped={wrapped} renamed={renamed} skipped={skipped} errors={errors}")
    else:
        print(f"done: planned={planned} skipped={skipped} errors={errors}")
    return 0 if errors == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
