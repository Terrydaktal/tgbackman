"""Durable fetch staging and verification helpers.

Staging rows are intentionally kept after failures.  The next run validates
the local media and resumes at the earliest unsafe message instead of asking
Telegram for an already downloaded prefix again.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from ..config import RUN_MESSAGES_TABLE, RUNS_TABLE
from ..errors import ExportError
from .media import sha256_file


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def verify_record_media(record: dict[str, Any], target_dir: Path) -> None:
    relative = record.get("file") or record.get("photo")
    if not relative:
        return
    path = (target_dir / str(relative)).resolve()
    if not _path_is_under(path, target_dir) or not path.is_file():
        raise ExportError(f"downloaded media is missing or outside the chat directory: {path}")
    expected_size = record.get("media_size")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise ExportError(f"downloaded media has size {path.stat().st_size}, expected {expected_size}: {path}")
    expected_hash = record.get("media_sha256")
    if expected_hash and sha256_file(path) != expected_hash:
        raise ExportError(f"downloaded media hash mismatch: {path}")


def staged_resume_after_id(conn: sqlite3.Connection, run_key: str,
                           target_dir: Path) -> tuple[Optional[int], int]:
    rows = conn.execute(f"SELECT message_id, record_json, media_error FROM {RUN_MESSAGES_TABLE} "
                        "WHERE run_key=? ORDER BY message_id", (run_key,))
    staged_count = 0
    staged_max_id: Optional[int] = None
    earliest_retry_id: Optional[int] = None
    for row in rows:
        message_id = int(row["message_id"])
        staged_count += 1
        staged_max_id = message_id if staged_max_id is None else max(staged_max_id, message_id)
        try:
            record = json.loads(str(row["record_json"]))
        except (TypeError, ValueError):
            earliest_retry_id = message_id if earliest_retry_id is None else min(earliest_retry_id, message_id)
            continue
        if not record.get("media_type") or record.get("media_skipped"):
            continue
        if row["media_error"] or record.get("media_error"):
            earliest_retry_id = message_id if earliest_retry_id is None else min(earliest_retry_id, message_id)
            continue
        try:
            verify_record_media(record, target_dir)
        except ExportError:
            earliest_retry_id = message_id if earliest_retry_id is None else min(earliest_retry_id, message_id)
    return (max(0, earliest_retry_id - 1) if earliest_retry_id is not None else staged_max_id, staged_count)


def prune_completed_staging(conn: sqlite3.Connection, *, run_key: Optional[str] = None,
                            commit: bool = True) -> int:
    run_filter = " AND runs.run_key = ?" if run_key is not None else ""
    parameters: tuple[Any, ...] = (run_key,) if run_key is not None else ()
    deleted = conn.execute(f"""DELETE FROM {RUN_MESSAGES_TABLE}
        WHERE EXISTS (SELECT 1 FROM {RUNS_TABLE} AS runs
                      WHERE runs.run_key = {RUN_MESSAGES_TABLE}.run_key
                        AND runs.status = 'completed'{run_filter})""", parameters).rowcount
    if commit:
        conn.commit()
    return max(0, int(deleted))


__all__ = ["verify_record_media", "staged_resume_after_id", "prune_completed_staging"]
