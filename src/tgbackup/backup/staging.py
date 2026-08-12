"""Durable fetch staging and verification helpers.

Staging rows are intentionally kept after failures.  The next run validates
the local media and resumes at the earliest unsafe message instead of asking
Telegram for an already downloaded prefix again.
"""

from __future__ import annotations

import base64
import hashlib
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


def _verify_tl_envelope(envelope: Any, description: str) -> None:
    if not isinstance(envelope, dict) or not isinstance(envelope.get("json"), dict):
        raise ExportError(f"{description} has no structured JSON representation")
    if envelope.get("tl_encoding") != "base64" or not envelope.get("tl_data"):
        raise ExportError(f"{description} has no exact TL payload")
    try:
        payload = base64.b64decode(str(envelope["tl_data"]), validate=True)
    except (TypeError, ValueError) as exc:
        raise ExportError(f"{description} has invalid TL base64") from exc
    if len(payload) != int(envelope.get("tl_size", -1)):
        raise ExportError(f"{description} TL payload size does not match")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != envelope.get("tl_sha256") or digest != envelope.get("snapshot_sha256"):
        raise ExportError(f"{description} TL payload hash does not match")
    if envelope.get("telethon_layer") is None or not envelope.get("telethon_version"):
        raise ExportError(f"{description} has no Telethon layer/version")


def verify_record_metadata(record: dict[str, Any]) -> None:
    if int(record.get("metadata_schema_version") or 0) < 2:
        return
    if not isinstance(record.get("raw_message"), dict):
        raise ExportError(f"message {record.get('id')} has no complete raw JSON object")
    _verify_tl_envelope(record.get("raw_message_tl"), f"message {record.get('id')}")
    sender_status = record.get("sender_entity_status")
    if sender_status not in {"complete", "not_exposed", "not_applicable"}:
        raise ExportError(f"message {record.get('id')} has invalid sender-entity status")
    if sender_status == "complete":
        _verify_tl_envelope(
            record.get("sender_entity"),
            f"sender entity for message {record.get('id')}",
        )
        sender_id = record.get("from_id") or record.get("actor_id")
        entity_peer_id = record["sender_entity"].get("peer_id")
        if sender_id is not None and entity_peer_id is not None and int(sender_id) != int(entity_peer_id):
            raise ExportError(f"message {record.get('id')} sender entity has the wrong peer ID")
    elif sender_status == "not_exposed" and not record.get("sender_entity_error"):
        raise ExportError(f"message {record.get('id')} does not explain unavailable sender metadata")
    elif sender_status == "not_exposed" and record.get("sender_entity") is not None:
        _verify_tl_envelope(
            record.get("sender_entity"),
            f"partial sender entity for message {record.get('id')}",
        )
    expanded = record.get("expanded_metadata")
    if not isinstance(expanded, dict) or expanded.get("schema_version") != 1:
        raise ExportError(f"message {record.get('id')} has no expanded-metadata ledger")
    for category in ("reactions", "poll_votes"):
        detail = expanded.get(category)
        if not isinstance(detail, dict) or detail.get("status") not in {
            "complete",
            "not_exposed",
            "not_applicable",
        }:
            raise ExportError(f"message {record.get('id')} has incomplete {category} metadata")
        for page_number, page in enumerate(detail.get("pages") or [], 1):
            _verify_tl_envelope(
                page,
                f"{category} page {page_number} for message {record.get('id')}",
            )
        if detail["status"] == "complete":
            try:
                api_count = int(detail["api_count"])
                fetched_count = int(detail["fetched_count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ExportError(
                    f"message {record.get('id')} has no verified {category} counts"
                ) from exc
            if api_count != fetched_count:
                raise ExportError(
                    f"message {record.get('id')} has incomplete {category} pagination"
                )
            if fetched_count and not detail.get("pages"):
                raise ExportError(
                    f"message {record.get('id')} has no {category} result pages"
                )
        elif detail["status"] == "not_exposed" and not detail.get("reason"):
            raise ExportError(
                f"message {record.get('id')} does not explain unavailable {category} metadata"
            )


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
        try:
            verify_record_metadata(record)
            if record.get("media_type") and not record.get("media_skipped"):
                if row["media_error"] or record.get("media_error"):
                    raise ExportError(f"message {message_id} has an unresolved media error")
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


__all__ = [
    "prune_completed_staging",
    "staged_resume_after_id",
    "verify_record_media",
    "verify_record_metadata",
]
