#!/usr/bin/env python3
"""
Incremental Telegram user-account exporter.

This is deliberately separate from Telegram Desktop.  It uses Telegram's
MTProto user API through Telethon, reads the active-chat selection from the
existing tgbackman SQLite database, stores messages transactionally in that
database, and keeps downloaded media in each chat's stable media directory.
Legacy JSON range exports remain available only through an explicit option.

The first authentication is interactive.  The resulting session file is a
local bearer credential and must be protected like a password.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import getpass
import hashlib
import json
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

from .config import (
    BLACKLIST_TABLE,
    DEFAULT_CONFIG,
    DEFAULT_DB,
    DEFAULT_OUTPUT,
    DEFAULT_SESSION,
    DIALOGS_TABLE,
    EXPORTS_TABLE,
    MEDIA_ALIASES,
    MEDIA_TYPES,
    PURGES_TABLE,
    RUN_ARCHIVE_TABLE,
    RUN_ATTEMPTS_TABLE,
    RUN_MESSAGES_TABLE,
    RUNS_TABLE,
    SCRIPT_DIR,
    TARGETS_TABLE,
    TARGET_CHAT_LINKS_TABLE,
    credentials,
    database_mtime_ns,
    default_database_path,
    ensure_private_dir,
    parse_env_file,
    parse_media_selection,
    parse_size,
    safe_component,
    secure_session_file,
    telethon_session_file,
    write_credentials,
)
from .errors import ExportError
from .db import (
    active_chats,
    ensure_targets_schema as canonical_ensure_targets_schema,
    refresh_chat_statistics as canonical_refresh_chat_statistics,
    upsert_archival_message,
    upsert_chat_entity_snapshot,
    blacklisted_target_keys,
    database_chats,
    load_targets,
    row_to_target,
    runnable_targets,
    set_target_blacklisted,
)
from .db.connection import open_database
from .models import (
    BackupDateDecision,
    DatabaseChat,
    ExportStats,
    MediaDownloadPlan,
    PurgePlan,
    Target,
)
from .progress import ExportLock, ProgressReporter, human_bytes, human_duration
from .backup.media import (
    _document_media_type,
    _filename_from_file,
    _legacy_media_type_for,
    _photo_size_expected_size,
    _photo_size_sort_key,
    download_media,
    flood_wait_seconds,
    media_download_plan,
    media_filename,
    media_type_for,
    sha256_file,
)
from .backup.staging import (
    prune_completed_staging,
    staged_resume_after_id,
    verify_record_media,
    verify_record_metadata,
)
from .telegram.client import connect_client, require_telethon, resolve_peer, target_input_peer
from .backup.records import (
    database_run_key,
    json_safe,
    range_dir_name,
    range_dir_name_from_stats,
    reply_metadata,
    sender_label,
    tl_object_envelope,
)
from .backup import target_mapping as target_mapping_service
from .backup.target_mapping import (
    auto_map_database_chats as _auto_map_database_chats,
    cache_dialog,
    consolidate_migrated_targets,
    legacy_chat_id,
    link_target_chat,
    match_dialogs_to_database_chats,
    materialize_unbacked_target_chats,
    migrated_peer_destination,
)
from .backup.targets import (
    database_peer_hint,
    direct_target_output_dir,
    entity_description,
    generated_chat_id,
    normalized_chat_name,
    path_is_under,
    target_key,
    target_output_dir,
)

# Compatibility names retained for callers that historically imported these
# schema helpers from exporter.py; implementation now lives in tgbackup.db.
ensure_targets_schema = canonical_ensure_targets_schema
refresh_chat_statistics = canonical_refresh_chat_statistics

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def unix_now() -> int:
    return int(time.time())


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def upsert_target(*args: Any, **kwargs: Any) -> Target:
    """Compatibility wrapper that keeps legacy monkey-patching working."""
    target_mapping_service.entity_description = entity_description
    target_mapping_service.target_key = target_key
    return target_mapping_service.upsert_target(*args, **kwargs)


def auto_map_database_chats(*args: Any, **kwargs: Any) -> tuple[int, int, int, list[tuple[DatabaseChat, str]]]:
    target_mapping_service.entity_description = entity_description
    target_mapping_service.target_key = target_key
    return _auto_map_database_chats(*args, **kwargs)


def open_db(path: Path) -> sqlite3.Connection:
    # Schema/migration ownership belongs to tgbackup.db; this keeps the
    # exporter independent from the legacy importer and makes direct API
    # writes use the same migration path as the GUI/indexer.
    return open_database(path, ensure_schema=canonical_ensure_targets_schema)


def export_key(path: Path) -> str:
    """Return a stable key for one atomically completed export directory."""
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()


def record_export(
    conn: sqlite3.Connection,
    target: Target,
    final_dir: Path,
    stats: ExportStats,
) -> None:
    """Record a completed export before its watermark is advanced."""
    conn.execute(
        f"""
        INSERT INTO {EXPORTS_TABLE} (
            export_key, target_key, source_name, chat_id, output_path,
            message_count, first_message_id, last_message_id,
            first_message_unix, last_message_unix, created_unix
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(export_key) DO UPDATE SET
            message_count = excluded.message_count,
            first_message_id = excluded.first_message_id,
            last_message_id = excluded.last_message_id,
            first_message_unix = excluded.first_message_unix,
            last_message_unix = excluded.last_message_unix
        """,
        (
            export_key(final_dir),
            target.target_key,
            target.source_name,
            target.chat_id,
            str(final_dir),
            stats.message_count,
            stats.first_message_id,
            stats.last_message_id,
            stats.first_message_unix,
            stats.last_message_unix,
            unix_now(),
        ),
    )
    conn.commit()


def pending_exports(conn: sqlite3.Connection, output_root: Optional[Path] = None) -> list[sqlite3.Row]:
    rows = conn.execute(
        f"SELECT * FROM {EXPORTS_TABLE} WHERE indexed_unix IS NULL ORDER BY created_unix, output_path"
    ).fetchall()
    if output_root is None:
        return rows
    return [row for row in rows if path_is_under(Path(str(row["output_path"])), output_root)]


def apply_export_watermarks(
    conn: sqlite3.Connection,
    *,
    require_indexed: bool = False,
    output_root: Optional[Path] = None,
    export_keys: Optional[set[str]] = None,
    allow_unindexed: bool = False,
    activate_chat: bool = True,
) -> int:
    """Advance target watermarks only for exports safely recorded in the ledger."""
    predicate = (
        "indexed_unix IS NOT NULL AND applied_unix IS NULL"
        if require_indexed or not allow_unindexed
        else "applied_unix IS NULL"
    )
    params: list[str] = []
    if export_keys:
        placeholders = ",".join("?" for _ in export_keys)
        predicate += f" AND export_key IN ({placeholders})"
        params.extend(sorted(export_keys))
    rows = conn.execute(
        f"SELECT * FROM {EXPORTS_TABLE} WHERE {predicate} ORDER BY created_unix", params
    ).fetchall()
    if output_root is not None:
        rows = [row for row in rows if path_is_under(Path(str(row["output_path"])), output_root)]
    applied = 0
    for row in rows:
        target_row = conn.execute(
            f"SELECT * FROM {TARGETS_TABLE} WHERE target_key = ?", (row["target_key"],)
        ).fetchone()
        if target_row is None:
            # A removed mapping cannot be advanced, but the export remains indexed
            # and will be visible for manual reconciliation.
            continue
        target = row_to_target(target_row)
        current_id = target.last_message_id or 0
        current_unix = target.last_message_unix or 0
        new_id = max(current_id, int(row["last_message_id"] or 0)) or None
        new_unix = max(current_unix, int(row["last_message_unix"] or 0)) or None
        conn.execute(
            f"""
            UPDATE {TARGETS_TABLE}
            SET last_message_id = ?, last_message_unix = ?, last_export_unix = ?, updated_unix = ?
            WHERE target_key = ?
            """,
            (new_id, new_unix, unix_now(), unix_now(), target.target_key),
        )
        if row["output_path"]:
            register_chat_backup(
                conn,
                target,
                Path(str(row["output_path"])),
                commit=False,
                activate_chat=activate_chat,
            )
        conn.execute(
            f"UPDATE {EXPORTS_TABLE} SET applied_unix = ? WHERE export_key = ?",
            (unix_now(), row["export_key"]),
        )
        applied += 1
    conn.commit()
    return applied


def mark_exports_indexed(conn: sqlite3.Connection, output_root: Optional[Path] = None) -> None:
    rows = pending_exports(conn, output_root)
    for row in rows:
        conn.execute(
            f"UPDATE {EXPORTS_TABLE} SET indexed_unix = ? WHERE export_key = ?",
            (unix_now(), row["export_key"]),
        )
    conn.commit()


def baseline_for_target(conn: sqlite3.Connection, target: Target) -> tuple[Optional[int], Optional[int]]:
    if target.last_message_id is not None or target.last_message_unix is not None:
        return target.last_message_id, target.last_message_unix

    row = conn.execute(
        "SELECT MAX(message_id), MAX(timestamp_unix) FROM messages WHERE chat_id = ? AND COALESCE(is_deleted, 0)=0",
        (target.chat_id,),
    ).fetchone()
    message_id = int(row[0]) if row and row[0] is not None else None
    message_unix = int(row[1]) if row and row[1] is not None else None

    return message_id, message_unix


async def message_record(
    message: Any,
    media_root: Path,
    allow_media_errors: bool,
    selected_media: set[str],
    max_file_size: int,
    download_media_enabled: bool,
    media_retries: int,
    progress: Optional[ProgressReporter] = None,
) -> tuple[dict[str, Any], Optional[str]]:
    date = getattr(message, "date", None)
    if date is None:
        timestamp = None
        timestamp_unix = None
    else:
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        timestamp = iso_utc(date)
        timestamp_unix = int(date.timestamp())

    sender = getattr(message, "sender", None)
    sender_id = getattr(message, "sender_id", None)
    if (sender is None or bool(getattr(sender, "min", False))) and sender_id is not None:
        while True:
            try:
                sender = await message.get_sender()
                break
            except Exception as exc:
                wait_seconds = flood_wait_seconds(exc)
                if wait_seconds is None:
                    raise ExportError(
                        f"sender entity for message {message.id}: {type(exc).__name__}: {exc}"
                    ) from exc
                print(f"Telegram requested a {wait_seconds}s sender wait; sleeping", file=sys.stderr)
                await asyncio.sleep(wait_seconds)

    media_plan = media_download_plan(message)
    media_type = media_plan.media_type if media_plan else None
    media_path = None
    media_size = media_plan.expected_size if media_plan else None
    error: Optional[str] = None
    intentional_skip = False
    if media_type:
        if media_type not in selected_media:
            error = f"media for message {message.id} skipped by media selection ({media_type})"
            intentional_skip = True
        elif max_file_size and media_size is not None and int(media_size) > max_file_size:
            error = (
                f"media for message {message.id} skipped by max file size "
                f"({int(media_size)} > {max_file_size} bytes)"
            )
            intentional_skip = True
        elif download_media_enabled:
            try:
                media_path = await download_media(
                    message,
                    media_root,
                    media_type,
                    media_retries,
                    int(media_size) if media_size is not None else None,
                    progress,
                    media_plan,
                    max_file_size=max_file_size,
                )
                if media_path is None:
                    error = f"Telegram returned no downloadable file for message {message.id}"
                else:
                    media_size = (media_root.parent / media_path).stat().st_size
            except Exception as exc:  # Telethon has several media-specific RPC errors.
                if flood_wait_seconds(exc) is not None:
                    # Keep the typed FloodWait visible to the outer retry policy;
                    # otherwise the requested server delay would be lost in a
                    # generic media_error string.
                    raise
                error = f"media for message {message.id}: {exc}"
            if error and not allow_media_errors:
                raise ExportError(error)

    text = getattr(message, "raw_text", None) or ""
    if not text and media_type:
        text = f"[{media_type}]"
    if not text and getattr(message, "action", None) is not None:
        text = "[service]"

    forwarded_from = None
    forward = getattr(message, "forward", None)
    if forward is not None:
        fwd_sender = getattr(forward, "sender", None)
        fwd_id = getattr(forward, "sender_id", None)
        forwarded_from = sender_label(fwd_sender, fwd_id, False)

    record: dict[str, Any] = {
        "id": int(message.id),
        "type": "service" if getattr(message, "action", None) is not None else "message",
        "date": timestamp,
        "date_unixtime": str(timestamp_unix) if timestamp_unix is not None else None,
        "from": sender_label(sender, sender_id, bool(getattr(message, "out", False))),
        "from_id": str(sender_id) if sender_id is not None else None,
        "text": text,
    }
    record.update(reply_metadata(message))
    if forwarded_from:
        record["forwarded_from"] = forwarded_from
    if media_type:
        record["media_type"] = media_type
        if media_size is not None:
            record["media_size"] = int(media_size)
        if media_path:
            record["file"] = media_path
            record["media_sha256"] = sha256_file(media_root.parent / media_path)
        elif error and intentional_skip:
            record["media_skipped"] = error
        elif error and allow_media_errors:
            record["media_error"] = error
    if getattr(message, "edit_date", None) is not None:
        edit_date = message.edit_date
        if edit_date.tzinfo is None:
            edit_date = edit_date.replace(tzinfo=timezone.utc)
        record["edited"] = iso_utc(edit_date)
    for field_name in (
        "entities",
        "reactions",
        "reply_markup",
        "action",
        "media",
        "forward",
    ):
        value = getattr(message, field_name, None)
        if value is not None:
            record[field_name] = json_safe(value)
    for field_name in ("grouped_id", "via_bot_id", "post_author", "views", "forwards"):
        value = getattr(message, field_name, None)
        if value is not None:
            record[field_name] = json_safe(value)
    # Preserve every field returned by Telethon, including metadata that does
    # not yet have a dedicated searchable column.  The canonical DB stores
    # this compressed in messages.raw_payload.
    record["raw_message"] = json_safe(message)
    try:
        message_envelope = tl_object_envelope(message, require_binary=True)
        sender_envelope = tl_object_envelope(sender, require_binary=True)
    except ValueError as exc:
        raise ExportError(f"message {message.id}: {exc}") from exc
    if message_envelope is not None:
        record["metadata_schema_version"] = 2
        record["raw_message_tl"] = message_envelope
    if sender_envelope is not None and not bool(getattr(sender, "min", False)):
        record["sender_entity"] = sender_envelope
        record["sender_entity_status"] = "complete"
    elif sender_envelope is not None:
        record["sender_entity"] = sender_envelope
        record["sender_entity_status"] = "not_exposed"
        record["sender_entity_error"] = (
            "Telegram returned only a minimal sender entity after an explicit refresh"
        )
    elif sender_id is not None:
        record["sender_entity_status"] = "not_exposed"
        record["sender_entity_error"] = "Telegram returned no sender entity for this sender ID"
    else:
        record["sender_entity_status"] = "not_applicable"
    return record, error


_METADATA_NOT_EXPOSED_ERRORS = {
    "BroadcastPublicVotersForbiddenError",
    "ChatAdminRequiredError",
}


async def _metadata_request(client: Any, request: Any, description: str) -> Any:
    while True:
        try:
            return await client(request)
        except Exception as exc:
            wait_seconds = flood_wait_seconds(exc)
            if wait_seconds is None:
                raise ExportError(f"{description}: {type(exc).__name__}: {exc}") from exc
            print(
                f"Telegram requested a {wait_seconds}s metadata wait for {description}; sleeping",
                file=sys.stderr,
            )
            await asyncio.sleep(wait_seconds)


async def _reaction_metadata(
    client: Any,
    entity: Any,
    message: Any,
    request_delay: float,
    progress: Optional[ProgressReporter] = None,
) -> dict[str, Any]:
    summary = getattr(message, "reactions", None)
    if summary is None:
        return {"status": "not_applicable"}
    expected_summary = sum(
        int(getattr(result, "count", 0) or 0)
        for result in (getattr(summary, "results", None) or [])
    )
    metadata: dict[str, Any] = {
        "status": "complete" if expected_summary == 0 else "pending",
        "summary_count": expected_summary,
        "can_see_list": bool(getattr(summary, "can_see_list", False)),
        "pages": [],
    }
    if expected_summary == 0:
        metadata.update({"api_count": 0, "fetched_count": 0})
        return metadata
    if not metadata["can_see_list"]:
        metadata["status"] = "not_exposed"
        metadata["reason"] = "Telegram did not expose the complete reactor list"
        return metadata

    from telethon.tl.functions.messages import GetMessageReactionsListRequest

    offset: Optional[str] = None
    seen_offsets: set[str] = set()
    fetched = 0
    api_count: Optional[int] = None
    while True:
        if request_delay > 0:
            await asyncio.sleep(request_delay)
        try:
            page = await _metadata_request(
                client,
                GetMessageReactionsListRequest(
                    peer=entity,
                    id=int(message.id),
                    limit=100,
                    offset=offset,
                ),
                f"reaction list for message {message.id}",
            )
        except ExportError as exc:
            cause_name = type(exc.__cause__).__name__ if exc.__cause__ is not None else ""
            if cause_name in _METADATA_NOT_EXPOSED_ERRORS:
                metadata.update({"status": "not_exposed", "reason": str(exc)})
                return metadata
            raise
        envelope = tl_object_envelope(page, require_binary=True)
        if envelope is None:
            raise ExportError(f"reaction list for message {message.id} was not a TL object")
        metadata["pages"].append(envelope)
        page_items = getattr(page, "reactions", None) or []
        fetched += len(page_items)
        page_count = int(getattr(page, "count", fetched) or 0)
        api_count = page_count if api_count is None else api_count
        if progress and (page_count > 100 or getattr(page, "next_offset", None)):
            progress.phase(
                f"metadata message {message.id}: reactors {fetched:,}/{page_count:,}"
            )
        next_offset = getattr(page, "next_offset", None)
        if not next_offset:
            break
        if str(next_offset) in seen_offsets:
            raise ExportError(f"reaction pagination repeated offset for message {message.id}")
        seen_offsets.add(str(next_offset))
        offset = str(next_offset)
    if api_count is None or fetched != api_count:
        raise ExportError(
            f"reaction list for message {message.id} returned {fetched} of {api_count} reactors"
        )
    metadata.update({"status": "complete", "api_count": api_count, "fetched_count": fetched})
    return metadata


async def _poll_vote_metadata(
    client: Any,
    entity: Any,
    message: Any,
    request_delay: float,
    progress: Optional[ProgressReporter] = None,
) -> dict[str, Any]:
    media = getattr(message, "media", None)
    poll = getattr(media, "poll", None)
    results = getattr(media, "results", None)
    if poll is None:
        return {"status": "not_applicable"}
    expected = int(getattr(results, "total_voters", 0) or 0)
    metadata: dict[str, Any] = {
        "status": "complete" if expected == 0 else "pending",
        "summary_count": expected,
        "public_voters": bool(getattr(poll, "public_voters", False)),
        "pages": [],
    }
    if expected == 0:
        metadata.update({"api_count": 0, "fetched_count": 0})
        return metadata
    if not metadata["public_voters"]:
        metadata["status"] = "not_exposed"
        metadata["reason"] = "Telegram marks this poll as anonymous"
        return metadata

    from telethon.tl.functions.messages import GetPollVotesRequest

    offset: Optional[str] = None
    seen_offsets: set[str] = set()
    fetched = 0
    api_count: Optional[int] = None
    while True:
        if request_delay > 0:
            await asyncio.sleep(request_delay)
        try:
            page = await _metadata_request(
                client,
                GetPollVotesRequest(
                    peer=entity,
                    id=int(message.id),
                    limit=100,
                    offset=offset,
                ),
                f"poll voters for message {message.id}",
            )
        except ExportError as exc:
            cause_name = type(exc.__cause__).__name__ if exc.__cause__ is not None else ""
            if cause_name in _METADATA_NOT_EXPOSED_ERRORS:
                metadata.update({"status": "not_exposed", "reason": str(exc)})
                return metadata
            raise
        envelope = tl_object_envelope(page, require_binary=True)
        if envelope is None:
            raise ExportError(f"poll voter list for message {message.id} was not a TL object")
        metadata["pages"].append(envelope)
        page_items = getattr(page, "votes", None) or []
        fetched += len(page_items)
        page_count = int(getattr(page, "count", fetched) or 0)
        api_count = page_count if api_count is None else api_count
        if progress and (page_count > 100 or getattr(page, "next_offset", None)):
            progress.phase(
                f"metadata message {message.id}: poll voters {fetched:,}/{page_count:,}"
            )
        next_offset = getattr(page, "next_offset", None)
        if not next_offset:
            break
        if str(next_offset) in seen_offsets:
            raise ExportError(f"poll-voter pagination repeated offset for message {message.id}")
        seen_offsets.add(str(next_offset))
        offset = str(next_offset)
    if api_count is None or fetched != api_count:
        raise ExportError(
            f"poll voter list for message {message.id} returned {fetched} of {api_count} voters"
        )
    metadata.update({"status": "complete", "api_count": api_count, "fetched_count": fetched})
    return metadata


async def expanded_message_metadata(
    client: Any,
    entity: Any,
    message: Any,
    request_delay: float,
    progress: Optional[ProgressReporter] = None,
) -> dict[str, Any]:
    """Fetch secondary metadata needed to reconstruct all API-visible details."""
    return {
        "schema_version": 1,
        "reactions": await _reaction_metadata(
            client, entity, message, request_delay, progress
        ),
        "poll_votes": await _poll_vote_metadata(
            client, entity, message, request_delay, progress
        ),
    }


async def full_chat_metadata(
    client: Any,
    entity: Any,
    target: Target,
    request_delay: float,
) -> dict[str, Any]:
    """Capture Telegram's complete chat/user information response."""
    if target.peer_kind == "user":
        from telethon.tl.functions.users import GetFullUserRequest

        request = GetFullUserRequest(id=entity)
    elif target.peer_kind == "channel":
        from telethon.tl.functions.channels import GetFullChannelRequest

        request = GetFullChannelRequest(channel=entity)
    elif target.peer_kind == "group":
        from telethon.tl.functions.messages import GetFullChatRequest

        request = GetFullChatRequest(chat_id=int(target.peer_id))
    else:
        raise ExportError(f"unsupported Telegram peer kind: {target.peer_kind}")
    if request_delay > 0:
        await asyncio.sleep(request_delay)
    response = await _metadata_request(
        client,
        request,
        f"full chat metadata for {target.source_name}",
    )
    envelope = tl_object_envelope(response, require_binary=True)
    if envelope is None:
        raise ExportError(
            f"Telegram returned no serializable full metadata for {target.source_name}"
        )
    full_object = getattr(response, "full_user", None) or getattr(
        response, "full_chat", None
    )
    full_peer_id = getattr(full_object, "id", None)
    if full_peer_id is None or abs(int(full_peer_id)) != abs(int(target.peer_id)):
        raise ExportError(
            f"Telegram returned full metadata for the wrong peer: expected "
            f"{target.peer_id}, got {full_peer_id}"
        )
    return envelope


async def iter_message_records(
    client: Any,
    entity: Any,
    baseline_id: Optional[int],
    baseline_unix: Optional[int],
    media_root: Path,
    overlap_ids: int,
    overlap_seconds: int,
    allow_media_errors: bool,
    selected_media: set[str],
    max_file_size: int,
    download_media_enabled: bool,
    media_retries: int,
    discard_overlap: bool,
    full_rescan: bool,
    max_messages: Optional[int],
    request_delay: float = 1.0,
    progress: Optional[ProgressReporter] = None,
    resume_after_id: Optional[int] = None,
) -> AsyncIterator[tuple[dict[str, Any], Optional[str]]]:
    """Yield records incrementally, optionally including a bounded older overlap."""
    cutoff = (baseline_unix or 0) - overlap_seconds if baseline_unix else 0
    selected = 0

    async def build_record(message: Any) -> tuple[dict[str, Any], Optional[str]]:
        record, media_error = await message_record(
            message,
            media_root,
            allow_media_errors,
            selected_media,
            max_file_size,
            download_media_enabled,
            media_retries,
            progress,
        )
        if record.get("metadata_schema_version") == 2:
            record["expanded_metadata"] = await expanded_message_metadata(
                client,
                entity,
                message,
                request_delay,
                progress,
            )
        return record, media_error

    if not full_rescan and baseline_id:
        if discard_overlap:
            query_min_id = baseline_id
        elif overlap_ids > 0:
            query_min_id = max(0, baseline_id - overlap_ids)
        elif overlap_seconds > 0 and not discard_overlap:
            # A date-only safety window has no ID lower bound; Telethon must
            # inspect older IDs to find messages in the requested time span.
            query_min_id = 0
        else:
            query_min_id = baseline_id
        if resume_after_id is not None:
            query_min_id = max(query_min_id, int(resume_after_id))
        iterator = client.iter_messages(
            entity, min_id=query_min_id, reverse=True, wait_time=request_delay
        )
        async for message in iterator:
            message_id = int(message.id)
            if resume_after_id is not None and message_id <= int(resume_after_id):
                continue
            in_overlap = (
                not discard_overlap
                and message_id <= baseline_id
                and (overlap_ids <= 0 or message_id > baseline_id - overlap_ids)
                and (overlap_ids > 0 or overlap_seconds > 0)
                and (
                    overlap_seconds <= 0
                    or (
                        cutoff
                        and getattr(message, "date", None) is not None
                        and int(message.date.replace(tzinfo=message.date.tzinfo or timezone.utc).timestamp()) >= cutoff
                    )
                )
            )
            if message_id > baseline_id or in_overlap:
                record, error = await build_record(message)
                selected += 1
                if progress:
                    progress.observe(record, error)
                yield record, error
                if max_messages and selected >= max_messages:
                    break
    elif not full_rescan and cutoff:
        # No trustworthy message ID exists.  Walk chronologically and discard
        # records up to the date boundary as they arrive so a large first-run
        # history is still bounded in memory.
        iterator = client.iter_messages(
            entity,
            min_id=int(resume_after_id or 0),
            reverse=True,
            wait_time=request_delay,
        )
        async for message in iterator:
            if resume_after_id is not None and int(message.id) <= int(resume_after_id):
                continue
            date = getattr(message, "date", None)
            date_unix = int(date.replace(tzinfo=date.tzinfo or timezone.utc).timestamp()) if date else 0
            if date:
                if overlap_seconds > 0 and date_unix < cutoff:
                    continue
                # A date-only watermark has one-second precision.  Re-read its
                # boundary second and rely on the canonical unique-key upsert;
                # using 'less than or equal' could permanently miss another
                # message created within that same second.
                if overlap_seconds == 0 and date_unix < (baseline_unix or 0):
                    continue
            record, error = await build_record(message)
            selected += 1
            if progress:
                progress.observe(record, error)
            yield record, error
            if max_messages and selected >= max_messages:
                break
    else:
        iterator = client.iter_messages(
            entity,
            min_id=int(resume_after_id or 0),
            reverse=True,
            wait_time=request_delay,
        )
        async for message in iterator:
            if resume_after_id is not None and int(message.id) <= int(resume_after_id):
                continue
            record, error = await build_record(message)
            selected += 1
            if progress:
                progress.observe(record, error)
            yield record, error
            if max_messages and selected >= max_messages:
                break


async def collect_messages(
    client: Any,
    entity: Any,
    baseline_id: Optional[int],
    baseline_unix: Optional[int],
    media_root: Path,
    overlap_ids: int,
    overlap_seconds: int,
    allow_media_errors: bool,
    selected_media: set[str],
    max_file_size: int,
    download_media_enabled: bool,
    media_retries: int,
    discard_overlap: bool,
    full_rescan: bool,
    max_messages: Optional[int],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Compatibility wrapper for callers that explicitly need an in-memory result."""
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    async for record, error in iter_message_records(
        client,
        entity,
        baseline_id,
        baseline_unix,
        media_root,
        overlap_ids,
        overlap_seconds,
        allow_media_errors,
        selected_media,
        max_file_size,
        download_media_enabled,
        media_retries,
        discard_overlap,
        full_rescan,
        max_messages,
    ):
        if error:
            errors.append(error)
        records.append(record)
    records.sort(key=lambda item: (item.get("id") is None, item.get("id", 0)))
    return records, errors


async def write_export_stream(
    target: Target,
    records: AsyncIterator[tuple[dict[str, Any], Optional[str]]],
    output_root: Path,
    staging_dir: Path,
    target_dir_override: Optional[Path] = None,
) -> tuple[Optional[Path], ExportStats]:
    now = utc_now()
    # The command-line output root is authoritative.  This lets a user move
    # the export destination without having to remap every target.
    target_dir = target_dir_override or target_output_dir(output_root, target)
    target_dir.mkdir(parents=True, exist_ok=True)
    stats = ExportStats()
    result_path = staging_dir / "result.json"
    with result_path.open("w", encoding="utf-8") as handle:
        handle.write('{\n  "chats": {\n    "about": "Incremental export created by tgbackman Telegram API exporter",\n    "list": [{\n')
        handle.write(f'      "id": {json.dumps(target.chat_id)},\n')
        handle.write(f'      "name": {json.dumps(target.title, ensure_ascii=False)},\n')
        handle.write(
            f'      "type": {json.dumps("personal_chat" if target.peer_kind == "user" else target.peer_kind)},\n'
        )
        handle.write(f'      "username": {json.dumps(target.username)},\n      "messages": [\n')
        first = True
        async for record, error in records:
            if not first:
                handle.write(",\n")
            json.dump(record, handle, ensure_ascii=False, indent=2)
            first = False
            stats.observe(record, error)
        handle.write("\n      ]\n    }]\n  }\n}\n")
        handle.flush()
        os.fsync(handle.fileno())
    with contextlib.suppress(OSError):
        result_path.chmod(0o600)

    if stats.message_count == 0:
        return None, stats

    meta = {
        "tool": "tgbackman",
        "kind": "telegram_api_incremental",
        "chat_name": target.title,
        "chat_id": target.chat_id,
        "peer_kind": target.peer_kind,
        "peer_id": target.peer_id,
        "created_utc": iso_utc(now),
        "scope": "messages_and_message_media",
        "limitations": [
            "Telegram-deleted or inaccessible media cannot be downloaded",
            "profile photos, stories, and non-message account data are not exported",
        ],
        "message_count": stats.message_count,
        "media_count": stats.media_count,
        "skipped_media_count": stats.skipped_media_count,
        "media_errors": stats.media_errors,
    }
    metadata_path = staging_dir / ".backman_export_meta.json"
    metadata_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with contextlib.suppress(OSError):
        metadata_path.chmod(0o600)
    with contextlib.suppress(FileNotFoundError):
        (staging_dir / ".partial_state.json").unlink()
    final_dir = target_dir / range_dir_name_from_stats(stats, now)
    if final_dir.exists():
        final_dir = target_dir / f"{final_dir.name}__run-{time.time_ns()}"
        while final_dir.exists():
            final_dir = target_dir / f"{final_dir.name}-1"
    os.replace(staging_dir, final_dir)
    return final_dir, stats


async def write_database_stream(
    conn: sqlite3.Connection,
    target: Target,
    records: AsyncIterator[tuple[dict[str, Any], Optional[str]]],
    target_dir: Path,
    run_key: str,
    baseline_id: Optional[int],
    baseline_unix: Optional[int],
    full_rescan: bool,
    progress: Optional[ProgressReporter] = None,
    activate_chat: bool = True,
    chat_entity_snapshot: Optional[dict[str, Any]] = None,
    chat_full_snapshot: Optional[dict[str, Any]] = None,
) -> ExportStats:
    """Stage a resumable fetch, then atomically merge messages and watermark."""
    target_dir.mkdir(parents=True, exist_ok=True)
    attempt_key = hashlib.sha256(f"{run_key}\0{time.time_ns()}".encode("utf-8")).hexdigest()
    conn.execute(
        f"INSERT INTO {RUN_ATTEMPTS_TABLE}(attempt_key, run_key, started_unix, status) VALUES (?, ?, ?, 'running')",
        (attempt_key, run_key, unix_now()),
    )
    conn.execute(
        f"""INSERT INTO {RUNS_TABLE}(
                run_key, target_key, chat_id, baseline_message_id, baseline_unix,
                full_rescan, status, started_unix
            ) VALUES (?, ?, ?, ?, ?, ?, 'running', ?)
            ON CONFLICT(run_key) DO UPDATE SET status='running', error=NULL""",
        (
            run_key, target.target_key, target.chat_id, baseline_id, baseline_unix,
            int(full_rescan), unix_now(),
        ),
    )
    conn.commit()
    staged_since_commit = 0
    merging = False
    try:
        async for record, error in records:
            conn.execute(
                f"""INSERT INTO {RUN_MESSAGES_TABLE}(run_key, message_id, record_json, media_error)
                    VALUES (?, ?, ?, ?) ON CONFLICT(run_key, message_id) DO UPDATE SET
                    record_json=excluded.record_json, media_error=excluded.media_error""",
                (
                    run_key,
                    int(record["id"]),
                    json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    error,
                ),
            )
            staged_since_commit += 1
            if staged_since_commit >= 250:
                conn.commit()
                staged_since_commit = 0
        conn.commit()
        if progress:
            progress.phase("fetch complete; verifying staged records and media")

        stats = ExportStats()
        digest = hashlib.sha256()
        run_metadata_schema_version = 2
        for row in conn.execute(
            f"SELECT record_json, media_error FROM {RUN_MESSAGES_TABLE} WHERE run_key=? ORDER BY message_id",
            (run_key,),
        ):
            raw = str(row["record_json"])
            digest.update(raw.encode("utf-8"))
            digest.update(b"\n")
            record = json.loads(raw)
            verify_record_metadata(record)
            verify_record_media(record, target_dir)
            stats.observe(record, str(row["media_error"]) if row["media_error"] else None)
            run_metadata_schema_version = min(
                run_metadata_schema_version,
                int(record.get("metadata_schema_version") or 0),
            )

        if stats.message_count == 0 and not full_rescan:
            if chat_entity_snapshot is not None:
                upsert_chat_entity_snapshot(
                    conn, target.chat_id, chat_entity_snapshot, None, role="entity"
                )
            if chat_full_snapshot is not None:
                upsert_chat_entity_snapshot(
                    conn, target.chat_id, chat_full_snapshot, None, role="full"
                )
            record_chat_backup_run(conn, target, "completed_no_new_messages")
            conn.execute(
                f"UPDATE {RUNS_TABLE} SET status='completed', completed_unix=? WHERE run_key=?",
                (unix_now(), run_key),
            )
            conn.execute(
                f"UPDATE {RUN_ATTEMPTS_TABLE} SET status='completed', completed_unix=? WHERE attempt_key=?",
                (unix_now(), attempt_key),
            )
            prune_completed_staging(conn, run_key=run_key, commit=False)
            conn.commit()
            return stats

        if run_metadata_schema_version >= 2 and (
            chat_entity_snapshot is None or chat_full_snapshot is None
        ):
            raise ExportError(
                f"lossless Telegram metadata for {target.source_name} has no complete chat snapshot"
            )

        records_sha256 = digest.hexdigest()
        chat_snapshot_hashes = {
            "entity": (
                str(chat_entity_snapshot["snapshot_sha256"])
                if chat_entity_snapshot is not None
                else None
            ),
            "full": (
                str(chat_full_snapshot["snapshot_sha256"])
                if chat_full_snapshot is not None
                else None
            ),
        }
        source_key = hashlib.sha256(
            (
                f"telegram_api\0{target.peer_kind}\0{target.peer_id}\0{records_sha256}\0"
                f"{chat_snapshot_hashes['entity'] or ''}\0{chat_snapshot_hashes['full'] or ''}"
            ).encode("utf-8")
        ).hexdigest()
        manifest = json.dumps(
            {
                "kind": "telegram_api_run",
                "run_key": run_key,
                "target_key": target.target_key,
                "peer_kind": target.peer_kind,
                "peer_id": target.peer_id,
                "records_sha256": records_sha256,
                "message_count": stats.message_count,
                "metadata_schema_version": run_metadata_schema_version,
                "chat_metadata_schema_version": (
                    1
                    if chat_entity_snapshot is not None and chat_full_snapshot is not None
                    else 0
                ),
                "chat_snapshot_sha256": chat_snapshot_hashes,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        payload = zlib.compress(manifest, 9)
        now = unix_now()
        if progress:
            progress.phase("verification complete; committing messages and watermark atomically")
        merging = True
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """INSERT INTO backup_sources(
                   source_key, source_format, original_path, content_sha256,
                   content_size, compressed_size, compression, payload,
                   imported_unix, message_count
               ) VALUES (?, 'telegram_api', ?, ?, ?, ?, 'zlib', ?, ?, ?)
               ON CONFLICT(source_key) DO UPDATE SET imported_unix=excluded.imported_unix""",
            (
                source_key, f"telegram://{target.peer_kind}/{target.peer_id}/{run_key}",
                records_sha256, len(manifest), len(payload), sqlite3.Binary(payload), now,
                stats.message_count,
            ),
        )
        conn.execute(
            """INSERT INTO backup_imports(
                   source_key, source_format, original_path, chat_id,
                   expected_messages, imported_messages, skipped_records, completed_unix
               ) VALUES (?, 'telegram_api', ?, ?, ?, ?, 0, ?)
               ON CONFLICT(source_key) DO UPDATE SET
                   expected_messages=excluded.expected_messages,
                   imported_messages=excluded.imported_messages,
                   completed_unix=excluded.completed_unix""",
            (
                source_key, f"telegram://{target.peer_kind}/{target.peer_id}/{run_key}",
                target.chat_id, stats.message_count, stats.message_count, now,
            ),
        )
        conn.execute(
            """INSERT INTO backup_import_files(
                   source_key, original_path, source_format, chat_id,
                   expected_messages, imported_messages, skipped_records, completed_unix
               ) VALUES (?, ?, 'telegram_api', ?, ?, ?, 0, ?)
               ON CONFLICT(source_key, original_path) DO UPDATE SET
                   expected_messages=excluded.expected_messages,
                   imported_messages=excluded.imported_messages,
                   completed_unix=excluded.completed_unix""",
            (
                source_key, f"telegram://{target.peer_kind}/{target.peer_id}/{run_key}",
                target.chat_id, stats.message_count, stats.message_count, now,
            ),
        )
        register_chat_backup(
            conn,
            target,
            target_dir,
            commit=False,
            activate_chat=activate_chat,
        )
        if chat_entity_snapshot is not None:
            upsert_chat_entity_snapshot(
                conn,
                target.chat_id,
                chat_entity_snapshot,
                source_key,
                role="entity",
            )
        if chat_full_snapshot is not None:
            upsert_chat_entity_snapshot(
                conn,
                target.chat_id,
                chat_full_snapshot,
                source_key,
                role="full",
            )
        for row in conn.execute(
            f"SELECT message_id, record_json, media_error FROM {RUN_MESSAGES_TABLE} WHERE run_key=? ORDER BY message_id",
            (run_key,),
        ):
            conn.execute(
                f"""INSERT INTO {RUN_ARCHIVE_TABLE}(run_key, message_id, record_json, media_error)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(run_key, message_id) DO UPDATE SET
                      record_json=excluded.record_json, media_error=excluded.media_error""",
                (run_key, int(row["message_id"]), str(row["record_json"]), row["media_error"]),
            )
            upsert_archival_message(
                conn, target.chat_id, json.loads(str(row["record_json"])),
                str(target_dir), source_key, "telegram_api",
            )
        if full_rescan:
            # Preserve previously archived rows, but record messages no longer
            # returned by a successful complete server walk as tombstones.
            conn.execute(
                f"""UPDATE messages SET is_deleted=1, deleted_unix=?
                    WHERE chat_id=? AND NOT EXISTS (
                        SELECT 1 FROM {RUN_MESSAGES_TABLE} AS fetched
                        WHERE fetched.run_key=? AND fetched.message_id=messages.message_id
                    )""",
                (now, target.chat_id, run_key),
            )
        canonical_refresh_chat_statistics(conn, target.chat_id)
        conn.execute(
            f"""INSERT INTO {EXPORTS_TABLE}(
                   export_key, target_key, source_name, chat_id, output_path,
                   message_count, first_message_id, last_message_id,
                   first_message_unix, last_message_unix, created_unix,
                   indexed_unix, applied_unix
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(export_key) DO UPDATE SET
                   message_count=excluded.message_count,
                   indexed_unix=excluded.indexed_unix,
                   applied_unix=excluded.applied_unix""",
            (
                run_key, target.target_key, target.source_name, target.chat_id,
                f"sqlite:{run_key}", stats.message_count, stats.first_message_id,
                stats.last_message_id, stats.first_message_unix, stats.last_message_unix,
                now, now, now,
            ),
        )
        new_watermark_id = (
            (stats.last_message_id if full_rescan else max(target.last_message_id or 0, stats.last_message_id or 0))
            or None
        )
        new_watermark_unix = (
            (stats.last_message_unix if full_rescan else max(target.last_message_unix or 0, stats.last_message_unix or 0))
            or None
        )
        conn.execute(
            f"""UPDATE {TARGETS_TABLE} SET
                   last_message_id=?, last_message_unix=?, last_export_unix=?, updated_unix=?
               WHERE target_key=?""",
            (
                new_watermark_id,
                new_watermark_unix,
                now, now, target.target_key,
            ),
        )
        conn.execute(
            f"UPDATE {RUNS_TABLE} SET status='completed', completed_unix=?, error=NULL WHERE run_key=?",
            (now, run_key),
        )
        conn.execute(
            f"UPDATE {RUN_ATTEMPTS_TABLE} SET status='completed', completed_unix=?, error=NULL WHERE attempt_key=?",
            (now, attempt_key),
        )
        prune_completed_staging(conn, run_key=run_key, commit=False)
        conn.commit()
        return stats
    except Exception as exc:
        if merging:
            conn.rollback()
        else:
            # Fetch-stage rows are safe resume data and must survive a network,
            # media, or process-level failure before the canonical merge starts.
            conn.commit()
        conn.execute(
            f"UPDATE {RUNS_TABLE} SET status='failed', error=? WHERE run_key=?",
            (str(exc), run_key),
        )
        conn.execute(
            f"UPDATE {RUN_ATTEMPTS_TABLE} SET status='failed', completed_unix=?, error=? WHERE attempt_key=?",
            (unix_now(), str(exc), attempt_key),
        )
        record_chat_backup_run(conn, target, "failed")
        conn.commit()
        raise


def remove_staging(path: Optional[Path]) -> None:
    """Remove only a staging directory created by this process."""
    if path is None or not path.exists() or not path.is_dir():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()


def update_watermark(
    conn: sqlite3.Connection,
    target: Target,
    records: list[dict[str, Any]],
) -> None:
    ids = [int(item["id"]) for item in records if item.get("id") is not None]
    dates = [
        int(item["date_unixtime"])
        for item in records
        if item.get("date_unixtime") is not None and str(item["date_unixtime"]).isdigit()
    ]
    last_id = max([target.last_message_id or 0, *ids], default=0) or None
    last_unix = max([target.last_message_unix or 0, *dates], default=0) or None
    conn.execute(
        f"""
        UPDATE {TARGETS_TABLE}
        SET last_message_id = ?, last_message_unix = ?, last_export_unix = ?, updated_unix = ?
        WHERE target_key = ?
        """,
        (last_id, last_unix, unix_now(), unix_now(), target.target_key),
    )
    conn.commit()


def record_chat_backup_run(
    conn: sqlite3.Connection,
    target: Target,
    status: str,
    *,
    run_unix: Optional[int] = None,
) -> None:
    """Record the latest completed/failed run separately from content time.

    ``last_backup_unix`` intentionally means the last run that committed
    message content.  A no-new-message ``--all`` pass still needs a visible
    run timestamp, otherwise the GUI appears stale even though the chat was
    checked successfully.
    """
    timestamp = unix_now() if run_unix is None else int(run_unix)
    updated = conn.execute(
        """UPDATE chats
           SET last_backup_run_unix=?, last_backup_run_status=?
           WHERE chat_id=?""",
        (timestamp, status, target.chat_id),
    )
    if updated.rowcount == 0:
        conn.execute(
            """INSERT OR IGNORE INTO chats
               (chat_id, chat_name, chat_type, backup_path, is_active,
                last_backup_run_unix, last_backup_run_status)
               VALUES (?, ?, ?, ?, 0, ?, ?)""",
            (
                target.chat_id,
                target.title,
                "personal_chat" if target.peer_kind == "user" else target.peer_kind,
                target.output_dir,
                timestamp,
                status,
            ),
        )


def register_chat_backup(
    conn: sqlite3.Connection,
    target: Target,
    final_dir: Path,
    *,
    commit: bool = True,
    activate_chat: bool = True,
) -> None:
    """Keep the existing GUI's latest-backup path/date in sync with exports."""
    now = unix_now()
    updated = conn.execute(
        """
        UPDATE chats
        SET chat_name = ?, chat_type = ?, backup_path = ?,
            is_active = CASE WHEN ? THEN 1 ELSE is_active END,
            last_backup_unix = ?,
            last_backup_run_unix = ?,
            last_backup_run_status = 'committed',
            last_backup_source = 'telegram_api_nonempty_commit',
            last_backup_confidence = 'high',
            last_backup_evidence = 'Committed Telegram API run containing messages'
        WHERE chat_id = ?
        """,
        (
            target.title,
            "personal_chat" if target.peer_kind == "user" else target.peer_kind,
            str(final_dir),
            int(activate_chat),
            now,
            now,
            target.chat_id,
        ),
    )
    if updated.rowcount == 0:
        conn.execute(
            """
            INSERT OR IGNORE INTO chats
                (chat_id, chat_name, chat_type, backup_path, is_active, last_backup_unix,
                 last_backup_run_unix, last_backup_run_status,
                 last_backup_source, last_backup_confidence, last_backup_evidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'committed', 'telegram_api_nonempty_commit', 'high',
                    'Committed Telegram API run containing messages')
            """,
            (
                target.chat_id,
                target.title,
                "personal_chat" if target.peer_kind == "user" else target.peer_kind,
                str(final_dir),
                int(activate_chat),
                now,
                now,
            ),
        )
    if commit:
        conn.commit()


def print_targets(conn: sqlite3.Connection) -> None:
    targets = load_targets(conn)
    if not targets:
        print("No Telegram targets are mapped. Run `map` after authentication.")
        return
    active_keys = {target.target_key for target in load_targets(conn, active_only=True)}
    blocked_keys = blacklisted_target_keys(conn)
    linked_counts = {
        str(row["target_key"]): int(row["linked_count"])
        for row in conn.execute(
            f"""
            SELECT target_key, count(*) AS linked_count
            FROM {TARGET_CHAT_LINKS_TABLE}
            GROUP BY target_key
            """
        )
    }
    for target in targets:
        status = (
            "blacklist" if target.target_key in blocked_keys
            else "disabled" if not target.enabled
            else "active" if target.target_key in active_keys
            else "inactive"
        )
        watermark = target.last_message_id if target.last_message_id is not None else "none"
        print(
            f"{status:8} {target.source_name} -> {target.title} "
            f"[{target.peer_kind}:{target.peer_id}] chat_id={target.chat_id} "
            f"target_key={target.target_key} watermark={watermark} "
            f"links={linked_counts.get(target.target_key, 0)} "
            f"output_dir={target.output_dir or 'derived-from-output'}"
        )


def blacklist_chat_command(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser().resolve()
    lock = ExportLock(db_path.parent / f".{db_path.name}.tgbackman.lock")
    lock.acquire()
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = open_db(db_path)
        target = purge_target(conn, args.target)
        set_target_blacklisted(
            conn,
            target,
            blacklisted=not args.remove,
            reason=args.reason,
        )
        if not args.remove:
            inserted, _ = materialize_unbacked_target_chats(conn)
            conn.commit()
            print(
                f"Blacklisted {target.title} [{target.peer_kind}:{target.peer_id}] "
                f"as {target.target_key}; it will never run, including with --all "
                f"or an explicit --target. Placeholder chats added: {inserted}."
            )
        else:
            print(
                f"Removed {target.title} [{target.peer_kind}:{target.peer_id}] "
                "from the blacklist. Its active setting was not changed."
            )
        return 0
    finally:
        if conn is not None:
            conn.close()
        lock.release()


def _parse_created_utc(value: Any) -> Optional[int]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def _converted_export_metadata(path: Path) -> Optional[tuple[str, int]]:
    meta_path = path / ".backman_export_meta.json"
    if not meta_path.is_file() or meta_path.is_symlink():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if data.get("kind") != "html_single_chat_export_converted":
        return None
    converted_from = data.get("converted_from")
    if not isinstance(converted_from, dict):
        return None
    export_root = converted_from.get("export_root")
    created_unix = _parse_created_utc(data.get("created_utc"))
    if not isinstance(export_root, str) or not export_root.strip() or created_unix is None:
        return None
    return export_root.strip(), created_unix


def _original_asset_max_mtime(path: Path, conversion_unix: int) -> tuple[Optional[int], int]:
    """Return preserved source-asset time, excluding Backman's generated HTML."""
    latest: Optional[int] = None
    count = 0
    try:
        walker = os.walk(path, followlinks=False)
        for directory, dirnames, filenames in walker:
            base = Path(directory)
            dirnames[:] = [
                name for name in dirnames
                if not (base / name).is_symlink()
            ]
            for filename in filenames:
                if filename == ".backman_export_meta.json" or re.fullmatch(
                    r"messages\d*\.html", filename, flags=re.IGNORECASE
                ):
                    continue
                candidate = base / filename
                if candidate.is_symlink() or not candidate.is_file():
                    continue
                try:
                    modified = int(candidate.stat().st_mtime)
                except OSError:
                    continue
                # A converted output cannot prove an original export timestamp
                # from an asset written after the conversion itself.
                if modified > conversion_unix:
                    continue
                latest = modified if latest is None else max(latest, modified)
                count += 1
    except OSError:
        return None, 0
    return latest, count


def calculate_backup_date_repairs(
    conn: sqlite3.Connection,
    *,
    backup_root: Optional[Path] = None,
) -> list[BackupDateDecision]:
    rows = conn.execute(
        """SELECT chat_id, backup_path, max_timestamp_unix, COALESCE(msg_count, 0) AS msg_count
           FROM chats ORDER BY chat_id"""
    ).fetchall()
    mapped_target_chat_ids = {
        str(row[0])
        for row in conn.execute(
            f"""SELECT chat_id FROM {TARGETS_TABLE}
                UNION SELECT chat_id FROM {TARGET_CHAT_LINKS_TABLE}"""
        )
    }
    api_times: dict[str, int] = {}
    for row in conn.execute(
        f"""SELECT chat_id, MAX(committed_unix)
            FROM (
                SELECT chat_id,
                       COALESCE(applied_unix, indexed_unix, created_unix) AS committed_unix
                FROM {EXPORTS_TABLE}
                WHERE message_count > 0
                UNION ALL
                SELECT chat_id, completed_unix AS committed_unix
                FROM backup_imports
                WHERE source_format='telegram_api' AND imported_messages > 0
            )
            GROUP BY chat_id"""
    ):
        if row[1] is not None:
            api_times[str(row[0])] = int(row[1])

    paths: dict[str, Path] = {}
    converted: dict[str, tuple[str, int]] = {}
    for row in rows:
        raw_path = str(row["backup_path"] or "").strip()
        if not raw_path:
            continue
        path = Path(raw_path).expanduser()
        if backup_root is not None:
            try:
                if not path_is_under(path.resolve(strict=False), backup_root):
                    continue
            except OSError:
                continue
        paths[raw_path] = path
        metadata = _converted_export_metadata(path)
        if metadata is not None:
            converted[raw_path] = metadata

    batch_times: dict[str, int] = {}
    batch_asset_counts: dict[str, int] = {}
    for raw_path, (export_root, conversion_unix) in converted.items():
        asset_time, asset_count = _original_asset_max_mtime(paths[raw_path], conversion_unix)
        if asset_time is not None:
            batch_times[export_root] = max(batch_times.get(export_root, 0), asset_time)
            batch_asset_counts[export_root] = batch_asset_counts.get(export_root, 0) + asset_count

    now = unix_now()
    decisions: list[BackupDateDecision] = []
    for row in rows:
        chat_id = str(row["chat_id"])
        max_message = int(row["max_timestamp_unix"]) if row["max_timestamp_unix"] is not None else None
        raw_path = str(row["backup_path"] or "").strip()
        path = paths.get(raw_path)
        timestamp: Optional[int] = None
        source = "unknown"
        confidence = "unknown"
        evidence = "No authoritative backup-date evidence is available"

        if chat_id in api_times:
            timestamp = api_times[chat_id]
            source = "telegram_api_nonempty_commit"
            confidence = "high"
            evidence = "Latest committed API run containing one or more processed messages"
        elif (
            int(row["msg_count"] or 0) == 0 and chat_id in mapped_target_chat_ids
        ) or (path is not None and (path / ".tgbackman_target.json").is_file()):
            source = "telegram_api_no_content_commit"
            confidence = "exact"
            evidence = "Mapped API target has zero archived messages and no non-empty committed run"
        elif path is not None and (path / "database.sqlite").is_file():
            database_path = path / "database.sqlite"
            try:
                timestamp = int(database_path.stat().st_mtime)
                source = "unofficial_snapshot_database_mtime"
                confidence = "high"
                evidence = str(database_path)
            except OSError:
                pass
        elif raw_path in converted:
            export_root, _ = converted[raw_path]
            timestamp = batch_times.get(export_root)
            source = "converted_desktop_export_asset_batch"
            confidence = "high" if timestamp is not None else "unknown"
            evidence = (
                f"Preserved completion time across {batch_asset_counts.get(export_root, 0)} "
                f"original assets from {export_root}"
            )
        elif path is not None:
            candidates: list[tuple[Path, str, str]] = []
            for filename in ("result.json", "results.json"):
                candidate = path / filename
                if candidate.is_file() and not candidate.is_symlink():
                    candidates.append((candidate, "legacy_json_mtime", "medium"))
            if not candidates:
                for candidate in path.glob("messages*.html"):
                    if candidate.is_file() and not candidate.is_symlink():
                        candidates.append((candidate, "legacy_html_mtime", "medium"))
            if candidates:
                selected, source, confidence = max(
                    candidates, key=lambda item: item[0].stat().st_mtime
                )
                timestamp = int(selected.stat().st_mtime)
                evidence = str(selected)
            elif path.is_dir():
                source = "filesystem_directory_mtime"
                confidence = "low"
                timestamp = int(path.stat().st_mtime)
                evidence = str(path)

        if timestamp is not None and timestamp > now + 86400:
            evidence = f"Rejected future timestamp {timestamp}; {evidence}"
            timestamp = None
            source = "invalid_future_timestamp"
            confidence = "unknown"
        if timestamp is not None and max_message is not None and timestamp < max_message:
            evidence = (
                f"Rejected timestamp preceding newest archived message ({timestamp} < {max_message}); "
                f"{evidence}"
            )
            timestamp = None
            source = "timestamp_predates_messages"
            confidence = "unknown"

        decisions.append(
            BackupDateDecision(chat_id, timestamp, source, confidence, evidence)
        )
    return decisions


def repair_backup_dates_command(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser().resolve()
    backup_root = Path(args.backup_root).expanduser().resolve() if args.backup_root else None
    lock = ExportLock(db_path.parent / f".{db_path.name}.tgbackman.lock")
    lock.acquire()
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = open_db(db_path)
        decisions = calculate_backup_date_repairs(conn, backup_root=backup_root)
        existing = {
            str(row["chat_id"]): (
                int(row["last_backup_unix"]) if row["last_backup_unix"] is not None else None,
                str(row["last_backup_source"] or ""),
                str(row["last_backup_confidence"] or ""),
                str(row["last_backup_evidence"] or ""),
            )
            for row in conn.execute(
                """SELECT chat_id, last_backup_unix, last_backup_source,
                          last_backup_confidence, last_backup_evidence FROM chats"""
            )
        }
        changed = [
            decision for decision in decisions
            if existing.get(decision.chat_id) != (
                decision.timestamp, decision.source, decision.confidence, decision.evidence
            )
        ]
        counts: dict[str, int] = {}
        for decision in decisions:
            counts[decision.source] = counts.get(decision.source, 0) + 1
            if args.list_all:
                value = (
                    datetime.fromtimestamp(decision.timestamp, timezone.utc).isoformat()
                    if decision.timestamp is not None else "unknown"
                )
                print(
                    f"{decision.chat_id}: {value} [{decision.confidence}; {decision.source}] "
                    f"{decision.evidence}"
                )
        print(
            f"Backup-date repair: {len(decisions):,} chat row(s) evaluated; "
            f"{len(changed):,} would change."
        )
        print("Evidence summary: " + ", ".join(f"{key}={value:,}" for key, value in sorted(counts.items())))
        if args.dry_run:
            print("Dry run only; no cached backup dates were changed.")
            return 0
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany(
            """UPDATE chats
               SET last_backup_unix=?, last_backup_source=?,
                   last_backup_confidence=?, last_backup_evidence=?
               WHERE chat_id=?""",
            [
                (
                    decision.timestamp,
                    decision.source,
                    decision.confidence,
                    decision.evidence,
                    decision.chat_id,
                )
                for decision in decisions
            ],
        )
        conn.commit()
        print(f"Updated {len(changed):,} cached backup-date record(s) atomically.")
        return 0
    except Exception:
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if conn is not None:
            conn.close()
        lock.release()


def purge_target(conn: sqlite3.Connection, target_key_value: str) -> Target:
    row = conn.execute(
        f"SELECT * FROM {TARGETS_TABLE} WHERE target_key = ?",
        (target_key_value,),
    ).fetchone()
    if row is None:
        raise ExportError(
            f"Unknown target key {target_key_value!r}. Use `targets` and copy the exact target_key."
        )
    return row_to_target(row)


def purge_chat_ids(conn: sqlite3.Connection, target: Target) -> list[str]:
    rows = conn.execute(
        f"SELECT chat_id FROM {TARGET_CHAT_LINKS_TABLE} WHERE target_key = ? ORDER BY chat_id",
        (target.target_key,),
    ).fetchall()
    return sorted({target.chat_id, *(str(row[0]) for row in rows)})


def sqlite_placeholders(values: list[str]) -> str:
    if not values:
        raise ExportError("Internal error: an empty purge scope was generated")
    return ",".join("?" for _ in values)


def local_backup_path(backup_path: Optional[str], media_path: Optional[str]) -> Optional[Path]:
    if not media_path:
        return None
    value = str(media_path)
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        if not backup_path:
            return None
        path = Path(str(backup_path)).expanduser() / path
    return Path(os.path.abspath(path))


def path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def symlink_component(path: Path, root: Path) -> Optional[Path]:
    if not path_within(path, root):
        return path
    current = root
    if current.is_symlink():
        return current
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            return current
    return None


def validate_backup_root(path: Path) -> Path:
    raw = Path(os.path.abspath(path.expanduser()))
    current = raw
    while True:
        if current.is_symlink():
            raise ExportError(f"Backup root ancestry must not contain symbolic links: {current}")
        if current == current.parent:
            break
        current = current.parent
    expanded = raw.resolve(strict=True)
    if expanded == Path(expanded.anchor) or expanded == Path.home():
        raise ExportError(f"Refusing unsafe backup root: {expanded}")
    if not expanded.is_dir():
        raise ExportError(f"Backup root is not an existing directory: {expanded}")
    if expanded.is_symlink():
        raise ExportError(f"Backup root must not be a symbolic link: {expanded}")
    return expanded


def infer_purge_backup_root(target: Target, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        return validate_backup_root(Path(explicit))
    if not target.output_dir:
        return None
    output_dir = Path(os.path.abspath(Path(target.output_dir).expanduser()))
    return validate_backup_root(output_dir.parent)


def matching_target_marker(path: Path, target: Target, chat_ids: set[str]) -> bool:
    marker = path / ".tgbackman_target.json"
    if not marker.is_file() or marker.is_symlink():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return (
        str(payload.get("target_key", "")) == target.target_key
        or str(payload.get("chat_id", "")) in chat_ids
    )


def scan_directory_without_links(path: Path, root: Path) -> tuple[list[Path], list[Path], int]:
    if path == root or not path_within(path, root):
        raise ExportError(f"Refusing unsafe directory deletion target: {path}")
    if path.is_mount():
        raise ExportError(f"Refusing to delete a mountpoint: {path}")
    linked = symlink_component(path, root)
    if linked is not None:
        raise ExportError(f"Refusing directory containing a symbolic-link path component: {linked}")
    files: list[Path] = []
    directories: list[Path] = []
    total = 0
    for current, dirnames, filenames in os.walk(path, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in [*dirnames, *filenames]:
            child = current_path / name
            if child.is_symlink():
                raise ExportError(f"Refusing directory tree containing symbolic link: {child}")
        directories.append(current_path)
        for name in filenames:
            child = current_path / name
            if not child.is_file():
                raise ExportError(f"Refusing non-regular file in deletion tree: {child}")
            files.append(child)
            total += child.stat().st_size
    return files, directories, total


def archival_source_scope(
    conn: sqlite3.Connection,
    chat_ids: list[str],
) -> tuple[list[str], list[str], list[str]]:
    placeholders = sqlite_placeholders(chat_ids)
    candidates = {
        str(row[0])
        for row in conn.execute(
            f"""
            SELECT source_key FROM message_sources WHERE chat_id IN ({placeholders})
            UNION SELECT source_key FROM messages
                  WHERE chat_id IN ({placeholders}) AND source_key IS NOT NULL
            UNION SELECT source_key FROM backup_imports WHERE chat_id IN ({placeholders})
            UNION SELECT source_key FROM backup_import_files WHERE chat_id IN ({placeholders})
            """,
            (*chat_ids, *chat_ids, *chat_ids, *chat_ids),
        )
    }
    exclusive: list[str] = []
    retained: list[str] = []
    retained_paths: set[str] = set()
    for source_key in sorted(candidates):
        outside_messages = int(
            conn.execute(
                f"SELECT count(*) FROM message_sources WHERE source_key=? AND chat_id NOT IN ({placeholders})",
                (source_key, *chat_ids),
            ).fetchone()[0]
        )
        outside_imports = int(
            conn.execute(
                f"""SELECT
                       (SELECT count(*) FROM backup_imports
                        WHERE source_key=? AND (chat_id IS NULL OR chat_id NOT IN ({placeholders}))) +
                       (SELECT count(*) FROM backup_import_files
                        WHERE source_key=? AND (chat_id IS NULL OR chat_id NOT IN ({placeholders})))""",
                (source_key, *chat_ids, source_key, *chat_ids),
            ).fetchone()[0]
        )
        if outside_messages == 0 and outside_imports == 0:
            exclusive.append(source_key)
        else:
            retained.append(source_key)
        for row in conn.execute(
            """SELECT original_path FROM backup_sources WHERE source_key=?
               UNION SELECT original_path FROM backup_imports WHERE source_key=?
               UNION SELECT original_path FROM backup_import_files WHERE source_key=?""",
            (source_key, source_key, source_key),
        ):
            value = str(row[0])
            if not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
                retained_paths.add(value)
    return exclusive, retained, sorted(retained_paths)


def plan_chat_purge(
    conn: sqlite3.Connection,
    target: Target,
    *,
    delete_media: bool,
    backup_root: Optional[Path],
) -> PurgePlan:
    chat_ids = purge_chat_ids(conn, target)
    placeholders = sqlite_placeholders(chat_ids)
    chat_rows = conn.execute(
        f"SELECT chat_id, chat_name, backup_path FROM chats WHERE chat_id IN ({placeholders})",
        chat_ids,
    ).fetchall()
    chat_names = sorted({str(row["chat_name"] or row["chat_id"]) for row in chat_rows})
    bases = {str(row["chat_id"]): str(row["backup_path"]) if row["backup_path"] else None for row in chat_rows}
    message_count = int(
        conn.execute(
            f"SELECT count(*) FROM messages WHERE chat_id IN ({placeholders}) AND COALESCE(is_deleted, 0)=0", chat_ids
        ).fetchone()[0]
    )
    exclusive_sources, retained_sources, raw_source_paths = archival_source_scope(conn, chat_ids)

    if not delete_media:
        return PurgePlan(
            target, chat_ids, chat_names, message_count, backup_root, [], [], [], [], [],
            exclusive_sources, retained_sources, raw_source_paths, 0,
        )
    if backup_root is None:
        raise ExportError(
            "Cannot infer a bounded backup root for media deletion; pass --backup-root explicitly"
        )

    retained_bases = [
        Path(os.path.abspath(Path(str(row[0])).expanduser()))
        for row in conn.execute(
            f"SELECT backup_path FROM chats WHERE chat_id NOT IN ({placeholders}) AND backup_path IS NOT NULL",
            chat_ids,
        )
    ]
    retained_media: set[Path] = set()
    for row in conn.execute(
        f"""SELECT m.media_path, c.backup_path
            FROM messages AS m LEFT JOIN chats AS c ON c.chat_id=m.chat_id
            WHERE m.chat_id NOT IN ({placeholders}) AND COALESCE(m.is_deleted, 0)=0 AND m.media_path IS NOT NULL""",
        chat_ids,
    ):
        path = local_backup_path(row["backup_path"], row["media_path"])
        if path is not None:
            retained_media.add(path)

    directories: set[Path] = set()
    unsafe: set[str] = set()
    candidate_directories = {
        Path(os.path.abspath(Path(value).expanduser()))
        for value in [target.output_dir, *(bases.values())]
        if value
    }
    chat_id_set = set(chat_ids)
    for path in sorted(candidate_directories):
        if not path.is_dir() or not matching_target_marker(path, target, chat_id_set):
            continue
        if not path_within(path, backup_root) or path == backup_root:
            unsafe.add(f"marker-owned directory outside backup root: {path}")
            continue
        if any(path_within(item, path) for item in retained_media):
            unsafe.add(f"marker-owned directory contains retained-chat media: {path}")
            continue
        if any(item == path or path_within(item, path) for item in retained_bases):
            unsafe.add(f"marker-owned directory contains another chat backup path: {path}")
            continue
        try:
            scan_directory_without_links(path, backup_root)
        except ExportError as exc:
            unsafe.add(str(exc))
            continue
        directories.add(path)
    directories = {
        path for path in directories
        if not any(path != other and path_within(path, other) for other in directories)
    }

    owned_files: set[Path] = set()
    shared_files: set[Path] = set()
    missing: set[str] = set()
    for row in conn.execute(
        f"""SELECT m.chat_id, m.message_id, m.media_path, c.backup_path
            FROM messages AS m LEFT JOIN chats AS c ON c.chat_id=m.chat_id
            WHERE m.chat_id IN ({placeholders}) AND COALESCE(m.is_deleted, 0)=0 AND m.media_path IS NOT NULL""",
        chat_ids,
    ):
        path = local_backup_path(row["backup_path"], row["media_path"])
        if path is None:
            missing.add(f"{row['chat_id']}:{row['message_id']} unresolved {row['media_path']}")
            continue
        if not path.exists():
            missing.add(str(path))
            continue
        if not path.is_file():
            unsafe.add(f"media path is not a regular file: {path}")
            continue
        if not path_within(path, backup_root):
            unsafe.add(f"media path outside backup root: {path}")
            continue
        linked = symlink_component(path, backup_root)
        if linked is not None:
            unsafe.add(f"media path contains symbolic link {linked}: {path}")
            continue
        if path in retained_media:
            shared_files.add(path)
            continue
        owned_files.add(path)
        if path.name.endswith("_thumb.jpg"):
            stem = Path(str(path)[:-10])
            for primary in (stem, Path(f"{stem}.jpg")):
                if primary.is_file() and primary not in retained_media:
                    if path_within(primary, backup_root) and symlink_component(primary, backup_root) is None:
                        owned_files.add(primary)

    files_outside_directories = {
        path for path in owned_files
        if not any(path_within(path, directory) for directory in directories)
    }
    bytes_to_delete = sum(path.stat().st_size for path in files_outside_directories)
    for directory in directories:
        _, _, directory_bytes = scan_directory_without_links(directory, backup_root)
        bytes_to_delete += directory_bytes
    raw_source_paths = [
        value for value in raw_source_paths
        if not any(
            path_within(Path(os.path.abspath(Path(value).expanduser())), directory)
            for directory in directories
        )
    ]
    return PurgePlan(
        target=target,
        chat_ids=chat_ids,
        chat_names=chat_names,
        message_count=message_count,
        backup_root=backup_root,
        media_files=sorted(files_outside_directories),
        media_directories=sorted(directories),
        shared_media=sorted(shared_files),
        missing_media=sorted(missing),
        unsafe_media=sorted(unsafe),
        exclusive_source_keys=exclusive_sources,
        retained_source_keys=retained_sources,
        retained_source_paths=raw_source_paths,
        bytes_to_delete=bytes_to_delete,
    )


def print_purge_plan(plan: PurgePlan, *, list_all: bool = False) -> None:
    print(f"Purge target: {plan.target.title} [{plan.target.peer_kind}:{plan.target.peer_id}]")
    print(f"Target key: {plan.target.target_key}")
    print(f"Linked chat IDs ({len(plan.chat_ids)}): {', '.join(plan.chat_ids)}")
    print(f"Messages to remove: {plan.message_count:,}")
    print(f"Exclusive archived sources to remove: {len(plan.exclusive_source_keys):,}")
    print(f"Shared archived sources retained: {len(plan.retained_source_keys):,}")
    print(
        f"Media deletion: {len(plan.media_files):,} individual file(s), "
        f"{len(plan.media_directories):,} marker-owned directorie(s), "
        f"{human_bytes(plan.bytes_to_delete)}"
    )
    limit = None if list_all else 25
    for label, values in (
        ("directory", [str(path) for path in plan.media_directories]),
        ("file", [str(path) for path in plan.media_files]),
        ("shared media retained", [str(path) for path in plan.shared_media]),
        ("missing media", plan.missing_media),
        ("unsafe media retained", plan.unsafe_media),
        ("raw source retained", plan.retained_source_paths),
    ):
        shown = values if limit is None else values[:limit]
        for value in shown:
            print(f"  {label}: {value}")
        if limit is not None and len(values) > limit:
            print(f"  {label}: ... {len(values) - limit:,} more (use --list-all or --manifest)")


def delete_directory_tree(path: Path, root: Path) -> None:
    files, directories, _ = scan_directory_without_links(path, root)
    for file_path in files:
        if file_path.is_symlink() or not file_path.is_file():
            raise ExportError(f"Deletion target changed after validation: {file_path}")
        file_path.unlink()
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        if directory.is_symlink() or not directory.is_dir():
            raise ExportError(f"Deletion directory changed after validation: {directory}")
        directory.rmdir()


def delete_planned_media(plan: PurgePlan) -> None:
    if plan.backup_root is None:
        return
    for path in plan.media_files:
        linked = symlink_component(path, plan.backup_root)
        if linked is not None or path.is_symlink() or not path.is_file():
            raise ExportError(f"Deletion target changed after validation: {path}")
        path.unlink()
    for path in plan.media_directories:
        delete_directory_tree(path, plan.backup_root)
    for path in plan.media_files:
        parent = path.parent
        while parent != plan.backup_root and path_within(parent, plan.backup_root):
            if parent.is_symlink() or not parent.is_dir():
                break
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


def delete_manifest_media(manifest: dict[str, Any], backup_root: Path) -> None:
    """Finish a previously committed purge without touching database rows."""
    files = [Path(os.path.abspath(Path(str(value)).expanduser())) for value in manifest.get("media_files", [])]
    directories = [Path(os.path.abspath(Path(str(value)).expanduser())) for value in manifest.get("media_directories", [])]
    for path in [*files, *directories]:
        if not path_within(path, backup_root) or path == backup_root:
            raise ExportError(f"purge recovery path is outside backup root: {path}")
        linked = symlink_component(path, backup_root)
        if linked is not None:
            raise ExportError(f"purge recovery path contains symbolic link: {linked}")
    for path in files:
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise ExportError(f"purge recovery target changed: {path}")
        path.unlink()
    for path in directories:
        if path.exists():
            delete_directory_tree(path, backup_root)


def delete_purge_database_rows(
    conn: sqlite3.Connection,
    plan: PurgePlan,
    *,
    delete_media: bool,
) -> str:
    purge_key_value = hashlib.sha256(
        f"{plan.target.target_key}\0{time.time_ns()}".encode("utf-8")
    ).hexdigest()
    chat_placeholders = sqlite_placeholders(plan.chat_ids)
    manifest_json = json.dumps(plan.manifest(), ensure_ascii=False, sort_keys=True)
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            f"""INSERT INTO {PURGES_TABLE}(
                   purge_key, target_key, title, chat_ids_json, manifest_json,
                   status, created_unix
               ) VALUES (?, ?, ?, ?, ?, 'database-deleting', ?)""",
            (
                purge_key_value,
                plan.target.target_key,
                plan.target.title,
                json.dumps(plan.chat_ids),
                manifest_json,
                unix_now(),
            ),
        )
        run_keys = [
            str(row[0])
            for row in conn.execute(
                f"""SELECT run_key FROM {RUNS_TABLE}
                    WHERE target_key=? OR chat_id IN ({chat_placeholders})""",
                (plan.target.target_key, *plan.chat_ids),
            )
        ]
        if run_keys:
            run_placeholders = sqlite_placeholders(run_keys)
            conn.execute(
                f"DELETE FROM {RUN_MESSAGES_TABLE} WHERE run_key IN ({run_placeholders})",
                run_keys,
            )
            conn.execute(
                f"DELETE FROM {RUN_ARCHIVE_TABLE} WHERE run_key IN ({run_placeholders})",
                run_keys,
            )
            conn.execute(
                f"DELETE FROM {RUN_ATTEMPTS_TABLE} WHERE run_key IN ({run_placeholders})",
                run_keys,
            )
            conn.execute(
                f"DELETE FROM {RUNS_TABLE} WHERE run_key IN ({run_placeholders})",
                run_keys,
            )
        conn.execute(
            f"DELETE FROM message_sources WHERE chat_id IN ({chat_placeholders})",
            plan.chat_ids,
        )
        conn.execute(
            f"DELETE FROM message_source_media WHERE chat_id IN ({chat_placeholders})",
            plan.chat_ids,
        )
        conn.execute(
            f"DELETE FROM telegram_message_entity_refs WHERE chat_id IN ({chat_placeholders})",
            plan.chat_ids,
        )
        conn.execute(
            f"DELETE FROM telegram_chat_snapshot_sources WHERE chat_id IN ({chat_placeholders})",
            plan.chat_ids,
        )
        conn.execute(
            f"DELETE FROM telegram_chat_entity_refs WHERE chat_id IN ({chat_placeholders})",
            plan.chat_ids,
        )
        conn.execute(
            f"DELETE FROM messages WHERE chat_id IN ({chat_placeholders})",
            plan.chat_ids,
        )
        if plan.exclusive_source_keys:
            source_placeholders = sqlite_placeholders(plan.exclusive_source_keys)
            for table in (
                "telegram_chat_snapshot_sources",
                "message_sources",
                "message_source_media",
                "backup_imports",
                "backup_import_files",
            ):
                conn.execute(
                    f"DELETE FROM {table} WHERE source_key IN ({source_placeholders})",
                    plan.exclusive_source_keys,
                )
            conn.execute(
                f"DELETE FROM backup_sources WHERE source_key IN ({source_placeholders})",
                plan.exclusive_source_keys,
            )
        conn.execute(
            f"DELETE FROM {EXPORTS_TABLE} WHERE target_key=? OR chat_id IN ({chat_placeholders})",
            (plan.target.target_key, *plan.chat_ids),
        )
        conn.execute(
            f"DELETE FROM {TARGET_CHAT_LINKS_TABLE} WHERE target_key=? OR chat_id IN ({chat_placeholders})",
            (plan.target.target_key, *plan.chat_ids),
        )
        conn.execute(
            f"DELETE FROM chats WHERE chat_id IN ({chat_placeholders})",
            plan.chat_ids,
        )
        conn.execute(
            """DELETE FROM telegram_entity_snapshots
               WHERE NOT EXISTS (
                   SELECT 1 FROM telegram_message_entity_refs AS r
                   WHERE r.snapshot_sha256=telegram_entity_snapshots.snapshot_sha256
               ) AND NOT EXISTS (
                   SELECT 1 FROM telegram_chat_entity_refs AS r
                   WHERE r.snapshot_sha256=telegram_entity_snapshots.snapshot_sha256
               ) AND NOT EXISTS (
                   SELECT 1 FROM telegram_chat_snapshot_sources AS r
                   WHERE r.snapshot_sha256=telegram_entity_snapshots.snapshot_sha256
               )"""
        )
        conn.execute(
            f"""UPDATE {TARGETS_TABLE}
                SET enabled=0, last_message_id=NULL, last_message_unix=NULL,
                    last_export_unix=NULL, updated_unix=?
                WHERE target_key=?""",
            (unix_now(), plan.target.target_key),
        )
        # Keep a blacklisted purge visible as an inert zero-message row in the
        # viewer. Disabled targets without a blacklist rule remain absent.
        materialize_unbacked_target_chats(conn)
        status = "media-pending" if delete_media else "completed-database-only"
        conn.execute(
            f"UPDATE {PURGES_TABLE} SET status=?, completed_unix=? WHERE purge_key=?",
            (status, None if delete_media else unix_now(), purge_key_value),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return purge_key_value


def purge_chat_command(args: argparse.Namespace) -> int:
    if args.dry_run and args.confirm:
        raise ExportError("--dry-run cannot be combined with --confirm")
    if not args.dry_run and not args.confirm:
        raise ExportError("Use --dry-run first, or provide --confirm TARGET_KEY to execute the purge")
    db_path = Path(args.db).expanduser().resolve()
    lock = ExportLock(db_path.parent / f".{db_path.name}.tgbackman.lock")
    lock.acquire()
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = open_db(db_path)
        target = purge_target(conn, args.target)
        if args.confirm and args.confirm != target.target_key:
            raise ExportError(
                f"Confirmation mismatch: expected --confirm {target.target_key}"
            )
        backup_root = infer_purge_backup_root(target, args.backup_root) if args.delete_media else None
        if not args.dry_run:
            # Freeze retained-message/media ownership while the destructive plan
            # is built. Exporter jobs are also excluded by the process lock.
            conn.execute("BEGIN IMMEDIATE")
        plan = plan_chat_purge(
            conn,
            target,
            delete_media=args.delete_media,
            backup_root=backup_root,
        )
        print_purge_plan(plan, list_all=args.list_all)
        if args.manifest:
            manifest_path = Path(args.manifest).expanduser()
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(plan.manifest(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.chmod(manifest_path, 0o600)
            print(f"Manifest written: {manifest_path}")
        if args.dry_run:
            print(f"Dry run only. Execute with --confirm {target.target_key} after reviewing the plan.")
            return 0
        purge_key_value = delete_purge_database_rows(
            conn,
            plan,
            delete_media=args.delete_media,
        )
        if args.delete_media:
            try:
                delete_planned_media(plan)
            except Exception as exc:
                conn.execute(
                    f"UPDATE {PURGES_TABLE} SET status='media-incomplete', error=? WHERE purge_key=?",
                    (str(exc), purge_key_value),
                )
                conn.commit()
                raise ExportError(
                    f"Database purge committed, but media deletion stopped: {exc}. "
                    f"The recovery manifest is stored as purge {purge_key_value}."
                ) from exc
            conn.execute(
                f"UPDATE {PURGES_TABLE} SET status='completed', completed_unix=?, error=NULL WHERE purge_key=?",
                (unix_now(), purge_key_value),
            )
            conn.commit()
        print(
            f"Purged {plan.message_count:,} message(s) for {target.title}; "
            f"target {target.target_key} is disabled. Purge ledger: {purge_key_value}"
        )
        if (
            plan.shared_media or plan.missing_media or plan.unsafe_media
            or plan.retained_source_keys or plan.retained_source_paths
        ):
            print(
                "Warning: shared, missing, unsafe, or raw archive items listed above were retained; "
                "this was not a forensic privacy erasure."
            )
        return 0
    finally:
        if conn is not None:
            conn.close()
        lock.release()


def purge_resume_command(args: argparse.Namespace) -> int:
    if args.confirm != args.purge_key:
        raise ExportError("purge recovery requires --confirm PURGE_KEY")
    db_path = Path(args.db).expanduser().resolve()
    lock = ExportLock(db_path.parent / f".{db_path.name}.tgbackman.lock")
    lock.acquire()
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = open_db(db_path)
        row = conn.execute(
            f"SELECT manifest_json, status FROM {PURGES_TABLE} WHERE purge_key=?",
            (args.purge_key,),
        ).fetchone()
        if row is None:
            raise ExportError(f"Unknown purge ledger key: {args.purge_key}")
        if str(row["status"]) == "completed":
            print(f"Purge {args.purge_key} is already complete.")
            return 0
        manifest = json.loads(str(row["manifest_json"]))
        root_value = args.backup_root or manifest.get("backup_root")
        if not root_value:
            raise ExportError("Purge manifest has no backup root; supply --backup-root")
        root = validate_backup_root(Path(root_value))
        delete_manifest_media(manifest, root)
        conn.execute(
            f"UPDATE {PURGES_TABLE} SET status='completed', completed_unix=?, error=NULL WHERE purge_key=?",
            (unix_now(), args.purge_key),
        )
        conn.commit()
        print(f"Completed media recovery for purge {args.purge_key}.")
        return 0
    finally:
        if conn is not None:
            conn.close()
        lock.release()


async def list_dialogs(args: argparse.Namespace) -> None:
    client = await connect_client(args)
    try:
        dialogs = await client.get_dialogs(limit=None)
        for index, dialog in enumerate(dialogs, 1):
            try:
                kind, title, username, peer_id, _, _ = entity_description(dialog.entity)
            except ExportError:
                continue
            suffix = f" @{username}" if username else ""
            print(f"{index:4} {kind:8} {title}{suffix}  peer={peer_id}")
    finally:
        await client.disconnect()


def doctor_offline(args: argparse.Namespace) -> int:
    problems = 0
    config_path = Path(args.config).expanduser()
    try:
        credentials(config_path)
        print(f"credentials: OK ({config_path})")
    except ExportError as exc:
        problems += 1
        print(f"credentials: ERROR: {exc}", file=sys.stderr)

    db_path = Path(args.db).expanduser()
    try:
        conn = open_db(db_path)
    except ExportError as exc:
        print(f"database: ERROR: {exc}", file=sys.stderr)
        return problems + 1
    try:
        active = active_chats(conn)
        targets = load_targets(conn, active_only=True)
        mapped_chat_ids = {
            str(row[0])
            for row in conn.execute(
                f"""
                SELECT links.chat_id
                FROM {TARGET_CHAT_LINKS_TABLE} AS links
                JOIN {TARGETS_TABLE} AS targets ON targets.target_key = links.target_key
                WHERE targets.enabled = 1
                UNION
                SELECT targets.chat_id
                FROM {TARGETS_TABLE} AS targets
                WHERE targets.enabled = 1
                """
            )
        }
        missing = [chat for chat in active if chat.chat_id not in mapped_chat_ids]
        print(f"database: OK ({len(active)} active chat row(s), {len(targets)} mapped target(s))")
        if missing:
            problems += len(missing)
            print(
                "unmapped active chats: "
                + ", ".join(f"{chat.name} [{chat.chat_id}]" for chat in missing),
                file=sys.stderr,
            )
        output = Path(args.output).expanduser()
        try:
            mount_point = infer_mount_point(output)
            if mount_point is not None and not mount_point.is_mount():
                problems += 1
                print(f"output: ERROR: expected mountpoint is not mounted: {mount_point}", file=sys.stderr)
            else:
                output.mkdir(parents=True, exist_ok=True)
                fd, probe = tempfile.mkstemp(prefix=".tgbackman-write-test-", dir=output)
                os.close(fd)
                os.unlink(probe)
                print(f"output: OK ({output})")
        except OSError as exc:
            problems += 1
            print(f"output: ERROR: {exc}", file=sys.stderr)
        session = Path(args.session).expanduser()
        session_file = telethon_session_file(session)
        if session_file.exists():
            secure_session_file(session)
            print(f"session: present ({session_file})")
        else:
            print(f"session: not created yet ({session_file})")
    finally:
        conn.close()
    return 1 if problems else 0


def verify_exports(args: argparse.Namespace) -> int:
    """Validate JSON shape, cross-run IDs, media integrity, and export ledger state."""
    root = Path(args.output).expanduser()
    if not root.is_dir():
        print(f"verify: output directory does not exist: {root}", file=sys.stderr)
        return 1
    result_files = sorted(root.rglob("result.json"))
    if not result_files:
        print(f"verify: no result.json files found under {root}", file=sys.stderr)
        return 1

    problems = 0
    messages = 0
    media = 0
    skipped = 0
    global_ids: dict[tuple[str, int], Path] = {}
    file_summaries: dict[Path, dict[str, Any]] = {}
    for path in result_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            chats = data["chats"]["list"]
            if not isinstance(chats, list):
                raise ValueError("chats.list is not a list")
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            problems += 1
            print(f"verify: INVALID {path}: {exc}", file=sys.stderr)
            continue
        summary: dict[str, Any] = {"count": 0, "keys": {}, "dates": []}
        for chat in chats:
            if not isinstance(chat, dict):
                problems += 1
                print(f"verify: chat entry is not an object: {path}", file=sys.stderr)
                continue
            chat_id = str(chat.get("id", "?"))
            rows = chat.get("messages", [])
            if not isinstance(rows, list):
                problems += 1
                print(f"verify: messages is not a list: {path}", file=sys.stderr)
                continue
            seen: set[int] = set()
            for row in rows:
                if not isinstance(row, dict):
                    problems += 1
                    print(f"verify: message entry is not an object: {path}", file=sys.stderr)
                    continue
                messages += 1
                summary["count"] += 1
                try:
                    message_id = int(row["id"])
                    if message_id in seen:
                        problems += 1
                        print(f"verify: duplicate message {chat_id}:{message_id} in {path}", file=sys.stderr)
                    seen.add(message_id)
                    key = (chat_id, message_id)
                    summary["keys"].setdefault(chat_id, set()).add(message_id)
                    date_value = row.get("date_unixtime")
                    if date_value is not None:
                        try:
                            summary["dates"].append(int(date_value))
                        except (TypeError, ValueError):
                            pass
                    previous = global_ids.get(key)
                    if previous is not None and previous != path and not args.allow_overlap:
                        problems += 1
                        print(
                            f"verify: cross-export duplicate message {chat_id}:{message_id} "
                            f"in {previous} and {path}",
                            file=sys.stderr,
                        )
                    global_ids[key] = path
                except (KeyError, TypeError, ValueError):
                    problems += 1
                    print(f"verify: message without integer id in {path}", file=sys.stderr)
                if row.get("media_type"):
                    media += 1
                    media_path = row.get("file")
                    if media_path:
                        candidate = (path.parent / str(media_path)).resolve()
                        try:
                            candidate.relative_to(path.parent.resolve())
                        except ValueError:
                            problems += 1
                            print(f"verify: media path escapes export: {path} -> {media_path}", file=sys.stderr)
                        else:
                            if not candidate.is_file() or candidate.stat().st_size == 0:
                                problems += 1
                                print(f"verify: missing/empty media: {path} -> {media_path}", file=sys.stderr)
                            else:
                                actual_size = candidate.stat().st_size
                                expected_size = row.get("media_size")
                                if expected_size is not None:
                                    try:
                                        if actual_size != int(expected_size):
                                            problems += 1
                                            print(
                                                f"verify: media size mismatch in {path}: "
                                                f"{actual_size} != {expected_size}",
                                                file=sys.stderr,
                                            )
                                    except (TypeError, ValueError):
                                        problems += 1
                                        print(f"verify: invalid media_size in {path}", file=sys.stderr)
                                expected_hash = row.get("media_sha256")
                                if expected_hash:
                                    try:
                                        actual_hash = sha256_file(candidate)
                                    except OSError as exc:
                                        problems += 1
                                        print(f"verify: could not hash media {candidate}: {exc}", file=sys.stderr)
                                    else:
                                        if actual_hash != str(expected_hash):
                                            problems += 1
                                            print(f"verify: media hash mismatch: {path} -> {media_path}", file=sys.stderr)
                    elif row.get("media_skipped"):
                        skipped += 1
                        if not args.allow_skipped:
                            problems += 1
                            print(
                                f"verify: skipped media in {path}: {row['media_skipped']}",
                                file=sys.stderr,
                            )
                    elif row.get("media_error"):
                        skipped += 1
                        if not args.allow_media_errors:
                            problems += 1
                            print(
                                f"verify: media download error in {path}: {row['media_error']}",
                                file=sys.stderr,
                            )
                    else:
                        problems += 1
                        print(f"verify: media message has neither file nor skip reason: {path}", file=sys.stderr)
        file_summaries[path.resolve()] = summary

    partials = [path for path in root.rglob(".partial-*") if path.is_dir()]
    if partials:
        problems += len(partials)
        for path in partials:
            print(f"verify: abandoned partial export: {path}", file=sys.stderr)
    if args.check_db:
        db_path = Path(args.db).expanduser()
        if not db_path.is_file():
            problems += 1
            print(f"verify: database does not exist: {db_path}", file=sys.stderr)
        else:
            conn = open_db(db_path)
            try:
                rows = conn.execute(f"SELECT * FROM {EXPORTS_TABLE}").fetchall()
                ledger_by_path = {
                    Path(str(row["output_path"])).resolve(): row
                    for row in rows
                    if path_is_under(Path(str(row["output_path"])), root)
                }
                for result_path in file_summaries:
                    if result_path.parent not in ledger_by_path:
                        problems += 1
                        print(f"verify: completed export is absent from ledger: {result_path.parent}", file=sys.stderr)
                for row in rows:
                    export_path = Path(str(row["output_path"]))
                    if not path_is_under(export_path, root):
                        continue
                    summary = file_summaries.get((export_path / "result.json").resolve())
                    if not export_path.is_dir() or not (export_path / "result.json").is_file():
                        problems += 1
                        print(f"verify: ledger export is missing: {export_path}", file=sys.stderr)
                    elif summary is None:
                        problems += 1
                        print(f"verify: ledger export JSON could not be summarized: {export_path}", file=sys.stderr)
                    else:
                        if int(row["message_count"] or 0) != int(summary["count"]):
                            problems += 1
                            print(
                                f"verify: ledger message count mismatch for {export_path}: "
                                f"{row['message_count']} != {summary['count']}",
                                file=sys.stderr,
                            )
                        ids = [message_id for values in summary["keys"].values() for message_id in values]
                        dates = summary["dates"]
                        if ids and int(row["last_message_id"] or 0) != max(ids):
                            problems += 1
                            print(f"verify: ledger last message ID mismatch for {export_path}", file=sys.stderr)
                        if dates and int(row["last_message_unix"] or 0) != max(dates):
                            problems += 1
                            print(f"verify: ledger last message date mismatch for {export_path}", file=sys.stderr)
                    if row["indexed_unix"] is None and not args.allow_unindexed:
                        problems += 1
                        print(f"verify: export is not indexed: {export_path}", file=sys.stderr)
                    if row["applied_unix"] is None:
                        problems += 1
                        print(f"verify: export watermark is not applied: {export_path}", file=sys.stderr)
                    target = conn.execute(
                        f"SELECT last_message_id, last_message_unix FROM {TARGETS_TABLE} WHERE target_key = ?",
                        (row["target_key"],),
                    ).fetchone()
                    if target is None:
                        problems += 1
                        print(f"verify: export target is missing from target ledger: {export_path}", file=sys.stderr)
                    elif row["applied_unix"] is not None:
                        if row["last_message_id"] is not None and int(target["last_message_id"] or 0) < int(row["last_message_id"]):
                            problems += 1
                            print(f"verify: target watermark is behind export {export_path}", file=sys.stderr)
                        if row["last_message_unix"] is not None and int(target["last_message_unix"] or 0) < int(row["last_message_unix"]):
                            problems += 1
                            print(f"verify: target date watermark is behind export {export_path}", file=sys.stderr)
                    if summary is not None and row["indexed_unix"] is not None:
                        for chat_id, message_ids in summary["keys"].items():
                            placeholders = ",".join("?" for _ in message_ids)
                            db_ids = {
                                int(item[0])
                                for item in conn.execute(
                                    f"SELECT message_id FROM messages WHERE chat_id = ? AND COALESCE(is_deleted, 0)=0 AND message_id IN ({placeholders})",
                                    (chat_id, *sorted(message_ids)),
                                )
                            }
                            missing_ids = sorted(message_ids - db_ids)
                            if missing_ids:
                                problems += 1
                                print(
                                    f"verify: indexed database is missing {len(missing_ids)} message(s) from {export_path} "
                                    f"for chat {chat_id}",
                                    file=sys.stderr,
                                )
            finally:
                conn.close()
    print(
        f"verify: {len(result_files)} export(s), {messages} message(s), "
        f"{media} media record(s), {skipped} skipped"
    )
    if problems:
        print(f"verify: FAILED with {problems} problem(s)", file=sys.stderr)
        return 1
    print("verify: OK")
    return 0


async def doctor_network(args: argparse.Namespace) -> int:
    client = await connect_client(args)
    conn = open_db(Path(args.db).expanduser())
    problems = 0
    try:
        me = await client.get_me()
        print(f"Telegram session: OK ({sender_label(me, getattr(me, 'id', None), False)})")
        for target in load_targets(conn, active_only=True):
            try:
                await client.get_entity(target_input_peer(target))
                print(f"peer: OK {target.source_name} -> {target.peer_kind}:{target.peer_id}")
            except Exception as exc:
                problems += 1
                print(f"peer: ERROR {target.source_name}: {exc}", file=sys.stderr)
    finally:
        conn.close()
        await client.disconnect()
    return 1 if problems else 0


async def doctor(args: argparse.Namespace) -> int:
    result = doctor_offline(args)
    if args.network:
        try:
            network_result = await doctor_network(args)
        except (ExportError, OSError) as exc:
            print(f"network: ERROR: {exc}", file=sys.stderr)
            network_result = 1
        result = max(result, network_result)
    return result


async def map_targets(args: argparse.Namespace) -> None:
    client = await connect_client(args)
    conn = open_db(Path(args.db).expanduser())
    try:
        active = active_chats(conn)
        all_chats = database_chats(conn)
        if args.map_all and (args.name or args.peer or args.chat_id):
            raise ExportError("--all cannot be combined with --name, --peer, or --chat-id")
        if not args.map_all and not active:
            raise ExportError("No active chats are selected in the database")
        if args.name and not args.peer:
            raise ExportError("--name requires --peer")
        if args.peer and not args.name:
            raise ExportError("--peer requires --name")

        if args.name:
            matches = [
                item for item in active
                if normalized_chat_name(item.name) == normalized_chat_name(args.name)
            ]
            if not matches:
                raise ExportError(
                    f"{args.name!r} is not currently marked active in the master database"
                )
            if len(matches) > 1 and not args.chat_id:
                raise ExportError(
                    f"{args.name!r} appears in {len(matches)} active database rows; "
                    "pass --chat-id to choose a stable mapping ID"
                )
            if args.chat_id:
                selected = [item for item in matches if item.chat_id == args.chat_id]
                if not selected:
                    raise ExportError(
                        f"{args.chat_id!r} is not an active database chat named {args.name!r}"
                    )
            else:
                selected = matches
            entity = await resolve_peer(client, args.peer)
            description = entity_description(entity)
            cache_dialog(conn, description)
            canonical_chat_id = args.chat_id or selected[0].chat_id
            target = upsert_target(
                conn,
                selected[0].name,
                entity,
                Path(args.output).expanduser(),
                canonical_chat_id,
                commit=False,
            )
            for chat in selected:
                link_target_chat(conn, target.target_key, chat.chat_id, "explicit")
            conn.commit()
            print(f"Mapped {target.source_name!r} to {target.title!r} as {target.chat_id}")
            return

        dialogs = await client.get_dialogs(limit=None)
        candidates: list[tuple[Any, tuple[str, str, Optional[str], int, Optional[int], str]]] = []
        for dialog in dialogs:
            try:
                candidates.append((dialog.entity, entity_description(dialog.entity)))
            except ExportError:
                continue
        scope = all_chats if args.map_all else active
        mapped, linked, discovered, unresolved = auto_map_database_chats(
            conn,
            scope,
            candidates,
            Path(args.output).expanduser(),
            include_unmatched_dialogs=args.map_all,
        )
        mode = "all Telegram dialogs" if args.map_all else "active database chats"
        print(
            f"Mapping complete for {mode}: stored {len(candidates)} dialog(s), "
            f"updated {mapped} target(s), linked {linked} database chat row(s), "
            f"added {discovered} unbacked chat placeholder(s)."
        )
        if unresolved:
            for chat, reason in unresolved[:50]:
                print(f"Unresolved {chat.name!r} [{chat.chat_id}]: {reason}")
            if len(unresolved) > 50:
                print(f"... {len(unresolved) - 50} more unresolved row(s) omitted")
            print(
                f"{len(unresolved)} database chat row(s) still need an explicit "
                "--name/--peer/--chat-id mapping. No ambiguous match was guessed."
            )
    finally:
        conn.close()
        await client.disconnect()


def invoke_indexer(output_root: Path, db_path: Path) -> int:
    """Run the indexer and return its status without hiding parser/ingest errors."""
    command = [
        sys.executable,
        "-m",
        "tgbackup.database.importer",
        str(output_root),
        "--export-db",
        str(db_path),
    ]
    print("Indexing completed exports...")
    return subprocess.run(command, check=False).returncode


def validate_pending_export_paths(rows: list[sqlite3.Row]) -> None:
    missing = [str(row["output_path"]) for row in rows if not Path(str(row["output_path"]), "result.json").is_file()]
    if missing:
        raise ExportError(
            "pending export files are missing; refusing to advance watermarks: "
            + ", ".join(missing[:3])
        )


def partial_path(target: Target, target_dir: Path, baseline_id: Optional[int], full_rescan: bool) -> Path:
    baseline = "full" if full_rescan else str(baseline_id if baseline_id is not None else "none")
    return target_dir / f".partial-{safe_component(target.target_key)}-{baseline}"


async def run_exports(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser().resolve()
    if args.run_all and args.target:
        raise ExportError("--all cannot be combined with --target")
    if args.run_all and args.chat_output_dir:
        raise ExportError("--all cannot be combined with --chat-output-dir")
    direct_output = (
        Path(args.chat_output_dir).expanduser().resolve()
        if args.chat_output_dir
        else None
    )
    if direct_output is not None and not args.target:
        raise ExportError("--chat-output-dir requires one explicit --target")
    output_root = direct_output or Path(args.output).expanduser().resolve()
    if args.max_messages is not None and args.max_messages <= 0:
        raise ExportError("--max-messages must be a positive integer")
    if args.max_messages is not None and not args.dry_run:
        raise ExportError("--max-messages is restricted to --dry-run so a test cannot advance a real watermark")
    mount_point = infer_mount_point(output_root)
    if mount_point is not None and not mount_point.is_mount():
        raise ExportError(f"refusing to write backup: expected mountpoint is not mounted: {mount_point}")
    if direct_output is not None:
        if not output_root.is_dir():
            raise ExportError(f"--chat-output-dir must be an existing directory: {output_root}")
    else:
        ensure_private_dir(output_root)
    # Lock beside the database rather than only inside the output root.  This
    # prevents two runs using different output arguments from racing the same
    # watermark ledger.
    lock = ExportLock(db_path.parent / f".{db_path.name}.tgbackman.lock")
    lock.acquire()
    conn: Optional[sqlite3.Connection] = None
    client: Optional[Any] = None
    exported = 0
    tested = 0
    failures = 0
    try:
        conn = open_db(db_path)
        if not args.dry_run:
            pruned_staging = prune_completed_staging(conn)
            if pruned_staging:
                print(
                    f"Pruned {pruned_staging:,} staged message record(s) from completed runs."
                )
        if args.legacy_json_export and args.index and not args.dry_run:
            pending = pending_exports(conn, output_root)
            if pending:
                validate_pending_export_paths(pending)
                conn.close()
                conn = None
                if invoke_indexer(output_root, db_path):
                    print("Pending export indexing failed; watermarks were not advanced.", file=sys.stderr)
                    return 1
                conn = open_db(db_path)
                mark_exports_indexed(conn, output_root)
                apply_export_watermarks(
                    conn,
                    require_indexed=True,
                    output_root=output_root,
                    activate_chat=not args.run_all,
                )
        elif args.legacy_json_export and not args.index and not args.dry_run:
            blocking = [row for row in pending_exports(conn, output_root) if row["applied_unix"] is None]
            if blocking:
                raise ExportError(
                    "cannot use --no-index while an earlier export has an unapplied watermark; "
                    "run once with --index to reconcile it first"
                )

        all_targets = load_targets(conn)
        blocked_keys = blacklisted_target_keys(conn)
        if args.target:
            wanted = args.target.strip().casefold()
            requested = [
                target for target in all_targets
                if target.source_name.strip().casefold() == wanted
                or target.target_key.casefold() == wanted
            ]
            if any(target.target_key in blocked_keys for target in requested):
                names = ", ".join(sorted(target.target_key for target in requested if target.target_key in blocked_keys))
                raise ExportError(
                    f"Refusing to back up blacklisted target(s): {names}. "
                    "Remove the blacklist rule in tgbackman or with `blacklist-chat --remove`."
                )
        targets = runnable_targets(conn, include_inactive=args.run_all)
        scoped_names = [target.source_name.strip().casefold() for target in targets]
        duplicate_scoped_names = {name for name in scoped_names if scoped_names.count(name) > 1}
        if args.target:
            wanted = args.target.strip().casefold()
            if wanted in duplicate_scoped_names:
                raise ExportError(
                    f"{args.target!r} is an ambiguous active display name; use the unique target key from `targets`"
                )
            targets = [
                target
                for target in targets
                if target.source_name.strip().casefold() == wanted or target.target_key.casefold() == wanted
            ]
        if not targets:
            scope = "all chats" if args.run_all else "active chats"
            raise ExportError(f"No enabled mapped targets correspond to {scope}")
        if direct_output is not None and len(targets) != 1:
            raise ExportError("--chat-output-dir must resolve to exactly one target key")

        config_values = parse_env_file(Path(args.config).expanduser())
        media_value = args.media or os.environ.get("TG_MEDIA", config_values.get("TG_MEDIA", "all"))
        selected_media = parse_media_selection(media_value)
        configured_size = os.environ.get("TG_MAX_FILE_SIZE", config_values.get("TG_MAX_FILE_SIZE", "0"))
        try:
            max_file_size = args.max_file_size if args.max_file_size is not None else parse_size(configured_size)
        except argparse.ArgumentTypeError as exc:
            raise ExportError(f"Invalid TG_MAX_FILE_SIZE: {exc}") from exc
        if args.media_retries < 0:
            raise ExportError("--media-retries cannot be negative")
        if args.request_delay < 0:
            raise ExportError("--request-delay cannot be negative")
        if args.progress_interval <= 0:
            raise ExportError("--progress-interval must be greater than zero")
        if args.progress_every <= 0:
            raise ExportError("--progress-every must be a positive integer")
        if args.full_rescan and args.max_messages:
            raise ExportError("--full-rescan cannot be combined with --max-messages; a partial walk cannot reconcile deletions")
        if args.overlap_ids < 0 or args.overlap_days < 0:
            raise ExportError("--overlap-ids and --overlap-days cannot be negative")
        download_media_enabled = not args.dry_run or args.download_media

        client = await connect_client(args)
        target_total = len(targets)
        summaries: list[dict[str, Any]] = []
        scope_label = (
            "enabled non-blacklisted mapped chat(s), including inactive chats"
            if args.run_all else "mapped active non-blacklisted chat(s)"
        )
        print(f"Starting backup for {target_total} {scope_label}.")
        for target_position, target in enumerate(targets, 1):
            print(
                f"[{target_position}/{target_total} {target.source_name}] "
                f"resolving Telegram peer {target.peer_kind}:{target.peer_id}..."
            )
            staging_dir: Optional[Path] = None
            staging_preexisting = False
            progress: Optional[ProgressReporter] = None
            try:
                entity = await client.get_input_entity(target_input_peer(target))
                chat_entity_snapshot: Optional[dict[str, Any]] = None
                chat_full_snapshot: Optional[dict[str, Any]] = None
                if (
                    not args.legacy_json_export
                    and not args.dry_run
                    and hasattr(client, "get_entity")
                ):
                    try:
                        chat_entity = await client.get_entity(entity)
                        chat_entity_snapshot = tl_object_envelope(
                            chat_entity,
                            require_binary=True,
                        )
                    except Exception as exc:
                        raise ExportError(
                            f"could not capture complete chat entity for {target.source_name}: {exc}"
                        ) from exc
                    if chat_entity_snapshot is None:
                        raise ExportError(
                            f"Telegram returned no serializable chat entity for {target.source_name}"
                        )
                    if bool(getattr(chat_entity, "min", False)):
                        raise ExportError(
                            f"Telegram returned only a minimal chat entity for {target.source_name}"
                        )
                    if (
                        chat_entity_snapshot.get("peer_kind") != target.peer_kind
                        or int(chat_entity_snapshot.get("peer_id") or 0) != int(target.peer_id)
                    ):
                        raise ExportError(
                            f"Telegram returned the wrong chat entity for {target.source_name}: "
                            f"expected {target.peer_kind}:{target.peer_id}, got "
                            f"{chat_entity_snapshot.get('peer_kind')}:"
                            f"{chat_entity_snapshot.get('peer_id')}"
                        )
                    print(
                        f"[{target.source_name}] capturing complete chat metadata..."
                    )
                    chat_full_snapshot = await full_chat_metadata(
                        client,
                        entity,
                        target,
                        args.request_delay,
                    )
                baseline_id, baseline_unix = baseline_for_target(conn, target)
                print(
                    f"[{target.source_name}] baseline message_id={baseline_id or 'none'} "
                    f"date={datetime.fromtimestamp(baseline_unix, timezone.utc).isoformat() if baseline_unix else 'none'}"
                )
                if baseline_id is None and baseline_unix is not None and args.overlap_days == 0:
                    print(
                        f"[{target.source_name}] date-only boundary: safely re-reading its final second; "
                        "the database upsert removes that minimal overlap",
                        file=sys.stderr,
                    )
                target_dir = (
                    direct_target_output_dir(output_root, target, write_marker=not args.dry_run)
                    if direct_output is not None
                    else target_output_dir(output_root, target)
                )
                if not args.legacy_json_export and not args.dry_run:
                    ensure_private_dir(target_dir)
                    run_key = database_run_key(
                        target,
                        baseline_id,
                        baseline_unix,
                        args,
                        effective_media=",".join(sorted(selected_media)),
                        effective_max_file_size=max_file_size,
                    )
                    resumed_messages = int(
                        conn.execute(
                            f"SELECT count(*) FROM {RUN_MESSAGES_TABLE} WHERE run_key=?",
                            (run_key,),
                        ).fetchone()[0]
                    )
                    resume_after_id, resumed_messages = staged_resume_after_id(
                        conn, run_key, target_dir
                    )
                    progress = ProgressReporter(
                        target.source_name,
                        interval=args.progress_interval,
                        every=args.progress_every,
                        enabled=not args.no_progress,
                        chat_position=target_position,
                        chat_total=target_total,
                    )
                    progress.start("direct-to-database backup", resumed_messages)
                    if resume_after_id is not None:
                        progress.phase(
                            f"reusing {resumed_messages:,} staged message(s); "
                            f"fetching messages after ID {resume_after_id:,}"
                        )
                    record_iter = iter_message_records(
                        client,
                        entity,
                        baseline_id,
                        baseline_unix,
                        target_dir / "media",
                        args.overlap_ids,
                        args.overlap_days * 86400,
                        args.allow_media_errors,
                        selected_media,
                        max_file_size,
                        True,
                        args.media_retries,
                        args.discard_overlap,
                        args.full_rescan,
                        args.max_messages,
                        args.request_delay,
                        progress,
                        resume_after_id,
                    )
                    stats = await write_database_stream(
                        conn,
                        target,
                        record_iter,
                        target_dir,
                        run_key,
                        baseline_id,
                        baseline_unix,
                        args.full_rescan,
                        progress,
                        activate_chat=not args.run_all,
                        chat_entity_snapshot=chat_entity_snapshot,
                        chat_full_snapshot=chat_full_snapshot,
                    )
                    if stats.message_count == 0:
                        progress.finish("no new messages; database unchanged")
                        summaries.append(
                            {
                                "position": target_position,
                                "name": target.source_name,
                                "status": "no new messages",
                                "messages": 0,
                            }
                        )
                        print(f"[{target.source_name}] no new messages")
                        continue
                    exported += stats.message_count
                    progress.finish(
                        f"commit complete; watermark advanced to message {stats.last_message_id}"
                    )
                    summaries.append(
                        {
                            "position": target_position,
                            "name": target.source_name,
                            "status": "committed",
                            "messages": stats.message_count,
                            "path": str(target_dir),
                        }
                    )
                    print(
                        f"[{target.source_name}] committed {stats.message_count} messages "
                        f"directly to {db_path} ({stats.media_count - stats.skipped_media_count} "
                        f"media, {stats.skipped_media_count} skipped); media root: {target_dir / 'media'}"
                    )
                    if stats.media_errors:
                        print(f"[{target.source_name}] media warnings: {len(stats.media_errors)}")
                    continue
                if args.dry_run:
                    staging_dir = Path(tempfile.mkdtemp(prefix=".dry-run-", dir=target_dir))
                else:
                    staging_dir = partial_path(target, target_dir, baseline_id, args.full_rescan)
                    staging_preexisting = staging_dir.exists()
                    ensure_private_dir(staging_dir)
                    state_path = staging_dir / ".partial_state.json"
                    if not state_path.exists():
                        state_path.write_text(
                            json.dumps(
                                {
                                    "target_key": target.target_key,
                                    "baseline_id": baseline_id,
                                    "baseline_unix": baseline_unix,
                                    "full_rescan": args.full_rescan,
                                },
                                indent=2,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                media_root = staging_dir / "media"
                progress = ProgressReporter(
                    target.source_name,
                    interval=args.progress_interval,
                    every=args.progress_every,
                    enabled=not args.no_progress,
                    chat_position=target_position,
                    chat_total=target_total,
                )
                progress.start(
                    "dry-run" if args.dry_run else "legacy JSON backup",
                    0,
                )
                record_iter = iter_message_records(
                    client,
                    entity,
                    baseline_id,
                    baseline_unix,
                    media_root,
                    args.overlap_ids,
                    args.overlap_days * 86400,
                    args.allow_media_errors,
                    selected_media,
                    max_file_size,
                    download_media_enabled,
                    args.media_retries,
                    args.discard_overlap,
                    args.full_rescan,
                    args.max_messages,
                    args.request_delay,
                    progress,
                )
                final_dir, stats = await write_export_stream(
                    target,
                    record_iter,
                    output_root,
                    staging_dir,
                    target_dir_override=target_dir,
                )
                if final_dir is None:
                    if not args.dry_run:
                        record_chat_backup_run(conn, target, "completed_no_new_messages")
                        conn.commit()
                    if args.dry_run or not staging_preexisting:
                        remove_staging(staging_dir)
                        staging_dir = None
                    progress.finish("no new messages")
                    summaries.append(
                        {
                            "position": target_position,
                            "name": target.source_name,
                            "status": "no new messages",
                            "messages": 0,
                        }
                    )
                    print(f"[{target.source_name}] no new messages")
                    continue
                staging_dir = None
                if args.dry_run:
                    remove_staging(final_dir)
                    tested += stats.message_count
                    progress.finish("dry-run complete; temporary files removed")
                    summaries.append(
                        {
                            "position": target_position,
                            "name": target.source_name,
                            "status": "dry-run",
                            "messages": stats.message_count,
                        }
                    )
                    print(f"[{target.source_name}] dry-run: would export {stats.message_count} messages")
                    continue
                record_export(conn, target, final_dir, stats)
                if not args.index:
                    apply_export_watermarks(
                        conn,
                        output_root=output_root,
                        export_keys={export_key(final_dir)},
                        allow_unindexed=True,
                        activate_chat=not args.run_all,
                    )
                exported += stats.message_count
                progress.finish(f"legacy JSON export complete: {final_dir}")
                summaries.append(
                    {
                        "position": target_position,
                        "name": target.source_name,
                        "status": "committed",
                        "messages": stats.message_count,
                        "path": str(final_dir),
                    }
                )
                print(
                    f"[{target.source_name}] wrote {stats.message_count} messages "
                    f"({stats.media_count - stats.skipped_media_count} media, "
                    f"{stats.skipped_media_count} skipped) to {final_dir}"
                )
                if stats.media_errors:
                    print(f"[{target.source_name}] media warnings: {len(stats.media_errors)}")
            except Exception as exc:
                if progress:
                    progress.fail(exc)
                if staging_dir is not None:
                    if args.dry_run:
                        remove_staging(staging_dir)
                    else:
                        print(f"[{target.source_name}] partial export preserved at {staging_dir}", file=sys.stderr)
                if not args.dry_run:
                    record_chat_backup_run(conn, target, "failed")
                    conn.commit()
                wait_seconds = flood_wait_seconds(exc)
                if wait_seconds is not None:
                    print(f"Telegram requested a {wait_seconds}s wait before continuing.", file=sys.stderr)
                    await asyncio.sleep(wait_seconds)
                failures += 1
                summaries.append(
                    {
                        "position": target_position,
                        "name": target.source_name,
                        "status": "failed",
                        "messages": progress.processed if progress else 0,
                        "error": str(exc),
                    }
                )
                print(f"[{target.source_name}] ERROR: {exc}", file=sys.stderr)
        if exported and args.legacy_json_export and args.index and not args.dry_run:
            if conn is not None:
                conn.close()
                conn = None
            if invoke_indexer(output_root, db_path):
                failures += 1
                print("Indexer failed; export ledger remains pending and watermarks were not advanced.", file=sys.stderr)
            else:
                conn = open_db(db_path)
                try:
                    mark_exports_indexed(conn, output_root)
                    apply_export_watermarks(
                        conn,
                        require_indexed=True,
                        output_root=output_root,
                        activate_chat=not args.run_all,
                    )
                finally:
                    conn.close()
                    conn = None
        if summaries:
            committed = sum(item["status"] == "committed" for item in summaries)
            no_new = sum(item["status"] == "no new messages" for item in summaries)
            failed = sum(item["status"] == "failed" for item in summaries)
            print(
                f"All-chat summary: {len(summaries)}/{target_total} processed; "
                f"committed={committed}, no-new={no_new}, failed={failed}."
            )
            for item in summaries:
                suffix = f" ({item['error']})" if item.get("error") else ""
                print(
                    f"  [{item['position']}/{target_total}] {item['name']}: "
                    f"{item['status']}, {item['messages']:,} message(s){suffix}"
                )
    finally:
        if client is not None:
            await client.disconnect()
        if conn is not None:
            conn.close()
        lock.release()

    if failures:
        return 1
    if args.dry_run:
        print(f"Dry-run complete: {tested} message(s) would be exported.")
        return 0
    print(f"Incremental export complete: {exported} message(s).")
    return 0


def infer_mount_point(output: Path) -> Optional[Path]:
    """Infer a removable-media mountpoint for the common /media/user/volume layout."""
    resolved = output.resolve()
    parts = resolved.parts
    if len(parts) >= 4 and parts[1] == "media":
        return Path("/") / parts[1] / parts[2] / parts[3]
    if len(parts) >= 3 and parts[1] == "mnt":
        return Path("/") / parts[1] / parts[2]
    if len(parts) >= 5 and parts[1:3] == ("run", "media"):
        return Path("/") / parts[1] / parts[2] / parts[3] / parts[4]
    existing = resolved
    while existing != existing.parent:
        if existing.is_mount():
            return existing
        existing = existing.parent
    return None


def install_systemd_example(args: argparse.Namespace) -> None:
    unit_dir = Path(args.unit_dir).expanduser()
    ensure_private_dir(unit_dir)
    # Invoke the installed package through the same interpreter that ran the
    # generator.  This keeps generated units independent of repository-root
    # compatibility launchers and of the user's PATH.
    db = Path(args.db).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    session = Path(args.session).expanduser().resolve()
    python = Path(sys.executable).resolve()
    mount_point = Path(args.mount_point).expanduser().resolve() if args.mount_point else infer_mount_point(output)
    command = shlex.join(
        [
            str(python),
            "-m",
            "tgbackup",
            "--db",
            str(db),
            "--config",
            str(config),
            "--session",
            str(session),
            "run",
            "--output",
            str(output),
        ]
    )
    mount_lines = ""
    if mount_point:
        mount_lines = f"ConditionPathIsMountPoint={mount_point}\nRequiresMountsFor={mount_point}\n"
    service = f"""[Unit]
Description=tgbackman incremental Telegram backup
{mount_lines}

[Service]
Type=oneshot
UMask=0077
TimeoutStartSec=infinity
WorkingDirectory={SCRIPT_DIR}
ExecStart={command}
"""
    timer = """[Unit]
Description=Run tgbackman incremental Telegram backup daily

[Timer]
OnCalendar=*-*-* 03:30:00
Persistent=true
RandomizedDelaySec=15m

[Install]
WantedBy=timers.target
"""
    (unit_dir / "tgbackman-telegram-backup.service").write_text(service, encoding="utf-8")
    (unit_dir / "tgbackman-telegram-backup.timer").write_text(timer, encoding="utf-8")
    print(f"Wrote units to {unit_dir}")
    if mount_point:
        print(f"Backup mount guard: {mount_point}")
    print("Enable with:")
    print("  systemctl --user daemon-reload")
    print("  systemctl --user enable --now tgbackman-telegram-backup.timer")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tgbackman-backup",
        description="Incremental Telegram user-account backup and current-db index integration.",
    )
    parser.add_argument("--db", default=os.environ.get("TGBACKMAN_DB", str(DEFAULT_DB)), help="Master SQLite database")
    parser.add_argument("--config", default=os.environ.get("TGBACKMAN_CONFIG", str(DEFAULT_CONFIG)), help="Credentials env file")
    parser.add_argument("--session", default=os.environ.get("TGBACKMAN_SESSION", str(DEFAULT_SESSION)), help="Telethon session path")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common_after_command(command_parser: argparse.ArgumentParser) -> None:
        # argparse normally requires global options before the subcommand.  We
        # also accept them after it because that is friendlier for copied CLI
        # commands and systemd unit generation.  SUPPRESS preserves any value
        # already parsed by the parent parser.
        command_parser.add_argument("--db", default=argparse.SUPPRESS, help="Master SQLite database")
        command_parser.add_argument("--config", default=argparse.SUPPRESS, help="Credentials env file")
        command_parser.add_argument("--session", default=argparse.SUPPRESS, help="Telethon session path")

    configure = sub.add_parser("configure", help="Securely save API ID and API hash")
    add_common_after_command(configure)
    configure.add_argument("--api-id", help="API ID (prompted if omitted)")

    auth = sub.add_parser("auth", help="Authenticate once and create the local user session")
    add_common_after_command(auth)
    targets = sub.add_parser("targets", help="List saved target mappings and watermarks")
    add_common_after_command(targets)
    dialogs = sub.add_parser("dialogs", help="List Telegram dialogs for mapping")
    add_common_after_command(dialogs)

    doctor_parser = sub.add_parser("doctor", help="Check credentials, database, mappings, output, and optionally the network")
    add_common_after_command(doctor_parser)
    doctor_parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Incremental export root to test")
    doctor_parser.add_argument("--network", action="store_true", help="Authenticate and resolve Telegram peers")

    verify_parser = sub.add_parser("verify", help="Verify completed exports and all downloaded media paths")
    add_common_after_command(verify_parser)
    verify_parser.add_argument("--output", required=True, help="Incremental export root to verify")
    verify_parser.add_argument(
        "--allow-skipped",
        action="store_true",
        help="Treat intentionally skipped media (size/category policy) as warnings instead of failures",
    )
    verify_parser.add_argument(
        "--allow-media-errors",
        action="store_true",
        help="Allow media records with a recorded download error",
    )
    verify_parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow the same chat/message ID to appear in multiple range exports",
    )
    verify_parser.add_argument(
        "--check-db",
        action="store_true",
        help="Also validate the export ledger and applied/indexed state in --db",
    )
    verify_parser.add_argument(
        "--allow-unindexed",
        action="store_true",
        help="Allow ledger entries that have not yet been indexed",
    )

    mapping = sub.add_parser("map", help="Map an active database chat to a Telegram dialog")
    add_common_after_command(mapping)
    mapping.add_argument("--name", help="Active database chat name")
    mapping.add_argument("--peer", help="Telegram username, numeric ID, or t.me peer")
    mapping.add_argument("--chat-id", help="Optional stable ID to reuse for indexing")
    mapping.add_argument(
        "--all",
        dest="map_all",
        action="store_true",
        help="Cache every Telegram dialog and map all safely matchable database chats, including inactive ones",
    )
    mapping.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Incremental export root")

    run = sub.add_parser("run", help="Export enabled mapped chats")
    add_common_after_command(run)
    run.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Incremental export root")
    run.add_argument(
        "--all",
        dest="run_all",
        action="store_true",
        help="Export every enabled mapped Telegram target, regardless of chats.is_active",
    )
    run.add_argument("--target", help="Run one source name or target key")
    run.add_argument(
        "--chat-output-dir",
        help="For one --target, use this existing chat directory as the stable media root",
    )
    run.add_argument(
        "--media",
        help="Media selection: all, or comma-separated photos,videos,voice,audio,files,stickers,animations",
    )
    run.add_argument(
        "--max-file-size",
        type=parse_size,
        default=None,
        help="Maximum media file size (0/unset means unlimited; accepts 500M, 4G, 4GiB)",
    )
    run.add_argument(
        "--media-retries",
        type=int,
        default=3,
        help="Retry attempts per media file after a download error (default: 3)",
    )
    run.add_argument(
        "--request-delay",
        type=float,
        default=1.0,
        help="Seconds between Telegram history requests (default: 1.0; increase if rate-limited)",
    )
    run.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="Maximum seconds between progress lines while work is advancing (default: 5)",
    )
    run.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Also report after this many processed messages (default: 100)",
    )
    run.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable periodic progress lines (errors and final summaries remain)",
    )
    run.add_argument("--overlap-ids", type=int, default=0, help="Older IDs to re-read for media/edit retries (default: 0, exact boundary)")
    run.add_argument("--overlap-days", type=int, default=0, help="Date overlap window for date-only baselines (default: 0, exact)")
    run.add_argument(
        "--discard-overlap",
        action="store_true",
        help="When an overlap is requested, omit IDs at/before the stored watermark from processing",
    )
    run.add_argument("--max-messages", type=int, help="Limit messages per target (testing only)")
    run.add_argument("--full-rescan", action="store_true", help="Ignore watermarks and walk full history")
    run.add_argument("--allow-media-errors", action="store_true", help="Write messages even if media cannot download")
    run.add_argument("--dry-run", action="store_true", help="Fetch and count without writing DB state or lasting files")
    run.add_argument(
        "--download-media",
        action="store_true",
        help="With --dry-run, actually download media into temporary files before cleanup",
    )
    run.add_argument(
        "--legacy-json-export",
        action="store_true",
        help="Write dated result.json folders and index them instead of committing directly to SQLite",
    )
    run.add_argument("--index", dest="index", action="store_true", help="Index legacy JSON exports (the default)")
    run.add_argument("--no-index", dest="index", action="store_false", help="Do not index legacy JSON exports")
    run.set_defaults(index=True)

    purge = sub.add_parser(
        "purge-chat",
        help="Safely remove one mapped chat, its aliases, and optionally its unshared media",
    )
    add_common_after_command(purge)
    purge.add_argument(
        "--target",
        required=True,
        help="Exact target_key from the `targets` command (display names are not accepted)",
    )
    purge.add_argument(
        "--delete-media",
        action="store_true",
        help="Also delete unshared media and matching marker-owned chat directories",
    )
    purge.add_argument(
        "--backup-root",
        help="Hard filesystem boundary for media deletion (normally inferred from target output_dir)",
    )
    purge.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the purge plan without deleting database rows or files",
    )
    purge.add_argument(
        "--confirm",
        metavar="TARGET_KEY",
        help="Execute only when this exactly matches --target",
    )
    purge.add_argument(
        "--manifest",
        help="Write the complete purge plan as a private JSON file",
    )
    purge.add_argument(
        "--list-all",
        action="store_true",
        help="Print every planned, shared, missing, and unsafe media path",
    )

    purge_resume = sub.add_parser(
        "purge-resume",
        help="Resume media deletion for a purge whose database phase already committed",
    )
    add_common_after_command(purge_resume)
    purge_resume.add_argument("--purge-key", required=True, help="Purge ledger key printed by purge-chat")
    purge_resume.add_argument("--confirm", required=True, help="Must exactly match --purge-key")
    purge_resume.add_argument("--backup-root", help="Override the root recorded in the purge manifest")

    blacklist = sub.add_parser(
        "blacklist-chat",
        help="Prevent one mapped Telegram peer from ever being backed up",
    )
    add_common_after_command(blacklist)
    blacklist.add_argument(
        "--target",
        required=True,
        help="Exact target_key from the `targets` command",
    )
    blacklist.add_argument(
        "--remove",
        action="store_true",
        help="Remove the never-back-up rule without making the chat active",
    )
    blacklist.add_argument(
        "--reason",
        help="Optional note stored with a newly added blacklist rule",
    )

    repair_dates = sub.add_parser(
        "repair-backup-dates",
        help="Rebuild cached backup dates from API ledgers and original export evidence",
    )
    add_common_after_command(repair_dates)
    repair_dates.add_argument(
        "--backup-root",
        help="Optional boundary: only inspect backup paths beneath this directory",
    )
    repair_dates.add_argument(
        "--dry-run",
        action="store_true",
        help="Report inferred dates without updating chats",
    )
    repair_dates.add_argument(
        "--list-all",
        action="store_true",
        help="Print every chat's timestamp, confidence, source, and evidence",
    )

    install = sub.add_parser("install-systemd", help="Generate a systemd user service and daily timer")
    add_common_after_command(install)
    install.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Incremental export root")
    install.add_argument("--unit-dir", default="~/.config/systemd/user", help="User unit directory")
    install.add_argument(
        "--mount-point",
        help="Required mountpoint for the output disk (inferred for /media, /mnt, and /run/media paths)",
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    if args.command == "auth":
        client = await connect_client(args)
        await client.disconnect()
        print("Telegram session authenticated successfully.")
        return 0
    if args.command == "dialogs":
        await list_dialogs(args)
        return 0
    if args.command == "doctor":
        return await doctor(args)
    if args.command == "map":
        await map_targets(args)
        return 0
    if args.command == "run":
        return await run_exports(args)
    raise ExportError(f"Unknown async command: {args.command}")


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "configure":
            api_id = args.api_id or input("Telegram API ID: ").strip()
            api_hash = getpass.getpass("Telegram API hash: ").strip()
            write_credentials(Path(args.config).expanduser(), api_id, api_hash)
            return 0
        if args.command == "targets":
            conn = open_db(Path(args.db).expanduser())
            try:
                print_targets(conn)
            finally:
                conn.close()
            return 0
        if args.command == "verify":
            return verify_exports(args)
        if args.command == "purge-chat":
            return purge_chat_command(args)
        if args.command == "purge-resume":
            return purge_resume_command(args)
        if args.command == "blacklist-chat":
            return blacklist_chat_command(args)
        if args.command == "repair-backup-dates":
            return repair_backup_dates_command(args)
        if args.command == "install-systemd":
            install_systemd_example(args)
            return 0
        return asyncio.run(async_main(args))
    except (ExportError, KeyboardInterrupt) as exc:
        if isinstance(exc, KeyboardInterrupt):
            print("Interrupted.", file=sys.stderr)
            return 130
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
