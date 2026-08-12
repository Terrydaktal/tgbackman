"""Conversion and upsert primitives for rich archival message rows."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import time
import urllib.parse
import zlib
from datetime import datetime
from typing import Any

from ..config import TARGET_CHAT_LINKS_TABLE, TARGETS_TABLE

ARCHIVAL_MESSAGE_UPSERT_SQL = """
INSERT INTO messages (
    message_id, chat_id, sender, sender_id, timestamp, timestamp_unix, text,
    media_type, media_path, reply_to_id, reply_to_chat_id, reply_to_peer_kind,
    reply_to_peer_id, reply_to_top_id, reply_to_story_id, reply_quote_text,
    reply_quote_entities_json, reply_quote_offset, reply_media_json,
    forwarded_from, message_type,
    edit_timestamp, edit_timestamp_unix, media_size, media_sha256, media_status,
    grouped_id, entities_json, reactions_json, reply_markup_json, action_json,
    forward_json, extra_json, raw_payload, raw_tl_payload, raw_tl_sha256,
    raw_tl_layer, raw_tl_library, expanded_metadata_json, source_key, source_format
) VALUES (
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
)
ON CONFLICT(chat_id, message_id) DO UPDATE SET
    sender=CASE WHEN excluded.source_format='telegram_api'
        OR (excluded.source_format='json' AND COALESCE(messages.source_format, '')!='telegram_api')
        OR (excluded.source_format='sqlite' AND COALESCE(messages.source_format, '') NOT IN ('telegram_api','json'))
        THEN excluded.sender ELSE messages.sender END,
    sender_id=CASE WHEN excluded.source_format='telegram_api'
        OR (excluded.source_format='json' AND COALESCE(messages.source_format, '')!='telegram_api')
        OR (excluded.source_format='sqlite' AND COALESCE(messages.source_format, '') NOT IN ('telegram_api','json'))
        THEN excluded.sender_id ELSE messages.sender_id END,
    timestamp=CASE WHEN excluded.source_format='telegram_api'
        OR (excluded.source_format='json' AND COALESCE(messages.source_format, '')!='telegram_api')
        OR (excluded.source_format='sqlite' AND COALESCE(messages.source_format, '') NOT IN ('telegram_api','json'))
        THEN excluded.timestamp ELSE messages.timestamp END,
    timestamp_unix=CASE WHEN excluded.source_format='telegram_api'
        OR (excluded.source_format='json' AND COALESCE(messages.source_format, '')!='telegram_api')
        OR (excluded.source_format='sqlite' AND COALESCE(messages.source_format, '') NOT IN ('telegram_api','json'))
        THEN excluded.timestamp_unix ELSE messages.timestamp_unix END,
    text=CASE WHEN excluded.source_format='telegram_api'
        OR (excluded.source_format='json' AND COALESCE(messages.source_format, '')!='telegram_api')
        OR (excluded.source_format='sqlite' AND COALESCE(messages.source_format, '') NOT IN ('telegram_api','json'))
        THEN excluded.text ELSE messages.text END,
    media_type=CASE WHEN excluded.source_format='telegram_api'
        OR (excluded.source_format='json' AND COALESCE(messages.source_format, '')!='telegram_api')
        OR (excluded.source_format='sqlite' AND COALESCE(messages.source_format, '') NOT IN ('telegram_api','json'))
        THEN excluded.media_type ELSE messages.media_type END,
    media_path=CASE WHEN excluded.source_format='telegram_api'
        OR (excluded.source_format='json' AND COALESCE(messages.source_format, '')!='telegram_api')
        OR (excluded.source_format='sqlite' AND COALESCE(messages.source_format, '') NOT IN ('telegram_api','json'))
        THEN excluded.media_path ELSE messages.media_path END,
    reply_to_id=CASE WHEN excluded.source_format='telegram_api'
        OR (excluded.source_format='json' AND COALESCE(messages.source_format, '')!='telegram_api')
        OR (excluded.source_format='sqlite' AND COALESCE(messages.source_format, '') NOT IN ('telegram_api','json'))
        THEN excluded.reply_to_id ELSE messages.reply_to_id END,
    reply_to_chat_id=excluded.reply_to_chat_id,
    reply_to_peer_kind=excluded.reply_to_peer_kind,
    reply_to_peer_id=excluded.reply_to_peer_id,
    reply_to_top_id=excluded.reply_to_top_id,
    reply_to_story_id=excluded.reply_to_story_id,
    reply_quote_text=excluded.reply_quote_text,
    reply_quote_entities_json=excluded.reply_quote_entities_json,
    reply_quote_offset=excluded.reply_quote_offset,
    reply_media_json=excluded.reply_media_json,
    forwarded_from=CASE WHEN excluded.source_format='telegram_api'
        OR (excluded.source_format='json' AND COALESCE(messages.source_format, '')!='telegram_api')
        OR (excluded.source_format='sqlite' AND COALESCE(messages.source_format, '') NOT IN ('telegram_api','json'))
        THEN excluded.forwarded_from ELSE messages.forwarded_from END,
    message_type=excluded.message_type,
    edit_timestamp=excluded.edit_timestamp,
    edit_timestamp_unix=excluded.edit_timestamp_unix,
    media_size=excluded.media_size,
    media_sha256=excluded.media_sha256,
    media_status=excluded.media_status,
    grouped_id=excluded.grouped_id,
    entities_json=excluded.entities_json,
    reactions_json=excluded.reactions_json,
    reply_markup_json=excluded.reply_markup_json,
    action_json=excluded.action_json,
    forward_json=excluded.forward_json,
    extra_json=excluded.extra_json,
    raw_payload=excluded.raw_payload,
    raw_tl_payload=excluded.raw_tl_payload,
    raw_tl_sha256=excluded.raw_tl_sha256,
    raw_tl_layer=excluded.raw_tl_layer,
    raw_tl_library=excluded.raw_tl_library,
    expanded_metadata_json=excluded.expanded_metadata_json,
    source_key=excluded.source_key,
    source_format=excluded.source_format,
    is_deleted=0,
    deleted_unix=NULL
WHERE CASE excluded.source_format
          WHEN 'telegram_api' THEN 3 WHEN 'json' THEN 2 WHEN 'sqlite' THEN 1 ELSE 0
      END >=
      CASE messages.source_format
          WHEN 'telegram_api' THEN 3 WHEN 'json' THEN 2 WHEN 'sqlite' THEN 1 ELSE 0
      END
"""


def _json_text(value: Any) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _decode_tl_envelope(envelope: Any) -> tuple[bytes | None, str | None, int | None, str | None]:
    if not isinstance(envelope, dict):
        return None, None, None, None
    encoded = envelope.get("tl_data")
    if encoded is None:
        return None, None, envelope.get("telethon_layer"), envelope.get("telethon_version")
    if envelope.get("tl_encoding") != "base64":
        raise ValueError("unsupported TL payload encoding")
    try:
        payload = base64.b64decode(str(encoded), validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid base64 TL payload") from exc
    expected_size = envelope.get("tl_size")
    if expected_size is not None and len(payload) != int(expected_size):
        raise ValueError(f"TL payload has size {len(payload)}, expected {expected_size}")
    expected_hash = envelope.get("tl_sha256")
    digest = hashlib.sha256(payload).hexdigest()
    if expected_hash and digest != str(expected_hash):
        raise ValueError("TL payload hash mismatch")
    return (
        payload,
        digest,
        int(envelope["telethon_layer"]) if envelope.get("telethon_layer") is not None else None,
        str(envelope["telethon_version"]) if envelope.get("telethon_version") is not None else None,
    )


def _unix_from_value(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp())
        except (TypeError, ValueError):
            return None


def _peer_identity(peer: Any) -> tuple[str, int] | None:
    if not isinstance(peer, dict):
        return None
    for key, kind in (
        ("user_id", "user"),
        ("chat_id", "group"),
        ("channel_id", "channel"),
    ):
        if peer.get(key) is not None:
            return kind, abs(int(peer[key]))
    return None


def _reply_metadata(message: dict[str, Any]) -> dict[str, Any]:
    """Read normalized reply fields, falling back to the raw Telethon header."""
    raw_message = message.get("raw_message")
    header = raw_message.get("reply_to") if isinstance(raw_message, dict) else None
    if not isinstance(header, dict):
        header = {}

    metadata: dict[str, Any] = {}
    direct_fields = (
        ("reply_to_message_id", "reply_to_msg_id"),
        ("reply_to_top_id", "reply_to_top_id"),
        ("reply_to_story_id", "story_id"),
        ("reply_quote_text", "quote_text"),
        ("reply_quote_entities", "quote_entities"),
        ("reply_quote_offset", "quote_offset"),
        ("reply_media", "reply_media"),
    )
    for record_key, header_key in direct_fields:
        value = message.get(record_key)
        if value is None:
            value = header.get(header_key)
        metadata[record_key] = value

    metadata["reply_to_chat_id"] = message.get("reply_to_chat_id")
    peer_kind = message.get("reply_to_peer_kind")
    peer_id = message.get("reply_to_peer_id")
    if peer_kind is None or peer_id is None:
        peer = header.get("reply_to_peer_id") or header.get("peer")
        identity = _peer_identity(peer)
        if identity is not None:
            peer_kind, peer_id = identity
    metadata["reply_to_peer_kind"] = str(peer_kind) if peer_kind is not None else None
    metadata["reply_to_peer_id"] = int(peer_id) if peer_id is not None else None
    return metadata


def _resolve_reply_chat_id(
    conn: sqlite3.Connection,
    chat_id: str,
    metadata: dict[str, Any],
) -> str | None:
    explicit = metadata.get("reply_to_chat_id")
    if explicit:
        return str(explicit)
    message_id = metadata.get("reply_to_message_id")
    story_id = metadata.get("reply_to_story_id")
    if message_id is None and story_id is None:
        return None
    peer_kind = metadata.get("reply_to_peer_kind")
    peer_id = metadata.get("reply_to_peer_id")
    if peer_kind is None or peer_id is None:
        return chat_id if message_id is not None else None

    tables = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?, ?)",
            (TARGETS_TABLE, TARGET_CHAT_LINKS_TABLE),
        )
    }
    if TARGETS_TABLE not in tables or TARGET_CHAT_LINKS_TABLE not in tables:
        return None
    row = conn.execute(
        f"""
        WITH candidates(chat_id, priority) AS (
            SELECT chat_id, 0 FROM {TARGETS_TABLE}
             WHERE peer_kind=? AND peer_id=?
            UNION
            SELECT links.chat_id, 1
              FROM {TARGETS_TABLE} AS targets
              JOIN {TARGET_CHAT_LINKS_TABLE} AS links
                ON links.target_key=targets.target_key
             WHERE targets.peer_kind=? AND targets.peer_id=?
        )
        SELECT candidates.chat_id
          FROM candidates
          LEFT JOIN messages AS parent
            ON parent.chat_id=candidates.chat_id AND parent.message_id=?
         ORDER BY CASE WHEN parent.id IS NULL THEN 1 ELSE 0 END, candidates.priority
         LIMIT 1
        """,
        (peer_kind, peer_id, peer_kind, peer_id, message_id),
    ).fetchone()
    return str(row[0]) if row is not None else None


def backfill_reply_metadata(conn: sqlite3.Connection) -> int:
    """Normalize reply headers preserved by older API exporter versions once."""
    migration_name = "normalize_reply_metadata_v1"
    already_applied = conn.execute(
        "SELECT 1 FROM archive_schema_migrations WHERE migration_name=?",
        (migration_name,),
    ).fetchone()
    if already_applied is not None:
        return 0

    affected = 0
    rows = conn.execute(
        """SELECT id, chat_id, raw_payload
             FROM messages
            WHERE source_format='telegram_api'
              AND raw_payload IS NOT NULL"""
    )
    while batch := rows.fetchmany(1000):
        for row in batch:
            try:
                record = json.loads(zlib.decompress(bytes(row[2])).decode("utf-8"))
            except (OSError, TypeError, UnicodeDecodeError, ValueError, zlib.error):
                continue
            if not isinstance(record, dict):
                continue
            reply = _reply_metadata(record)
            if not any(
                reply.get(key) is not None
                for key in (
                    "reply_to_message_id",
                    "reply_to_story_id",
                    "reply_to_top_id",
                    "reply_quote_text",
                )
            ):
                continue
            reply_chat_id = _resolve_reply_chat_id(conn, str(row[1]), reply)
            conn.execute(
                """UPDATE messages
                      SET reply_to_chat_id=?, reply_to_peer_kind=?, reply_to_peer_id=?,
                          reply_to_top_id=?, reply_to_story_id=?, reply_quote_text=?,
                          reply_quote_entities_json=?, reply_quote_offset=?, reply_media_json=?
                    WHERE id=?""",
                (
                    reply_chat_id,
                    reply.get("reply_to_peer_kind"),
                    reply.get("reply_to_peer_id"),
                    reply.get("reply_to_top_id"),
                    reply.get("reply_to_story_id"),
                    reply.get("reply_quote_text"),
                    _json_text(reply.get("reply_quote_entities")),
                    reply.get("reply_quote_offset"),
                    _json_text(reply.get("reply_media")),
                    int(row[0]),
                ),
            )
            affected += 1
    conn.execute(
        """INSERT INTO archive_schema_migrations(
               migration_name, applied_unix, affected_rows
           ) VALUES (?, ?, ?)""",
        (migration_name, int(time.time()), affected),
    )
    return affected


def _upsert_entity_snapshot(conn: sqlite3.Connection, envelope: Any) -> str | None:
    if not isinstance(envelope, dict) or not isinstance(envelope.get("json"), dict):
        return None
    entity_json = _json_text(envelope["json"])
    if entity_json is None:
        return None
    tl_payload, tl_sha256, layer, library = _decode_tl_envelope(envelope)
    digest_input = tl_payload if tl_payload is not None else entity_json.encode("utf-8")
    snapshot_sha256 = hashlib.sha256(digest_input).hexdigest()
    expected_snapshot = envelope.get("snapshot_sha256")
    if expected_snapshot and snapshot_sha256 != str(expected_snapshot):
        raise ValueError("entity snapshot hash mismatch")
    now = int(time.time())
    conn.execute(
        """INSERT INTO telegram_entity_snapshots(
               snapshot_sha256, peer_kind, peer_id, entity_type, entity_json,
               tl_payload, tl_sha256, telethon_layer, telethon_version,
               first_captured_unix, last_captured_unix
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(snapshot_sha256) DO UPDATE SET
               last_captured_unix=excluded.last_captured_unix""",
        (
            snapshot_sha256,
            envelope.get("peer_kind"),
            envelope.get("peer_id"),
            str(envelope.get("type") or "Unknown"),
            entity_json,
            sqlite3.Binary(tl_payload) if tl_payload is not None else None,
            tl_sha256,
            layer,
            library,
            now,
            now,
        ),
    )
    return snapshot_sha256


def upsert_chat_entity_snapshot(
    conn: sqlite3.Connection,
    chat_id: str,
    envelope: Any,
    source_key: str | None,
    *,
    role: str = "entity",
) -> str | None:
    snapshot_sha256 = _upsert_entity_snapshot(conn, envelope)
    if snapshot_sha256 is None:
        return None
    conn.execute(
        """INSERT INTO telegram_chat_entity_refs(
               chat_id, snapshot_sha256, captured_unix, source_key, role
           ) VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(chat_id, snapshot_sha256) DO UPDATE SET
               captured_unix=excluded.captured_unix,
               source_key=COALESCE(excluded.source_key, telegram_chat_entity_refs.source_key),
               role=excluded.role""",
        (chat_id, snapshot_sha256, int(time.time()), source_key, role),
    )
    if source_key is not None:
        conn.execute(
            """INSERT INTO telegram_chat_snapshot_sources(
                   chat_id, snapshot_sha256, source_key, role, captured_unix
               ) VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(chat_id, snapshot_sha256, source_key, role) DO UPDATE SET
                   captured_unix=excluded.captured_unix""",
            (chat_id, snapshot_sha256, source_key, role, int(time.time())),
        )
    return snapshot_sha256


def flatten_telegram_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text") is not None:
                parts.append(str(item["text"]))
        return "".join(parts)
    return ""


def resolve_local_media_path(backup_path: str, media_path: str | None) -> str | None:
    """Resolve range-relative media once, preserving remote URLs."""
    if not media_path:
        return media_path
    value = str(media_path)
    if urllib.parse.urlparse(value).scheme:
        return value
    root = os.path.abspath(backup_path)
    candidate = os.path.abspath(value) if os.path.isabs(value) else os.path.abspath(os.path.join(root, value))
    root_real = os.path.realpath(root)
    candidate_real = os.path.realpath(candidate)
    try:
        os.path.commonpath((root, candidate))
    except ValueError as exc:
        raise ValueError(f"media path is outside its declared backup root: {value}") from exc
    if os.path.commonpath((root, candidate)) != root or os.path.commonpath((root_real, candidate_real)) != root_real:
        raise ValueError(f"media path is outside its declared backup root: {value}")
    return candidate


def archival_message_values(
    chat_id: str,
    message: dict[str, Any],
    backup_path: str,
    source_key: str,
    source_format: str,
    *,
    resolved_reply_chat_id: str | None = None,
) -> tuple[Any, ...]:
    message_id = message.get("id")
    if message_id is None:
        raise ValueError("message has no id")
    media_path = resolve_local_media_path(backup_path, message.get("file") or message.get("photo"))
    media_type = message.get("media_type")
    text = flatten_telegram_text(message.get("text", ""))
    if not text and media_type:
        text = f"[{media_type}]"
    timestamp = message.get("date")
    timestamp_unix = _unix_from_value(message.get("date_unixtime") or timestamp)
    edited = message.get("edited") or message.get("edit_date")
    edit_unix = _unix_from_value(message.get("edit_date_unixtime") or edited)
    media_status = "skipped" if message.get("media_skipped") else "error" if message.get("media_error") else "downloaded" if media_path else "missing" if media_type else None
    raw_tl, raw_tl_sha256, raw_tl_layer, raw_tl_library = _decode_tl_envelope(
        message.get("raw_message_tl")
    )
    reply = _reply_metadata(message)
    reply_chat_id = resolved_reply_chat_id or reply.get("reply_to_chat_id")
    if reply_chat_id is None and reply.get("reply_to_message_id") is not None and reply.get("reply_to_peer_kind") is None:
        reply_chat_id = chat_id
    known = {"id", "type", "date", "date_unixtime", "from", "actor", "from_id", "actor_id", "text", "media_type", "file", "photo", "media_size", "media_sha256", "media_skipped", "media_error", "reply_to_message_id", "reply_to_chat_id", "reply_to_peer_kind", "reply_to_peer_id", "reply_to_top_id", "reply_to_story_id", "reply_quote_text", "reply_quote_entities", "reply_quote_offset", "reply_media", "forwarded_from", "edited", "edit_date", "edit_date_unixtime", "entities", "text_entities", "reactions", "reply_markup", "action", "forward", "grouped_id", "raw_message", "raw_message_tl", "sender_entity", "sender_entity_status", "sender_entity_error", "metadata_schema_version", "expanded_metadata"}
    extras = {key: item for key, item in message.items() if key not in known}
    raw = json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return (
        int(message_id), chat_id, message.get("from") or message.get("actor") or "Unknown",
        str(message.get("from_id") or message.get("actor_id")) if message.get("from_id") is not None or message.get("actor_id") is not None else None,
        timestamp, timestamp_unix, text, media_type, media_path, reply.get("reply_to_message_id"),
        reply_chat_id, reply.get("reply_to_peer_kind"), reply.get("reply_to_peer_id"),
        reply.get("reply_to_top_id"), reply.get("reply_to_story_id"), reply.get("reply_quote_text"),
        _json_text(reply.get("reply_quote_entities")), reply.get("reply_quote_offset"),
        _json_text(reply.get("reply_media")), message.get("forwarded_from"), message.get("type"),
        edited, edit_unix, message.get("media_size"),
        message.get("media_sha256"), media_status,
        str(message.get("grouped_id")) if message.get("grouped_id") is not None else None,
        _json_text(message.get("entities") or message.get("text_entities")), _json_text(message.get("reactions")),
        _json_text(message.get("reply_markup")), _json_text(message.get("action")), _json_text(message.get("forward")),
        _json_text(extras) if extras else None, sqlite3.Binary(zlib.compress(raw, 9)),
        sqlite3.Binary(raw_tl) if raw_tl is not None else None, raw_tl_sha256,
        raw_tl_layer, raw_tl_library, _json_text(message.get("expanded_metadata")),
        source_key, source_format,
    )

def upsert_archival_message(
    conn: sqlite3.Connection,
    chat_id: str,
    message: dict[str, Any],
    backup_path: str,
    source_key: str,
    source_format: str,
) -> None:
    reply = _reply_metadata(message)
    reply_chat_id = _resolve_reply_chat_id(conn, chat_id, reply)
    conn.execute(
        ARCHIVAL_MESSAGE_UPSERT_SQL,
        archival_message_values(
            chat_id,
            message,
            backup_path,
            source_key,
            source_format,
            resolved_reply_chat_id=reply_chat_id,
        ),
    )
    sender_snapshot = _upsert_entity_snapshot(conn, message.get("sender_entity"))
    if sender_snapshot is not None:
        conn.execute(
            """INSERT INTO telegram_message_entity_refs(
                   chat_id, message_id, role, snapshot_sha256
               ) VALUES (?, ?, 'sender', ?)
               ON CONFLICT(chat_id, message_id, role) DO UPDATE SET
                   snapshot_sha256=excluded.snapshot_sha256""",
            (chat_id, int(message["id"]), sender_snapshot),
        )
    conn.execute(
        "INSERT OR IGNORE INTO message_sources(chat_id, message_id, source_key) VALUES (?, ?, ?)",
        (chat_id, int(message["id"]), source_key),
    )
