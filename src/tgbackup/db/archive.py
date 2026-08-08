"""Conversion and upsert primitives for rich archival message rows."""

from __future__ import annotations

import json
import os
import sqlite3
import urllib.parse
import zlib
from datetime import datetime
from typing import Any

ARCHIVAL_MESSAGE_UPSERT_SQL = """
INSERT INTO messages (
    message_id, chat_id, sender, sender_id, timestamp, timestamp_unix, text,
    media_type, media_path, reply_to_id, forwarded_from, message_type,
    edit_timestamp, edit_timestamp_unix, media_size, media_sha256, media_status,
    grouped_id, entities_json, reactions_json, reply_markup_json, action_json,
    forward_json, extra_json, raw_payload, source_key, source_format
) VALUES (
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
    ?, ?, ?, ?
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
        THEN COALESCE(excluded.media_type, messages.media_type) ELSE messages.media_type END,
    media_path=CASE WHEN excluded.source_format='telegram_api'
        OR (excluded.source_format='json' AND COALESCE(messages.source_format, '')!='telegram_api')
        OR (excluded.source_format='sqlite' AND COALESCE(messages.source_format, '') NOT IN ('telegram_api','json'))
        THEN COALESCE(excluded.media_path, messages.media_path) ELSE messages.media_path END,
    reply_to_id=CASE WHEN excluded.source_format='telegram_api'
        OR (excluded.source_format='json' AND COALESCE(messages.source_format, '')!='telegram_api')
        OR (excluded.source_format='sqlite' AND COALESCE(messages.source_format, '') NOT IN ('telegram_api','json'))
        THEN excluded.reply_to_id ELSE messages.reply_to_id END,
    forwarded_from=CASE WHEN excluded.source_format='telegram_api'
        OR (excluded.source_format='json' AND COALESCE(messages.source_format, '')!='telegram_api')
        OR (excluded.source_format='sqlite' AND COALESCE(messages.source_format, '') NOT IN ('telegram_api','json'))
        THEN excluded.forwarded_from ELSE messages.forwarded_from END,
    message_type=COALESCE(excluded.message_type, messages.message_type),
    edit_timestamp=COALESCE(excluded.edit_timestamp, messages.edit_timestamp),
    edit_timestamp_unix=COALESCE(excluded.edit_timestamp_unix, messages.edit_timestamp_unix),
    media_size=COALESCE(excluded.media_size, messages.media_size),
    media_sha256=COALESCE(excluded.media_sha256, messages.media_sha256),
    media_status=COALESCE(excluded.media_status, messages.media_status),
    grouped_id=COALESCE(excluded.grouped_id, messages.grouped_id),
    entities_json=COALESCE(excluded.entities_json, messages.entities_json),
    reactions_json=COALESCE(excluded.reactions_json, messages.reactions_json),
    reply_markup_json=COALESCE(excluded.reply_markup_json, messages.reply_markup_json),
    action_json=COALESCE(excluded.action_json, messages.action_json),
    forward_json=COALESCE(excluded.forward_json, messages.forward_json),
    extra_json=COALESCE(excluded.extra_json, messages.extra_json),
    raw_payload=COALESCE(excluded.raw_payload, messages.raw_payload),
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
    if os.path.isabs(value):
        return os.path.abspath(value)
    return os.path.abspath(os.path.join(backup_path, value))


def archival_message_values(
    chat_id: str,
    message: dict[str, Any],
    backup_path: str,
    source_key: str,
    source_format: str,
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
    known = {"id", "type", "date", "date_unixtime", "from", "actor", "from_id", "actor_id", "text", "media_type", "file", "photo", "media_size", "media_sha256", "media_skipped", "media_error", "reply_to_message_id", "forwarded_from", "edited", "edit_date", "edit_date_unixtime", "entities", "text_entities", "reactions", "reply_markup", "action", "forward", "grouped_id", "raw_message"}
    extras = {key: item for key, item in message.items() if key not in known}
    raw = json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return (
        int(message_id), chat_id, message.get("from") or message.get("actor") or "Unknown",
        str(message.get("from_id") or message.get("actor_id")) if message.get("from_id") is not None or message.get("actor_id") is not None else None,
        timestamp, timestamp_unix, text, media_type, media_path, message.get("reply_to_message_id"),
        message.get("forwarded_from"), message.get("type"), edited, edit_unix, message.get("media_size"),
        message.get("media_sha256"), media_status,
        str(message.get("grouped_id")) if message.get("grouped_id") is not None else None,
        _json_text(message.get("entities") or message.get("text_entities")), _json_text(message.get("reactions")),
        _json_text(message.get("reply_markup")), _json_text(message.get("action")), _json_text(message.get("forward")),
        _json_text(extras) if extras else None, sqlite3.Binary(zlib.compress(raw, 9)), source_key, source_format,
    )

def upsert_archival_message(
    conn: sqlite3.Connection,
    chat_id: str,
    message: dict[str, Any],
    backup_path: str,
    source_key: str,
    source_format: str,
) -> None:
    conn.execute(ARCHIVAL_MESSAGE_UPSERT_SQL, archival_message_values(chat_id, message, backup_path, source_key, source_format))
    conn.execute(
        "INSERT OR IGNORE INTO message_sources(chat_id, message_id, source_key) VALUES (?, ?, ?)",
        (chat_id, int(message["id"]), source_key),
    )
