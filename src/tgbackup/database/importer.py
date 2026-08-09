#!/usr/bin/env python3
"""
tgbackman-db-import

High-performance database indexer for Telegram backup files.
Ingests messages from both JSON and HTML backup files recursively,
loading them into a SQLite database designed for ultra-fast queries
using indexes and FTS5 full-text search.
"""

from __future__ import annotations

import html
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
import urllib.parse
import zlib
from datetime import datetime, timezone, timedelta
from html.parser import HTMLParser
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

# Re-use regexes and helpers similar to backman.py
HTML_CHAT_NAME_RE = re.compile(r'<div class="text bold">\s*(.*?)\s*</div>', re.DOTALL)

# HTML message class and timestamp regex
HTML_DIV_MESSAGE_MARKER = '<div class="message'
HTML_DAY_SEPARATOR_MARKER = '<div class="message service" id="message-'
HTML_MESSAGE_ID_BYTES_RE = re.compile(
    br'<div class="message[^"]*" id="message-?\d+"'
)
HTML_DAY_SEPARATOR_BYTES_RE = re.compile(
    br'<div class="message service" id="message-\d+"'
)
RANGE_DIR_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z__\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$"
)
HTML_VOID_ELEMENTS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}

# Extract exact Telegram date: "24.05.2026 22:15:30 UTC+01:00" or similar
HTML_MSG_TS_RE = re.compile(
    r"(\d{2})\.(\d{2})\.(\d{4}) (\d{2}):(\d{2}):(\d{2})(?:\s+UTC([+-]\d{2}):(\d{2}))?"
)

MESSAGE_UPSERT_SQL = """
INSERT INTO messages
    (message_id, chat_id, sender, sender_id, timestamp, timestamp_unix,
     text, media_type, media_path, reply_to_id, forwarded_from)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(chat_id, message_id) DO UPDATE SET
    sender = CASE WHEN messages.source_format IN ('telegram_api', 'json', 'sqlite')
        THEN messages.sender ELSE COALESCE(excluded.sender, messages.sender) END,
    sender_id = CASE WHEN messages.source_format IN ('telegram_api', 'json', 'sqlite')
        THEN messages.sender_id ELSE COALESCE(excluded.sender_id, messages.sender_id) END,
    timestamp = CASE WHEN messages.source_format IN ('telegram_api', 'json', 'sqlite')
        THEN messages.timestamp ELSE COALESCE(excluded.timestamp, messages.timestamp) END,
    timestamp_unix = CASE WHEN messages.source_format IN ('telegram_api', 'json', 'sqlite')
        THEN messages.timestamp_unix ELSE COALESCE(excluded.timestamp_unix, messages.timestamp_unix) END,
    text = CASE WHEN messages.source_format IN ('telegram_api', 'json', 'sqlite')
        THEN messages.text ELSE COALESCE(excluded.text, messages.text) END,
    media_type = CASE WHEN messages.source_format IN ('telegram_api', 'json', 'sqlite')
        THEN messages.media_type ELSE COALESCE(excluded.media_type, messages.media_type) END,
    media_path = CASE WHEN messages.source_format IN ('telegram_api', 'json', 'sqlite')
        THEN messages.media_path ELSE COALESCE(excluded.media_path, messages.media_path) END,
    reply_to_id = CASE WHEN messages.source_format IN ('telegram_api', 'json', 'sqlite')
        THEN messages.reply_to_id ELSE COALESCE(excluded.reply_to_id, messages.reply_to_id) END,
    forwarded_from = CASE WHEN messages.source_format IN ('telegram_api', 'json', 'sqlite')
        THEN messages.forwarded_from ELSE COALESCE(excluded.forwarded_from, messages.forwarded_from) END
"""

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
        THEN excluded.media_type ELSE messages.media_type END,
    media_path=CASE WHEN excluded.source_format='telegram_api'
        OR (excluded.source_format='json' AND COALESCE(messages.source_format, '')!='telegram_api')
        OR (excluded.source_format='sqlite' AND COALESCE(messages.source_format, '') NOT IN ('telegram_api','json'))
        THEN excluded.media_path ELSE messages.media_path END,
    reply_to_id=CASE WHEN excluded.source_format='telegram_api'
        OR (excluded.source_format='json' AND COALESCE(messages.source_format, '')!='telegram_api')
        OR (excluded.source_format='sqlite' AND COALESCE(messages.source_format, '') NOT IN ('telegram_api','json'))
        THEN excluded.reply_to_id ELSE messages.reply_to_id END,
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


def upsert_message_batch(cursor: sqlite3.Cursor, batch: list[tuple[Any, ...]]) -> int:
    if not batch:
        return 0
    cursor.executemany(MESSAGE_UPSERT_SQL, batch)
    return cursor.rowcount


def resolve_local_media_path(backup_path: str, media_path: Optional[str]) -> Optional[str]:
    """Store local media as an absolute path so range folders remain resolvable."""
    if not media_path:
        return media_path
    value = str(media_path)
    if urllib.parse.urlparse(value).scheme:
        return value
    root = os.path.abspath(backup_path)
    candidate = os.path.abspath(value) if os.path.isabs(value) else os.path.abspath(os.path.join(root, value))
    try:
        if os.path.commonpath((root, candidate)) != root or os.path.commonpath((os.path.realpath(root), os.path.realpath(candidate))) != os.path.realpath(root):
            raise ValueError(f"media path is outside its declared backup root: {value}")
    except ValueError as exc:
        raise ValueError(f"media path is outside its declared backup root: {value}") from exc
    return candidate


def _json_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _unix_from_value(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp())
        except (TypeError, ValueError):
            return None


def archival_message_values(
    chat_id: str,
    message: Dict[str, Any],
    backup_path: str,
    source_key: str,
    source_format: str,
) -> tuple[Any, ...]:
    """Convert one Telegram JSON/API record into the canonical rich row."""
    message_id = message.get("id")
    if message_id is None:
        raise ValueError("message has no id")
    media_path = resolve_local_media_path(
        backup_path, message.get("file") or message.get("photo")
    )
    text = flatten_telegram_text(message.get("text", ""))
    media_type = message.get("media_type")
    if not text and media_type:
        text = f"[{media_type}]"
    timestamp = message.get("date")
    timestamp_unix = _unix_from_value(message.get("date_unixtime") or timestamp)
    edited = message.get("edited") or message.get("edit_date")
    edit_unix = _unix_from_value(message.get("edit_date_unixtime") or edited)
    if message.get("media_skipped"):
        media_status = "skipped"
    elif message.get("media_error"):
        media_status = "error"
    elif media_path:
        media_status = "downloaded"
    elif media_type:
        media_status = "missing"
    else:
        media_status = None
    known = {
        "id", "type", "date", "date_unixtime", "from", "actor", "from_id",
        "actor_id", "text", "media_type", "file", "photo", "media_size",
        "media_sha256", "media_skipped", "media_error", "reply_to_message_id",
        "forwarded_from", "edited", "edit_date", "edit_date_unixtime",
        "entities", "text_entities", "reactions", "reply_markup", "action",
        "forward", "grouped_id", "raw_message",
    }
    extras = {key: value for key, value in message.items() if key not in known}
    raw = json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return (
        int(message_id), chat_id, message.get("from") or message.get("actor") or "Unknown",
        str(message.get("from_id") or message.get("actor_id"))
        if message.get("from_id") is not None or message.get("actor_id") is not None else None,
        timestamp, timestamp_unix, text, media_type, media_path,
        message.get("reply_to_message_id"), message.get("forwarded_from"),
        message.get("type"), edited, edit_unix, message.get("media_size"),
        message.get("media_sha256"), media_status,
        str(message.get("grouped_id")) if message.get("grouped_id") is not None else None,
        _json_text(message.get("entities") or message.get("text_entities")),
        _json_text(message.get("reactions")), _json_text(message.get("reply_markup")),
        _json_text(message.get("action")), _json_text(message.get("forward")),
        _json_text(extras) if extras else None, sqlite3.Binary(zlib.compress(raw, 9)),
        source_key, source_format,
    )


def upsert_archival_message(
    conn: sqlite3.Connection,
    chat_id: str,
    message: Dict[str, Any],
    backup_path: str,
    source_key: str,
    source_format: str,
) -> None:
    conn.execute(
        ARCHIVAL_MESSAGE_UPSERT_SQL,
        archival_message_values(chat_id, message, backup_path, source_key, source_format),
    )
    conn.execute(
        "INSERT OR IGNORE INTO message_sources(chat_id, message_id, source_key) VALUES (?, ?, ?)",
        (chat_id, int(message["id"]), source_key),
    )

def parse_html_timestamp(ts_str: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse a Telegram HTML backup title timestamp and return:
    (ISO-8601 string in UTC, Unix epoch timestamp).
    """
    if not ts_str:
        return None, None
    try:
        m = HTML_MSG_TS_RE.match(ts_str.strip())
        if m:
            dd, mm, yyyy, hh, mi, ss, tz_h_s, tz_m_s = m.groups()
            tz_h = int(tz_h_s) if tz_h_s else 0
            tz_m = int(tz_m_s) if tz_m_s else 0
            # Calculate timezone offset
            offset = timezone(timedelta(hours=tz_h, minutes=(tz_m if tz_h >= 0 else -tz_m)))
            dt = datetime(int(yyyy), int(mm), int(dd), int(hh), int(mi), int(ss), tzinfo=offset)
            dt_utc = dt.astimezone(timezone.utc)
            return dt_utc.isoformat().replace("+00:00", "Z"), int(dt_utc.timestamp())
    except Exception:
        pass
    return None, None

def detect_media(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Check if a local link or src URL matches any known media subdirectory.
    Returns (media_type, media_path) if found, otherwise (None, None).
    """
    if not url:
        return None, None
    url_l = url.lower()
    if url_l.startswith(("http://", "https://", "tg:", "mailto:", "#", "javascript:")):
        return None, None
    try:
        u = urllib.parse.unquote(url).replace("\\", "/").strip()
    except Exception:
        u = url.replace("\\", "/").strip()

    parts = u.split("/")
    for part in parts:
        part_l = part.lower()
        if part_l == "photos":
            return "photo", u
        elif part_l == "video_files":
            return "video", u
        elif part_l == "voice_messages":
            return "voice_message", u
        elif part_l == "audio_files":
            return "audio_file", u
        elif part_l in ("sticker_files", "stickers"):
            return "sticker", u
        elif part_l in ("files", "documents"):
            return "file", u
        elif part_l == "animations":
            return "animation", u
    return None, None

def flatten_telegram_text(text_field: Any) -> str:
    """
    Convert rich Telegram JSON text formatting (strings and format dicts)
    into a flat plain text string.
    """
    if isinstance(text_field, str):
        return text_field
    if isinstance(text_field, list):
        parts = []
        for item in text_field:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                val = item.get("text")
                if val:
                    parts.append(str(val))
        return "".join(parts)
    return ""

def _extract_chat_name_from_messages_html(messages_html_path: str) -> Optional[str]:
    """
    Extract the chat title from messages.html header.
    """
    try:
        with open(messages_html_path, "r", encoding="utf-8", errors="ignore") as f:
            chunk = f.read(64 * 1024)
    except Exception:
        return None

    m = HTML_CHAT_NAME_RE.search(chunk)
    if not m:
        return None
    raw = m.group(1)
    raw = re.sub(r"<[^>]+>", "", raw)
    raw = html.unescape(raw)
    return raw.strip() or None

class TelegramHTMLParser(HTMLParser):
    """
    Optimized, high-performance HTML message parser for Telegram backup HTML files.
    """
    def __init__(self, on_message: Optional[Callable[[Dict[str, Any]], None]] = None):
        super().__init__()
        self.in_msg = False
        self.msg_depth = 0
        self.depth = 0
        self.cur = None
        self.field = None
        self.messages = []
        self.on_message = on_message
        self.last_sender = ""

    def handle_starttag(self, tag, attrs):
        if tag not in HTML_VOID_ELEMENTS:
            self.depth += 1
        attrs_dict = dict(attrs)
        cls = attrs_dict.get("class", "") or ""

        # Detect start of message block
        is_msg_div = (tag == "div" and
                      ((" message " in f" {cls} ") or cls.startswith("message")) and
                      attrs_dict.get("id", "").startswith("message"))

        if is_msg_div:
            self.flush_current()
            self.in_msg = True
            self.msg_depth = self.depth

            # Extract message_id
            m_id_str = attrs_dict.get("id", "")
            message_id = None
            suffix = m_id_str[7:]
            if suffix.isdigit():
                message_id = int(suffix)
            elif suffix.startswith("-") and suffix[1:].isdigit():
                message_id = int(suffix)

            self.cur = {
                "message_id": message_id,
                "sender": [],
                "sender_id": None,
                "timestamp": None,
                "timestamp_unix": None,
                "text": [],
                "media_type": None,
                "media_path": None,
                "reply_to_id": None,
                "forwarded_from": None,
                "is_day_separator": "service" in cls.split() and suffix.startswith("-"),
            }
            self.field = None
            return

        if not self.in_msg or not self.cur:
            return

        # Check for reply links
        if tag == "a" and attrs_dict.get("href", "").startswith("#message"):
            href = attrs_dict["href"]
            suffix = href[8:]
            if suffix.isdigit():
                self.cur["reply_to_id"] = int(suffix)
            elif suffix.startswith("-") and suffix[1:].isdigit():
                self.cur["reply_to_id"] = int(suffix)

        # Check for relative media links
        for attr_name in ("href", "src"):
            if attr_name in attrs_dict:
                m_type, m_path = detect_media(attrs_dict[attr_name])
                if m_type:
                    self.cur["media_type"] = m_type
                    self.cur["media_path"] = m_path

        # Tag-based structural states
        if tag == "div":
            if "from_name" in cls:
                self.field = "sender"
            elif cls.strip() == "text" or "text " in f"{cls} ":
                self.field = "text"
            elif "date" in cls and attrs_dict.get("title"):
                ts_raw = attrs_dict.get("title")
                iso_ts, unix_ts = parse_html_timestamp(ts_raw)
                if iso_ts:
                    self.cur["timestamp"] = iso_ts
                    self.cur["timestamp_unix"] = unix_ts

        elif tag == "br" and self.field == "text":
            self.cur["text"].append("\n")

        elif tag == "a" and self.field == "text":
            href = attrs_dict.get("href")
            if href and not href.startswith("#"):
                self.cur["text"].append(f" {href} ")

    def handle_endtag(self, tag):
        if tag in HTML_VOID_ELEMENTS:
            return
        if self.in_msg and self.depth == self.msg_depth:
            self.flush_current()
            self.in_msg = False
            self.field = None
        self.depth = max(0, self.depth - 1)
        if self.field and tag == "div":
            self.field = None

    def handle_data(self, data):
        if not (self.in_msg and self.field and self.cur):
            return
        if self.field == "sender":
            self.cur["sender"].append(data)
        elif self.field == "text":
            self.cur["text"].append(data)

    def flush_current(self):
        if (
            not self.cur
            or self.cur.get("message_id") is None
            or self.cur.get("is_day_separator")
        ):
            self.cur = None
            return

        # Sender resolution
        sender_raw = "".join(self.cur["sender"]).strip()
        sender = sender_raw or self.last_sender
        if sender_raw:
            self.last_sender = sender_raw

        text = "".join(self.cur["text"]).strip()

        # Fallback text representation if empty and media exists
        if not text and self.cur["media_type"]:
            text = f"[{self.cur['media_type']}]"

        record = {
            "message_id": self.cur["message_id"],
            "sender": sender or "Unknown",
            "sender_id": self.cur["sender_id"],
            "timestamp": self.cur["timestamp"],
            "timestamp_unix": self.cur["timestamp_unix"],
            "text": text,
            "media_type": self.cur["media_type"],
            "media_path": self.cur["media_path"],
            "reply_to_id": self.cur["reply_to_id"],
            "forwarded_from": self.cur["forwarded_from"]
        }
        if self.on_message is not None:
            self.on_message(record)
        else:
            self.messages.append(record)
        self.cur = None


def infer_html_identity(path: str, root_path: Optional[str] = None) -> Tuple[str, str, str]:
    """Return a stable chat ID/name/root independent of date-range subfolders."""
    parent = os.path.dirname(os.path.abspath(path))
    parent_name = os.path.basename(parent)
    if RANGE_DIR_RE.fullmatch(parent_name):
        chat_root = os.path.dirname(parent)
    elif os.path.basename(os.path.dirname(parent)).lower() == "chats":
        chat_root = parent
    else:
        chat_root = parent
    chat_name = _extract_chat_name_from_messages_html(path) or os.path.basename(chat_root) or "Single Chat"
    if root_path:
        try:
            relative = os.path.relpath(chat_root, os.path.abspath(root_path))
            if relative not in (".", "..") and not relative.startswith(f"..{os.sep}"):
                chat_id = relative.replace(os.sep, "/")
            else:
                chat_id = os.path.basename(chat_root)
        except ValueError:
            chat_id = os.path.basename(chat_root)
    else:
        chat_id = os.path.basename(chat_root)
    return chat_id or "single_chat", chat_name, chat_root


def expected_html_messages(path: str) -> int:
    count = 0
    carry = b""
    overlap = 256
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            data = carry + chunk
            if len(data) <= overlap:
                carry = data
                continue
            scan, carry = data[:-overlap], data[-overlap:]
            count += len(HTML_MESSAGE_ID_BYTES_RE.findall(scan))
            count -= len(HTML_DAY_SEPARATOR_BYTES_RE.findall(scan))
    count += len(HTML_MESSAGE_ID_BYTES_RE.findall(carry))
    count -= len(HTML_DAY_SEPARATOR_BYTES_RE.findall(carry))
    return count


def create_post_load_indexes(conn: sqlite3.Connection):
    """
    Create database indexes, FTS5 virtual search tables, and automated synchronization triggers.
    Running this after bulk loading is up to 10x faster.
    """
    cursor = conn.cursor()

    # Standard B-tree indexes for fast filtering and range querying
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_ts ON messages(chat_id, timestamp_unix);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages(sender);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(timestamp_unix);")

    # Set up FTS5 Virtual Search Table.  Keep a compatible table in place so
    # message upserts can be synchronized by the trigger below; older DBs may
    # have been created without the media_path column.
    fts_columns = {str(row[1]) for row in cursor.execute("PRAGMA table_info(messages_fts)").fetchall()}
    if fts_columns and "media_path" not in fts_columns:
        cursor.execute("DROP TABLE IF EXISTS messages_fts;")
    cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(text, media_path, content='messages', content_rowid='id');")

    # Synchronization triggers for FTS5 (keeps it updated automatically for any future modifications)
    cursor.execute("DROP TRIGGER IF EXISTS messages_ai;")
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, text, media_path) VALUES (new.id, new.text, new.media_path);
        END;
    """)
    cursor.execute("DROP TRIGGER IF EXISTS messages_ad;")
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, text, media_path) VALUES('delete', old.id, old.text, old.media_path);
        END;
    """)
    cursor.execute("DROP TRIGGER IF EXISTS messages_au;")
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, text, media_path) VALUES('delete', old.id, old.text, old.media_path);
            INSERT INTO messages_fts(rowid, text, media_path) VALUES (new.id, new.text, new.media_path);
        END;
    """)

    cursor.execute("INSERT INTO messages_fts(messages_fts) VALUES ('rebuild');")
    # Run optimizer
    cursor.execute("ANALYZE;")
    conn.commit()


def _parse_json_file_legacy(path: str, conn: sqlite3.Connection, batch_size: int = 50000) -> int:
    """
    Ingest a Telegram JSON export using streaming ijson (if available) or standard json load.
    Handles both multi-chat (export_results.json) and single-chat (result.json) layouts.
    """
    # Sniff if the JSON format is multi-chat (contains "chats") vs single-chat (contains "messages")
    is_multi = False
    try:
        with open(path, "rb") as f:
            chunk = f.read(10240).decode("utf-8", errors="ignore")
            if '"chats"' in chunk:
                is_multi = True
    except Exception:
        pass

    cursor = conn.cursor()
    messages_batch = []
    total_inserted = 0

    # Try streaming with ijson
    try:
        import ijson

        if is_multi:
            # Multi-chat JSON export
            with open(path, "rb") as f:
                chats = ijson.items(f, 'chats.list.item')
                for chat in chats:
                    chat_id = str(chat.get("id"))
                    chat_name = chat.get("name") or chat.get("title") or f"chat_{chat_id}"
                    chat_type = chat.get("type")

                    backup_path = os.path.abspath(os.path.dirname(path))
                    cursor.execute(
                        "INSERT OR IGNORE INTO chats (chat_id, chat_name, chat_type, backup_path) VALUES (?, ?, ?, ?)",
                        (chat_id, chat_name, chat_type, backup_path)
                    )
                    cursor.execute(
                        "UPDATE chats SET backup_path = ? WHERE chat_id = ? AND backup_path IS NULL",
                        (backup_path, chat_id)
                    )

                    msgs = chat.get("messages", [])
                    for msg in msgs:
                        if not isinstance(msg, dict):
                            continue
                        msg_id = msg.get("id")
                        if msg_id is None:
                            continue

                        text = flatten_telegram_text(msg.get("text", ""))
                        if not text:
                            mt = msg.get("media_type") or msg.get("type")
                            if mt and mt != "message":
                                text = f"[{mt}]"

                        sender = msg.get("from") or msg.get("actor") or "Unknown"
                        sender_id = str(msg.get("from_id")) if msg.get("from_id") is not None else None

                        ts_iso = msg.get("date")
                        ts_unix = msg.get("date_unixtime")
                        if ts_unix is not None:
                            ts_unix = int(ts_unix)
                        elif ts_iso:
                            try:
                                d = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
                                ts_unix = int(d.timestamp())
                            except Exception:
                                pass

                        media_type = msg.get("media_type")
                        media_path = resolve_local_media_path(backup_path, msg.get("file") or msg.get("photo"))
                        reply_to_id = msg.get("reply_to_message_id")
                        forwarded_from = msg.get("forwarded_from")

                        messages_batch.append((
                            msg_id, chat_id, sender, sender_id, ts_iso, ts_unix,
                            text, media_type, media_path, reply_to_id, forwarded_from
                        ))

                        if len(messages_batch) >= batch_size:
                            total_inserted += upsert_message_batch(cursor, messages_batch)
                            messages_batch.clear()
                            conn.commit()

            if messages_batch:
                total_inserted += upsert_message_batch(cursor, messages_batch)
                messages_batch.clear()
                conn.commit()

        else:
            # Single-chat JSON export
            chat_id = "single_chat"
            chat_name = os.path.basename(os.path.dirname(path)) or "Single Chat"
            backup_path = os.path.abspath(os.path.dirname(path))
            cursor.execute(
                "INSERT OR IGNORE INTO chats (chat_id, chat_name, chat_type, backup_path) VALUES (?, ?, ?, ?)",
                (chat_id, chat_name, "personal_chat", backup_path)
            )
            cursor.execute(
                "UPDATE chats SET backup_path = ? WHERE chat_id = ? AND backup_path IS NULL",
                (backup_path, chat_id)
            )

            with open(path, "rb") as f:
                msgs = ijson.items(f, 'messages.item')
                for msg in msgs:
                    if not isinstance(msg, dict):
                        continue
                    msg_id = msg.get("id")
                    if msg_id is None:
                        continue

                    text = flatten_telegram_text(msg.get("text", ""))
                    if not text:
                        mt = msg.get("media_type") or msg.get("type")
                        if mt and mt != "message":
                            text = f"[{mt}]"

                    sender = msg.get("from") or msg.get("actor") or "Unknown"
                    sender_id = str(msg.get("from_id")) if msg.get("from_id") is not None else None

                    ts_iso = msg.get("date")
                    ts_unix = msg.get("date_unixtime")
                    if ts_unix is not None:
                        ts_unix = int(ts_unix)
                    elif ts_iso:
                        try:
                            d = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
                            ts_unix = int(d.timestamp())
                        except Exception:
                            pass

                    media_type = msg.get("media_type")
                    media_path = resolve_local_media_path(backup_path, msg.get("file") or msg.get("photo"))
                    reply_to_id = msg.get("reply_to_message_id")
                    forwarded_from = msg.get("forwarded_from")

                    messages_batch.append((
                        msg_id, chat_id, sender, sender_id, ts_iso, ts_unix,
                        text, media_type, media_path, reply_to_id, forwarded_from
                    ))

                    if len(messages_batch) >= batch_size:
                        total_inserted += upsert_message_batch(cursor, messages_batch)
                        messages_batch.clear()
                        conn.commit()

            if messages_batch:
                total_inserted += upsert_message_batch(cursor, messages_batch)
                messages_batch.clear()
                conn.commit()

        return total_inserted

    except ImportError:
        # Fallback to standard json load if ijson is unavailable
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if is_multi:
            chats = data.get("chats", {}).get("list", [])
            for chat in chats:
                chat_id = str(chat.get("id"))
                chat_name = chat.get("name") or chat.get("title") or f"chat_{chat_id}"
                chat_type = chat.get("type")

                backup_path = os.path.abspath(os.path.dirname(path))
                cursor.execute(
                    "INSERT OR IGNORE INTO chats (chat_id, chat_name, chat_type, backup_path) VALUES (?, ?, ?, ?)",
                    (chat_id, chat_name, chat_type, backup_path)
                )
                cursor.execute(
                    "UPDATE chats SET backup_path = ? WHERE chat_id = ? AND backup_path IS NULL",
                    (backup_path, chat_id)
                )

                msgs = chat.get("messages", [])
                for msg in msgs:
                    if not isinstance(msg, dict):
                        continue
                    msg_id = msg.get("id")
                    if msg_id is None:
                        continue
                    text = flatten_telegram_text(msg.get("text", ""))
                    if not text:
                        mt = msg.get("media_type") or msg.get("type")
                        if mt and mt != "message":
                            text = f"[{mt}]"

                    sender = msg.get("from") or msg.get("actor") or "Unknown"
                    sender_id = str(msg.get("from_id")) if msg.get("from_id") is not None else None

                    ts_iso = msg.get("date")
                    ts_unix = msg.get("date_unixtime")
                    if ts_unix is not None:
                        ts_unix = int(ts_unix)

                    media_type = msg.get("media_type")
                    media_path = resolve_local_media_path(backup_path, msg.get("file") or msg.get("photo"))
                    reply_to_id = msg.get("reply_to_message_id")
                    forwarded_from = msg.get("forwarded_from")

                    messages_batch.append((
                        msg_id, chat_id, sender, sender_id, ts_iso, ts_unix,
                        text, media_type, media_path, reply_to_id, forwarded_from
                    ))

                    if len(messages_batch) >= batch_size:
                        total_inserted += upsert_message_batch(cursor, messages_batch)
                        messages_batch.clear()
                        conn.commit()
        else:
            chat_id = "single_chat"
            chat_name = os.path.basename(os.path.dirname(path)) or "Single Chat"
            backup_path = os.path.abspath(os.path.dirname(path))
            cursor.execute(
                "INSERT OR IGNORE INTO chats (chat_id, chat_name, chat_type, backup_path) VALUES (?, ?, ?, ?)",
                (chat_id, chat_name, "personal_chat", backup_path)
            )
            cursor.execute(
                "UPDATE chats SET backup_path = ? WHERE chat_id = ? AND backup_path IS NULL",
                (backup_path, chat_id)
            )

            msgs = data.get("messages", [])
            for msg in msgs:
                if not isinstance(msg, dict):
                    continue
                msg_id = msg.get("id")
                if msg_id is None:
                    continue
                text = flatten_telegram_text(msg.get("text", ""))
                if not text:
                    mt = msg.get("media_type") or msg.get("type")
                    if mt and mt != "message":
                        text = f"[{mt}]"

                sender = msg.get("from") or msg.get("actor") or "Unknown"
                sender_id = str(msg.get("from_id")) if msg.get("from_id") is not None else None

                ts_iso = msg.get("date")
                ts_unix = msg.get("date_unixtime")
                if ts_unix is not None:
                    ts_unix = int(ts_unix)

                media_type = msg.get("media_type")
                media_path = resolve_local_media_path(backup_path, msg.get("file") or msg.get("photo"))
                reply_to_id = msg.get("reply_to_message_id")
                forwarded_from = msg.get("forwarded_from")

                messages_batch.append((
                    msg_id, chat_id, sender, sender_id, ts_iso, ts_unix,
                    text, media_type, media_path, reply_to_id, forwarded_from
                ))

                if len(messages_batch) >= batch_size:
                    total_inserted += upsert_message_batch(cursor, messages_batch)
                    messages_batch.clear()
                    conn.commit()

        if messages_batch:
            total_inserted += upsert_message_batch(cursor, messages_batch)
            messages_batch.clear()
            conn.commit()

        return total_inserted


def infer_json_identity(path: str, root_path: Optional[str] = None) -> Tuple[str, str, str]:
    parent = os.path.dirname(os.path.abspath(path))
    if RANGE_DIR_RE.fullmatch(os.path.basename(parent)):
        chat_root = os.path.dirname(parent)
    else:
        chat_root = parent
    marker = os.path.join(chat_root, ".tgbackman_target.json")
    if os.path.isfile(marker):
        try:
            with open(marker, "r", encoding="utf-8") as handle:
                target = json.load(handle)
            chat_id = str(target.get("chat_id") or target.get("target_key") or "").strip()
            chat_name = str(target.get("title") or target.get("source_name") or "").strip()
            if chat_id:
                return chat_id, chat_name or os.path.basename(chat_root), chat_root
        except (OSError, ValueError, TypeError):
            pass
    if root_path:
        relative = os.path.relpath(chat_root, os.path.abspath(root_path))
        chat_id = relative.replace(os.sep, "/") if not relative.startswith("..") else os.path.basename(chat_root)
    else:
        chat_id = os.path.basename(chat_root)
    return chat_id or "single_chat", os.path.basename(chat_root) or "Single Chat", chat_root


def _iter_single_chat_json_messages(path: str) -> Iterable[Dict[str, Any]]:
    try:
        import ijson
    except ImportError:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        yield from (item for item in data.get("messages", []) if isinstance(item, dict))
        return
    with open(path, "rb") as handle:
        yield from (item for item in ijson.items(handle, "messages.item") if isinstance(item, dict))


def _iter_multi_chat_json(path: str) -> Iterable[tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
    """Stream one chat header and one message object at a time.

    ``ijson.items(..., 'chats.list.item')`` still materializes the complete
    ``messages`` array for a chat.  Telegram exports can contain very large
    chats, so this event parser keeps only the current message object alive.
    A ``None`` message marks a chat header (including chats with no messages).
    """
    try:
        import ijson
        from ijson.common import ObjectBuilder
    except ImportError:  # pragma: no cover - ijson is a runtime dependency
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for chat in payload.get("chats", {}).get("list", []):
            if not isinstance(chat, dict):
                continue
            header = {
                "id": chat.get("id"),
                "name": chat.get("name") or chat.get("title"),
                "title": chat.get("title"),
                "type": chat.get("type"),
            }
            yield header, None
            for message in chat.get("messages", []):
                if isinstance(message, dict):
                    yield header, message
        return

    base = "chats.list.item"
    current_chat: Dict[str, Any] = {}
    builder: Any = None
    with open(path, "rb") as handle:
        for prefix, event, value in ijson.parse(handle):
            if builder is not None:
                builder.event(event, value)
                if event == "end_map" and not builder.containers:
                    message = builder.value
                    if isinstance(message, dict):
                        yield dict(current_chat), message
                    builder = None
                continue

            if prefix == base and event == "start_map":
                current_chat = {}
            elif prefix == f"{base}.messages" and event == "start_array":
                yield dict(current_chat), None
            elif prefix == f"{base}.messages.item" and event == "start_map":
                builder = ObjectBuilder()
                builder.event(event, value)
            elif prefix in {
                f"{base}.id",
                f"{base}.name",
                f"{base}.title",
                f"{base}.type",
            } and event not in {"start_map", "end_map", "start_array", "end_array", "map_key"}:
                current_chat[prefix.rsplit(".", 1)[-1]] = value


def parse_json_file_archival(
    path: str,
    conn: sqlite3.Connection,
    batch_size: int = 50000,
    *,
    root_path: Optional[str] = None,
    source_key: Optional[str] = None,
) -> Tuple[int, Set[str]]:
    """Import rich JSON fields and provenance; stream single-chat message arrays."""
    del batch_size  # The caller owns the source-sized transaction.
    try:
        import ijson
    except ImportError:
        with open(path, "r", encoding="utf-8") as handle:
            probe = json.load(handle)
        is_multi = isinstance(probe, dict) and isinstance(probe.get("chats"), dict) and isinstance(
            probe.get("chats", {}).get("list"), list
        )
    else:
        is_multi = False
        with open(path, "rb") as handle:
            for prefix, event, _value in ijson.parse(handle):
                if prefix == "chats.list" and event == "start_array":
                    is_multi = True
                    break
                if prefix == "messages" and event == "start_array":
                    break
    imported = 0
    chat_ids: Set[str] = set()
    source_key = source_key or hashlib.sha256(os.path.abspath(path).encode()).hexdigest()
    if is_multi:
        backup_path = os.path.abspath(os.path.dirname(path))
        for chat, message in _iter_multi_chat_json(path):
            chat_id = str(chat.get("id"))
            chat_name = chat.get("name") or chat.get("title") or f"chat_{chat_id}"
            conn.execute(
                """INSERT INTO chats(chat_id, chat_name, chat_type, backup_path)
                   VALUES (?, ?, ?, ?) ON CONFLICT(chat_id) DO UPDATE SET
                   chat_name=COALESCE(excluded.chat_name, chats.chat_name),
                   chat_type=COALESCE(excluded.chat_type, chats.chat_type)""",
                (chat_id, chat_name, chat.get("type"), backup_path),
            )
            chat_ids.add(chat_id)
            if isinstance(message, dict) and message.get("id") is not None:
                upsert_archival_message(conn, chat_id, message, backup_path, source_key, "json")
                imported += 1
    else:
        chat_id, chat_name, chat_root = infer_json_identity(path, root_path)
        backup_path = os.path.dirname(os.path.abspath(path))
        conn.execute(
            """INSERT INTO chats(chat_id, chat_name, chat_type, backup_path)
               VALUES (?, ?, 'personal_chat', ?) ON CONFLICT(chat_id) DO UPDATE SET
               chat_name=COALESCE(excluded.chat_name, chats.chat_name)""",
            (chat_id, chat_name, chat_root),
        )
        chat_ids.add(chat_id)
        for message in _iter_single_chat_json_messages(path):
            if message.get("id") is not None:
                upsert_archival_message(conn, chat_id, message, backup_path, source_key, "json")
                imported += 1
    return imported, chat_ids


def parse_json_file(path: str, conn: sqlite3.Connection, batch_size: int = 50000) -> int:
    """Public one-file importer using the canonical rich/provenance path."""
    source_key = archive_source_file(conn, path, "json", archive_payload=False)
    try:
        imported, chat_ids = parse_json_file_archival(
            path, conn, batch_size, source_key=source_key
        )
        record_import(
            conn, source_key, "json", path,
            next(iter(chat_ids)) if len(chat_ids) == 1 else None,
            imported, imported,
        )
        conn.execute(
            "UPDATE backup_sources SET message_count=? WHERE source_key=?",
            (imported, source_key),
        )
        conn.commit()
        return imported
    except Exception:
        conn.rollback()
        raise


def parse_html_file(
    path: str,
    conn: sqlite3.Connection,
    batch_size: int = 50000,
    *,
    root_path: Optional[str] = None,
    source_key: Optional[str] = None,
) -> int:
    """
    Ingest a Telegram HTML message file (e.g. messages.html, messages2.html).
    Infers the chat_id from parent directory and extracts the chat_name from header.
    """
    chat_id, chat_name, chat_root = infer_html_identity(path, root_path)
    parent_dir = os.path.dirname(path)

    cursor = conn.cursor()

    # Ensure chat is registered in database
    backup_path = os.path.abspath(chat_root)
    cursor.execute(
        "INSERT OR IGNORE INTO chats (chat_id, chat_name, chat_type, backup_path) VALUES (?, ?, ?, ?)",
        (chat_id, chat_name, "personal_chat" if chat_id == "single_chat" else "unknown", backup_path)
    )
    cursor.execute(
        "UPDATE chats SET backup_path = ? WHERE chat_id = ? AND backup_path IS NULL",
        (backup_path, chat_id)
    )

    messages_batch: list[tuple[Any, ...]] = []
    message_ids: list[int] = []
    total_inserted = 0

    def flush_batch() -> None:
        nonlocal total_inserted
        if not messages_batch:
            return
        total_inserted += upsert_message_batch(cursor, messages_batch)
        if source_key:
            cursor.executemany(
                "INSERT OR IGNORE INTO message_sources (chat_id, message_id, source_key) VALUES (?, ?, ?)",
                [(chat_id, message_id, source_key) for message_id in message_ids],
            )
            cursor.execute(
                "UPDATE messages SET source_key=COALESCE(source_key, ?), source_format=COALESCE(source_format, 'html') "
                "WHERE chat_id=? AND message_id IN (SELECT message_id FROM message_sources WHERE chat_id=? AND source_key=?)",
                (source_key, chat_id, chat_id, source_key),
            )
        messages_batch.clear()
        message_ids.clear()

    def consume_message(msg: Dict[str, Any]) -> None:
        message_id = int(msg["message_id"])
        messages_batch.append((
            message_id, chat_id, msg["sender"], msg["sender_id"],
            msg["timestamp"], msg["timestamp_unix"], msg["text"],
            msg["media_type"], resolve_local_media_path(parent_dir, msg["media_path"]),
            msg["reply_to_id"], msg["forwarded_from"],
        ))
        message_ids.append(message_id)
        if len(messages_batch) >= batch_size:
            flush_batch()

    # Parse HTML messages incrementally.  The parser emits each record as its
    # closing div arrives, so a multi-million-message export never needs to
    # retain the complete history or source-link list in RAM.
    parser = TelegramHTMLParser(on_message=consume_message)
    with open(path, "r", encoding="utf-8", errors="strict") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), ""):
            parser.feed(chunk)
    parser.close()
    parser.flush_current()
    flush_batch()
    return total_inserted


def parse_sqlite_backup_file(
    path: str,
    conn: sqlite3.Connection,
    batch_size: int = 50000,
    *,
    source_key: Optional[str] = None,
) -> int:
    """
    Ingest messages and chats from an old Telegram jar backup (database.sqlite).
    """
    try:
        src_conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except Exception:
        src_conn = sqlite3.connect(path)

    src_cursor = src_conn.cursor()
    dest_cursor = conn.cursor()

    def _table_cols(con, table: str) -> Set[str]:
        try:
            cur = con.cursor()
            cur.execute(f"pragma table_info({table})")
            return {str(r[1]) for r in cur.fetchall()}
        except Exception:
            return set()

    try:
        src_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in src_cursor.fetchall()}
        if not {"messages", "chats", "users"}.issubset(tables):
            src_conn.close()
            raise RuntimeError("unsupported SQLite backup schema: messages/chats/users required")
    except Exception:
        src_conn.close()
        raise

    user_names: Dict[str, str] = {}
    chat_names: Dict[str, str] = {}

    u_cols = _table_cols(src_conn, "users")
    if "first_name" in u_cols or "last_name" in u_cols:
        first_col = "first_name" if "first_name" in u_cols else "NULL"
        last_col = "last_name" if "last_name" in u_cols else "NULL"
        try:
            src_cursor.execute(f"SELECT id, {first_col}, {last_col} FROM users")
            for uid, fn, ln in src_cursor.fetchall():
                parts = [str(x).strip() for x in (fn, ln) if x is not None and str(x).strip()]
                user_names[str(uid)] = " ".join(parts).strip() or f"User {uid}"
        except Exception:
            pass
    elif "name" in u_cols:
        try:
            src_cursor.execute("SELECT id, name FROM users")
            for uid, name in src_cursor.fetchall():
                user_names[str(uid)] = str(name).strip() if name else f"User {uid}"
        except Exception:
            pass

    c_cols = _table_cols(src_conn, "chats")
    if "name" in c_cols:
        try:
            src_cursor.execute("SELECT id, name FROM chats")
            for cid, name in src_cursor.fetchall():
                chat_names[str(cid)] = str(name).strip() if name else f"Chat {cid}"
        except Exception:
            pass
    elif "title" in c_cols:
        try:
            src_cursor.execute("SELECT id, title FROM chats")
            for cid, title in src_cursor.fetchall():
                chat_names[str(cid)] = str(title).strip() if title else f"Chat {cid}"
        except Exception:
            pass

    try:
        src_cursor.execute(
            """SELECT DISTINCT source_type, source_id FROM messages
               WHERE source_type IS NOT NULL AND source_id IS NOT NULL"""
        )
        for stype, sid in src_cursor.fetchall():
            chat_id = f"{stype}_{sid}"
            title = None
            if stype == "dialog":
                title = user_names.get(str(sid))
            elif stype == "group":
                title = chat_names.get(str(sid))
            chat_name = title or f"{stype}_{sid}"

            backup_path = os.path.abspath(os.path.dirname(path))
            dest_cursor.execute(
                "INSERT OR IGNORE INTO chats (chat_id, chat_name, chat_type, backup_path) VALUES (?, ?, ?, ?)",
                (chat_id, chat_name, "personal_chat" if stype == "dialog" else "group", backup_path)
            )
            dest_cursor.execute(
                "UPDATE chats SET backup_path = ? WHERE chat_id = ? AND backup_path IS NULL",
                (backup_path, chat_id)
            )
    except Exception:
        pass

    m_cols = _table_cols(src_conn, "messages")
    needed = {"message_id", "source_id", "source_type", "sender_id", "text", "time"}
    if not needed.issubset(m_cols):
        src_conn.close()
        raise RuntimeError(f"unsupported SQLite messages schema; missing {sorted(needed - m_cols)}")

    optional = [
        "media_type", "media_file", "message_type", "fwd_from_id", "media_size",
        "media_json", "markup_json", "data", "api_layer",
    ]
    selected_optional = [name if name in m_cols else f"NULL AS {name}" for name in optional]
    src_cursor.execute(
        "SELECT message_id, source_id, source_type, sender_id, text, time, "
        + ", ".join(selected_optional)
        + " FROM messages WHERE source_id IS NOT NULL AND source_type IS NOT NULL"
    )

    messages_batch = []
    total_inserted = 0
    for row in src_cursor:
        mid, sid, stype, sender_id, text, t_val = row[0], row[1], row[2], row[3], row[4], row[5]
        (
            mtype, mfile, message_type, fwd_from_id, media_size, media_json,
            markup_json, raw_data, api_layer,
        ) = row[6:15]

        if mid is None or sid is None:
            continue
        chat_id = f"{stype}_{sid}"
        sender_id_s = str(sender_id) if sender_id is not None else None

        sender = user_names.get(sender_id_s) if sender_id_s else None
        if not sender:
            sender = "User " + sender_id_s if sender_id_s else "Unknown"

        ts_unix = int(t_val) if t_val is not None else None
        ts_iso = None
        if ts_unix is not None:
            try:
                dt = datetime.fromtimestamp(ts_unix, tz=timezone.utc)
                ts_iso = dt.isoformat().replace("+00:00", "Z")
            except Exception:
                pass

        mapped_mtype = None
        if mtype:
            mt_l = mtype.lower()
            if mt_l == "photo":
                mapped_mtype = "photo"
            elif mt_l == "document" and mfile:
                mfile_l = mfile.lower()
                if mfile_l.startswith("t_voice") or any(mfile_l.endswith(ext) for ext in (".ogg", ".wav", ".m4a", ".mp3", ".flac")):
                    mapped_mtype = "voice_message"
                elif mfile_l.startswith("t_video") or any(mfile_l.endswith(ext) for ext in (".mp4", ".gif", ".3gp", ".avi", ".mov", ".mkv")):
                    mapped_mtype = "video"
                else:
                    mapped_mtype = "file"
            else:
                mapped_mtype = mt_l

        mpath = resolve_local_media_path(backup_path, f"files/{mfile}") if mfile else None

        message: Dict[str, Any] = {
            "id": int(mid),
            "type": message_type,
            "from": sender,
            "from_id": sender_id_s,
            "date": ts_iso,
            "date_unixtime": ts_unix,
            "text": str(text).strip() if text else "",
            "media_type": mapped_mtype,
            "file": f"files/{mfile}" if mfile else None,
            "media_size": media_size,
            "forwarded_from": str(fwd_from_id) if fwd_from_id is not None else None,
            "reply_markup": json.loads(markup_json) if markup_json else None,
            "media": json.loads(media_json) if media_json else None,
            "unofficial_api_layer": api_layer,
        }
        if source_key:
            upsert_archival_message(
                conn, chat_id, message, backup_path, source_key, "sqlite"
            )
            if raw_data is not None:
                raw_bytes = bytes(raw_data)
                conn.execute(
                    "UPDATE messages SET raw_payload=? WHERE chat_id=? AND message_id=?",
                    (sqlite3.Binary(zlib.compress(raw_bytes, 9)), chat_id, int(mid)),
                )
        else:
            messages_batch.append((
                int(mid), chat_id, sender, sender_id_s, ts_iso, ts_unix,
                str(text).strip() if text else "", mapped_mtype, mpath, None,
                str(fwd_from_id) if fwd_from_id is not None else None,
            ))
            if len(messages_batch) >= batch_size:
                total_inserted += upsert_message_batch(dest_cursor, messages_batch)
                messages_batch.clear()
        total_inserted += 1 if source_key else 0

    if messages_batch:
        total_inserted += upsert_message_batch(dest_cursor, messages_batch)
        messages_batch.clear()

    src_conn.close()
    return total_inserted


def _completed_source_import(
    conn: sqlite3.Connection,
    source_key: str,
    expected_messages: Optional[int] = None,
) -> Optional[int]:
    """Return a completed source's imported count, or ``None`` when retrying.

    Source keys include the content digest, so a completed row is safe to
    reuse on a later scan.  Failed/incomplete rows are deliberately absent
    from ``backup_imports`` and are parsed again from the beginning.
    """
    row = conn.execute(
        "SELECT expected_messages, imported_messages FROM backup_imports WHERE source_key=?",
        (source_key,),
    ).fetchone()
    if row is None:
        return None
    expected, imported = int(row[0]), int(row[1])
    if expected_messages is not None and expected != expected_messages:
        return None
    if expected != imported:
        return None
    return imported


def index_backup_folder(
    root_path: str,
    db_path: str,
    log_fn=print,
    *,
    archive_sources: bool = False,
) -> Tuple[int, int]:
    """
    Recursively scan the backup folder for JSON, HTML, and SQLite files, ingest them in bulk,
    and recreate high-speed database indexes and search tables.
    """
    if os.path.isdir(db_path) or db_path.endswith(("/", "\\")):
        os.makedirs(db_path, exist_ok=True)
        db_path = os.path.join(db_path, "telegram_backup.db")

    log_fn(f"Initializing target database at: {db_path}...")
    conn = setup_database(db_path)

    json_files = []
    html_files = []
    sqlite_files = []

    log_fn("Scanning backup directory recursively...")
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [
            name for name in dirnames
            if not name.startswith(".partial-") and not name.startswith(".dry-run-")
        ]
        if any(exc in dirpath.replace("\\", "/").split("/") for exc in ("css", "js", "profile_pictures", ".git", ".venv")):
            continue

        for fn in filenames:
            fn_l = fn.lower()
            full_p = os.path.join(dirpath, fn)

            if fn_l in ("result.json", "results.json", "export_results.json"):
                json_files.append(full_p)
            elif fn_l.startswith("messages") and fn_l.endswith(".html"):
                html_files.append(full_p)
            elif fn_l == "database.sqlite":
                sqlite_files.append(full_p)

    total_files = len(json_files) + len(html_files) + len(sqlite_files)
    log_fn(f"Discovered {len(json_files)} JSON export files, {len(html_files)} HTML message files, and {len(sqlite_files)} SQLite files.")

    total_messages = 0
    processed_count = 0
    ingest_errors: List[str] = []

    # Process JSON Exports first
    for p in sorted(json_files):
        processed_count += 1
        log_fn(f"[{processed_count}/{total_files}] Ingesting JSON: {os.path.basename(p)}...")
        try:
            source_key = archive_source_file(
                conn, p, "json", archive_payload=archive_sources
            )
            already_imported = _completed_source_import(conn, source_key)
            if already_imported is not None:
                total_messages += already_imported
                log_fn(f"  -> Already indexed {already_imported} messages; reusing completed source.")
                continue
            n, chat_ids = parse_json_file_archival(
                p, conn, root_path=root_path, source_key=source_key
            )
            record_import(
                conn,
                source_key,
                "json",
                p,
                next(iter(chat_ids)) if len(chat_ids) == 1 else None,
                n,
                n,
            )
            conn.execute(
                "UPDATE backup_sources SET message_count=? WHERE source_key=?",
                (n, source_key),
            )
            if archive_sources:
                backfill_source_media_integrity(conn, source_key)
            conn.commit()
            total_messages += n
            log_fn(f"  -> Ingested {n} messages.")
        except Exception as e:
            conn.rollback()
            log_fn(f"  -> ERROR ingesting JSON: {e}")
            ingest_errors.append(f"JSON {p}: {e}")

    # Process HTML Exports second
    for p in sorted(html_files):
        processed_count += 1
        log_fn(f"[{processed_count}/{total_files}] Ingesting HTML: {os.path.relpath(p, root_path)}...")
        try:
            source_key = archive_source_file(
                conn, p, "html", archive_payload=archive_sources
            )
            expected = expected_html_messages(p)
            already_imported = _completed_source_import(conn, source_key, expected)
            if already_imported is not None:
                total_messages += already_imported
                log_fn(f"  -> Already indexed {already_imported} messages; reusing completed source.")
                continue
            n = parse_html_file(
                p, conn, root_path=root_path, source_key=source_key
            )
            if n != expected:
                raise RuntimeError(
                    f"HTML coverage mismatch: expected {expected} real messages, parsed {n}"
                )
            chat_id, _chat_name, _chat_root = infer_html_identity(p, root_path)
            record_import(conn, source_key, "html", p, chat_id, expected, n)
            conn.execute(
                "UPDATE backup_sources SET message_count=? WHERE source_key=?",
                (n, source_key),
            )
            if archive_sources:
                backfill_source_media_integrity(conn, source_key)
            conn.commit()
            total_messages += n
            log_fn(f"  -> Ingested {n} messages.")
        except Exception as e:
            conn.rollback()
            log_fn(f"  -> ERROR ingesting HTML: {e}")
            ingest_errors.append(f"HTML {p}: {e}")

    # Process Unofficial SQLite Exports third
    for p in sorted(sqlite_files):
        processed_count += 1
        log_fn(f"[{processed_count}/{total_files}] Ingesting SQLite: {os.path.basename(p)}...")
        try:
            source_key = archive_source_file(
                conn, p, "sqlite", archive_payload=archive_sources
            )
            with sqlite3.connect(f"file:{p}?mode=ro", uri=True) as source_conn:
                expected = int(
                    source_conn.execute(
                        "SELECT count(*) FROM messages WHERE source_id IS NOT NULL AND source_type IS NOT NULL"
                    ).fetchone()[0]
                    )
            already_imported = _completed_source_import(conn, source_key, expected)
            if already_imported is not None:
                total_messages += already_imported
                log_fn(f"  -> Already indexed {already_imported} messages; reusing completed source.")
                continue
            n = parse_sqlite_backup_file(p, conn, source_key=source_key)
            if n != expected:
                raise RuntimeError(
                    f"SQLite coverage mismatch: expected {expected} meaningful messages, parsed {n}"
                )
            record_import(conn, source_key, "sqlite", p, None, expected, n)
            conn.execute(
                "UPDATE backup_sources SET message_count=? WHERE source_key=?",
                (n, source_key),
            )
            if archive_sources:
                backfill_source_media_integrity(conn, source_key)
            conn.commit()
            total_messages += n
            log_fn(f"  -> Ingested {n} messages.")
        except Exception as e:
            conn.rollback()
            log_fn(f"  -> ERROR ingesting SQLite: {e}")
            ingest_errors.append(f"SQLite {p}: {e}")

    log_fn("Generating post-load indexes, search tables, and synchronization triggers...")
    create_post_load_indexes(conn)

    cursor = conn.cursor()
    # Invalidate GUI stats cache in database if those columns exist
    try:
        cursor.execute("UPDATE chats SET min_msg_id = NULL, max_msg_id = NULL, msg_count = NULL, min_timestamp = NULL, max_timestamp = NULL, min_timestamp_unix = NULL, max_timestamp_unix = NULL;")
    except sqlite3.OperationalError:
        pass # Columns don't exist yet in the database (GUI hasn't run yet)

    # Invalidate clusters JSON cache if it exists
    clusters_path = db_path
    if clusters_path.endswith(".db"):
        clusters_path = clusters_path[:-3] + "_clusters.json"
    else:
        clusters_path = clusters_path + "_clusters.json"
    if os.path.exists(clusters_path):
        try:
            os.remove(clusters_path)
        except Exception:
            pass

    cursor.execute("SELECT count(*) FROM chats;")
    chats_count = cursor.fetchone()[0];

    cursor.execute("SELECT count(*) FROM messages;")
    msgs_count = cursor.fetchone()[0]

    # The denormalized-stat invalidation above is a real write; commit it
    # before closing so a successful import cannot leave stale cached ranges.
    conn.commit()
    conn.close()

    if ingest_errors:
        raise RuntimeError(
            f"{len(ingest_errors)} backup file(s) failed to index; first error: {ingest_errors[0]}"
        )

    log_fn("Database optimization completed successfully.")
    return chats_count, msgs_count


def verify_database_archive(
    db_path: str,
    *,
    require_archived_sources: bool = False,
    check_media: bool = False,
) -> List[str]:
    """Return strict integrity/coverage/media errors for one canonical database."""
    errors: List[str] = []
    conn = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        integrity = [str(row[0]) for row in conn.execute("PRAGMA integrity_check")]
        if integrity != ["ok"]:
            errors.append(f"SQLite integrity_check failed: {integrity[:3]}")
        tables = {
            str(row[0]) for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
            )
        }
        required_tables = {
            "chats", "messages", "backup_sources", "backup_import_files",
            "message_sources", "message_source_media",
        }
        missing_tables = sorted(required_tables - tables)
        if missing_tables:
            errors.append(
                "database has not been migrated to the archival schema; missing tables: "
                + ", ".join(missing_tables)
            )
            return errors
        foreign_keys = conn.execute("PRAGMA foreign_key_check").fetchmany(10)
        if foreign_keys:
            errors.append(f"foreign-key violations: {len(foreign_keys)} or more")
        coverage = conn.execute(
            """SELECT count(*) FROM backup_import_files
               WHERE expected_messages != imported_messages + skipped_records"""
        ).fetchone()[0]
        if coverage:
            errors.append(f"{coverage} source import(s) have incomplete record coverage")
        orphan_provenance = conn.execute(
            """SELECT count(*) FROM message_sources AS s
               LEFT JOIN messages AS m
                 ON m.chat_id=s.chat_id AND m.message_id=s.message_id
               WHERE m.id IS NULL"""
        ).fetchone()[0]
        if orphan_provenance:
            errors.append(f"{orphan_provenance} provenance row(s) refer to absent messages")
        if require_archived_sources:
            unprovenanced = conn.execute(
                """SELECT count(*) FROM messages AS m WHERE NOT EXISTS (
                       SELECT 1 FROM message_sources AS s
                       WHERE s.chat_id=m.chat_id AND s.message_id=m.message_id
                   )"""
            ).fetchone()[0]
            if unprovenanced:
                errors.append(f"{unprovenanced} message row(s) have no source provenance")

        if "messages_fts" in tables:
            try:
                message_count = int(conn.execute("SELECT count(*) FROM messages").fetchone()[0])
                fts_count = int(conn.execute("SELECT count(*) FROM messages_fts").fetchone()[0])
                if fts_count != message_count:
                    errors.append(
                        f"FTS row count {fts_count} does not match messages row count {message_count}"
                    )
                trigger_names = {
                    str(row[0]) for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ('messages_ai','messages_ad','messages_au')"
                    )
                }
                missing_triggers = {"messages_ai", "messages_ad", "messages_au"} - trigger_names
                if missing_triggers:
                    errors.append(f"FTS synchronization trigger(s) missing: {', '.join(sorted(missing_triggers))}")
                sample = conn.execute(
                    "SELECT text FROM messages WHERE text IS NOT NULL AND trim(text) != '' LIMIT 1"
                ).fetchone()
                if sample:
                    token = re.search(r"[\w]{3,}", str(sample[0]), flags=re.UNICODE)
                    if token and int(conn.execute(
                        "SELECT count(*) FROM messages_fts WHERE messages_fts MATCH ?",
                        (f'"{token.group(0)}"',),
                    ).fetchone()[0]) == 0:
                        errors.append("FTS index contains no searchable term for a non-empty message")
            except sqlite3.DatabaseError as exc:
                errors.append(f"FTS integrity check failed: {exc}")

        for source in conn.execute("SELECT * FROM backup_sources ORDER BY source_key"):
            compression = str(source["compression"])
            payload = bytes(source["payload"])
            source_format = str(source["source_format"])
            if require_archived_sources and source_format != "telegram_api" and compression != "zlib":
                errors.append(f"source is registered but not embedded: {source['original_path']}")
                continue
            if compression == "zlib":
                try:
                    decoded = zlib.decompress(payload)
                except zlib.error as exc:
                    errors.append(f"source payload cannot be decompressed ({source['source_key']}): {exc}")
                    continue
                if source_format != "telegram_api":
                    if len(decoded) != int(source["content_size"]):
                        errors.append(f"source size mismatch: {source['original_path']}")
                    if hashlib.sha256(decoded).hexdigest() != source["content_sha256"]:
                        errors.append(f"source hash mismatch: {source['original_path']}")
                else:
                    if "telegram_backup_run_messages" not in tables:
                        errors.append(
                            "Telegram API source exists but telegram_backup_run_messages is missing"
                        )
                        continue
                    try:
                        manifest = json.loads(decoded)
                        run_key = str(manifest["run_key"])
                    except (ValueError, KeyError, TypeError) as exc:
                        errors.append(f"invalid Telegram API manifest {source['source_key']}: {exc}")
                        continue
                    digest = hashlib.sha256()
                    count = 0
                    archive_table = "telegram_backup_run_records" if "telegram_backup_run_records" in tables else "telegram_backup_run_messages"
                    for row in conn.execute(
                        f"SELECT record_json FROM {archive_table} WHERE run_key=? ORDER BY message_id",
                        (run_key,),
                    ):
                        digest.update(str(row[0]).encode("utf-8"))
                        digest.update(b"\n")
                        count += 1
                    if digest.hexdigest() != source["content_sha256"]:
                        errors.append(f"Telegram API run hash mismatch: {run_key}")
                    if count != int(source["message_count"] or 0):
                        errors.append(f"Telegram API run count mismatch: {run_key}")

        if check_media:
            for row in conn.execute(
                """SELECT chat_id, message_id, media_path, media_size, media_sha256,
                          media_status FROM messages WHERE media_type IS NOT NULL"""
            ):
                path_value = row["media_path"]
                status = row["media_status"]
                if not path_value:
                    if status not in ("skipped", "error"):
                        errors.append(
                            f"media has no file/status: {row['chat_id']} message {row['message_id']}"
                        )
                    continue
                parsed = urllib.parse.urlparse(str(path_value))
                if parsed.scheme:
                    continue
                media_path = os.path.abspath(str(path_value))
                if not os.path.isfile(media_path):
                    errors.append(f"media file missing: {media_path}")
                    continue
                expected_size = row["media_size"]
                if expected_size is not None and os.path.getsize(media_path) != int(expected_size):
                    errors.append(f"media size mismatch: {media_path}")
                    continue
                expected_hash = row["media_sha256"]
                if expected_hash:
                    digest = hashlib.sha256()
                    with open(media_path, "rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                    if digest.hexdigest() != expected_hash:
                        errors.append(f"media hash mismatch: {media_path}")
    finally:
        conn.close()
    return errors


def main(argv: Optional[List[str]] = None) -> int:
    """Run the legacy-export importer or canonical database verifier."""
    import argparse

    class _Ansi:
        RESET = "\033[0m"
        CYAN = "\033[36m"
        GREEN = "\033[32m"
        RED = "\033[31m"

    def _use_color() -> bool:
        if os.environ.get("NO_COLOR"):
            return False
        try:
            return sys.stdout.isatty() and os.environ.get("TERM", "") not in ("", "dumb")
        except Exception:
            return False

    def _c(s: str, code: str) -> str:
        if not _use_color():
            return s
        return f"{code}{s}{_Ansi.RESET}"

    ap = argparse.ArgumentParser(description="Telegram Backup SQLite Indexer")
    ap.add_argument("path", nargs="?", help="Telegram export folder (will be scanned recursively)")
    ap.add_argument(
        "--export-db",
        required=False,
        help="Path to a SQLite database where all messages will be imported.",
    )
    ap.add_argument(
        "--archive-sources",
        action="store_true",
        help="Embed exact compressed HTML/JSON/SQLite source files for lossless archival",
    )
    ap.add_argument(
        "--verify-db",
        help="Verify an existing canonical database instead of importing sources",
    )
    ap.add_argument(
        "--require-archived-sources",
        action="store_true",
        help="Fail verification unless every file source is embedded and hash-valid",
    )
    ap.add_argument(
        "--check-media",
        action="store_true",
        help="Verify every local media path, expected size, and recorded SHA-256",
    )
    args = ap.parse_args(argv)

    if args.verify_db:
        problems = verify_database_archive(
            args.verify_db,
            require_archived_sources=args.require_archived_sources,
            check_media=args.check_media,
        )
        if problems:
            for problem in problems:
                print(_c(f"ERROR: {problem}", _Ansi.RED), file=sys.stderr)
            return 4
        print(_c(f"Verified canonical database: {os.path.abspath(args.verify_db)}", _Ansi.GREEN))
        return 0
    if not args.path or not args.export_db:
        ap.error("path and --export-db are required unless --verify-db is used")

    root = os.path.abspath(args.path)
    if not os.path.exists(root):
        print(f"Path does not exist: {root}", file=sys.stderr)
        return 2

    db_path = os.path.abspath(args.export_db)
    try:
        chats_count, msgs_count = index_backup_folder(
            root,
            db_path,
            log_fn=lambda msg: print(_c(msg, _Ansi.CYAN)),
            archive_sources=args.archive_sources,
        )
        print(_c(f"Successfully indexed {msgs_count} messages across {chats_count} chats.", _Ansi.GREEN))
        return 0
    except Exception as e:
        print(_c(f"Database ingestion failed: {e}", _Ansi.RED), file=sys.stderr)
        return 3


# Canonical database primitives live in ``tgbackup.db``.  Keep the historical
# names in this module as compatibility aliases for older importer callers;
# parser-specific code above remains local to the legacy ingestion service.
from ..db.archive import (  # noqa: E402  (aliases intentionally defined at module end)
    archival_message_values,
    flatten_telegram_text,
    resolve_local_media_path,
    upsert_archival_message,
)
from ..db.sources import archive_source_file, backfill_source_media_integrity, record_import  # noqa: E402
from ..db.schema import (  # noqa: E402
    ensure_search_schema,
    setup_database,
)


if __name__ == "__main__":
    raise SystemExit(main())
