#!/usr/bin/env python3
"""
db_indexer.py

High-performance database indexer for Telegram backup files.
Ingests messages from both JSON and HTML backup files recursively,
loading them into a SQLite database designed for ultra-fast queries
using indexes and FTS5 full-text search.
"""

from __future__ import annotations

import html
import json
import os
import re
import sqlite3
import sys
import urllib.parse
from datetime import datetime, timezone, timedelta
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# Re-use regexes and helpers similar to backman.py
HTML_CHAT_NAME_RE = re.compile(r'<div class="text bold">\s*(.*?)\s*</div>', re.DOTALL)

# HTML message class and timestamp regex
HTML_DIV_MESSAGE_MARKER = '<div class="message'
HTML_DAY_SEPARATOR_MARKER = '<div class="message service" id="message-'

# Extract exact Telegram date: "24.05.2026 22:15:30 UTC+01:00" or similar
HTML_MSG_TS_RE = re.compile(
    r"(\d{2})\.(\d{2})\.(\d{4}) (\d{2}):(\d{2}):(\d{2})(?:\s+UTC([+-]\d{2}):(\d{2}))?"
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
    def __init__(self):
        super().__init__()
        self.in_msg = False
        self.msg_depth = 0
        self.depth = 0
        self.cur = None
        self.field = None
        self.messages = []
        self.last_sender = ""

    def handle_starttag(self, tag, attrs):
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
                "forwarded_from": None
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
        if not self.cur or self.cur.get("message_id") is None:
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
            
        self.messages.append({
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
        })
        self.cur = None


def setup_database(db_path: str) -> sqlite3.Connection:
    """
    Connect to the SQLite database and initialize highly optimized performance settings
    along with the base table schema (indexes and triggers deferred for loading speed).
    """
    # Ensure parent directory exists
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    
    # High performance DB settings
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA cache_size = -2000000;")  # 2GB cache
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA foreign_keys = ON;")
    
    cursor = conn.cursor()
    
    # Create Chats table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            chat_id TEXT PRIMARY KEY,
            chat_name TEXT,
            chat_type TEXT,
            backup_path TEXT,
            is_active INTEGER DEFAULT 0,
            last_backup_unix INTEGER
        );
    """)
    
    # In-place migration: add backup_path if database already exists
    try:
        cursor.execute("ALTER TABLE chats ADD COLUMN backup_path TEXT;")
    except sqlite3.OperationalError:
        pass # Column already exists

    # In-place migration: add is_active if database already exists
    try:
        cursor.execute("ALTER TABLE chats ADD COLUMN is_active INTEGER DEFAULT 0;")
    except sqlite3.OperationalError:
        pass # Column already exists

    # In-place migration: add last_backup_unix if database already exists
    try:
        cursor.execute("ALTER TABLE chats ADD COLUMN last_backup_unix INTEGER;")
    except sqlite3.OperationalError:
        pass # Column already exists
    
    # Create Messages table (with Unique index to enforce idempotency via INSERT OR IGNORE)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL,
            chat_id TEXT NOT NULL,
            sender TEXT,
            sender_id TEXT,
            timestamp TEXT,
            timestamp_unix INTEGER,
            text TEXT,
            media_type TEXT,
            media_path TEXT,
            reply_to_id INTEGER,
            forwarded_from TEXT,
            UNIQUE(chat_id, message_id),
            FOREIGN KEY(chat_id) REFERENCES chats(chat_id)
        );
    """)
    conn.commit()
    return conn


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
    
    # Set up FTS5 Virtual Search Table (re-created to include media_path)
    cursor.execute("DROP TABLE IF EXISTS messages_fts;")
    cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(text, media_path, content='messages', content_rowid='id');")
    
    # Populate FTS5 with all currently loaded messages
    cursor.execute("INSERT OR IGNORE INTO messages_fts(rowid, text, media_path) SELECT id, text, media_path FROM messages;")
    
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
    
    # Run optimizer
    cursor.execute("ANALYZE;")
    conn.commit()


def parse_json_file(path: str, conn: sqlite3.Connection, batch_size: int = 50000) -> int:
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
                        media_path = msg.get("file") or msg.get("photo")
                        reply_to_id = msg.get("reply_to_message_id")
                        forwarded_from = msg.get("forwarded_from")
                        
                        messages_batch.append((
                            msg_id, chat_id, sender, sender_id, ts_iso, ts_unix,
                            text, media_type, media_path, reply_to_id, forwarded_from
                        ))
                        
                        if len(messages_batch) >= batch_size:
                            cursor.executemany(
                                """INSERT OR IGNORE INTO messages 
                                (message_id, chat_id, sender, sender_id, timestamp, timestamp_unix, 
                                text, media_type, media_path, reply_to_id, forwarded_from) 
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                messages_batch
                            )
                            total_inserted += cursor.rowcount
                            messages_batch.clear()
                            conn.commit()
                            
            if messages_batch:
                cursor.executemany(
                    """INSERT OR IGNORE INTO messages 
                    (message_id, chat_id, sender, sender_id, timestamp, timestamp_unix, 
                    text, media_type, media_path, reply_to_id, forwarded_from) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    messages_batch
                )
                total_inserted += cursor.rowcount
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
                    media_path = msg.get("file") or msg.get("photo")
                    reply_to_id = msg.get("reply_to_message_id")
                    forwarded_from = msg.get("forwarded_from")
                    
                    messages_batch.append((
                        msg_id, chat_id, sender, sender_id, ts_iso, ts_unix,
                        text, media_type, media_path, reply_to_id, forwarded_from
                    ))
                    
                    if len(messages_batch) >= batch_size:
                        cursor.executemany(
                            """INSERT OR IGNORE INTO messages 
                            (message_id, chat_id, sender, sender_id, timestamp, timestamp_unix, 
                            text, media_type, media_path, reply_to_id, forwarded_from) 
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            messages_batch
                        )
                        total_inserted += cursor.rowcount
                        messages_batch.clear()
                        conn.commit()
                        
            if messages_batch:
                cursor.executemany(
                    """INSERT OR IGNORE INTO messages 
                    (message_id, chat_id, sender, sender_id, timestamp, timestamp_unix, 
                    text, media_type, media_path, reply_to_id, forwarded_from) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    messages_batch
                )
                total_inserted += cursor.rowcount
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
                    media_path = msg.get("file") or msg.get("photo")
                    reply_to_id = msg.get("reply_to_message_id")
                    forwarded_from = msg.get("forwarded_from")
                    
                    messages_batch.append((
                        msg_id, chat_id, sender, sender_id, ts_iso, ts_unix,
                        text, media_type, media_path, reply_to_id, forwarded_from
                    ))
                    
                    if len(messages_batch) >= batch_size:
                        cursor.executemany(
                            """INSERT OR IGNORE INTO messages 
                            (message_id, chat_id, sender, sender_id, timestamp, timestamp_unix, 
                            text, media_type, media_path, reply_to_id, forwarded_from) 
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            messages_batch
                        )
                        total_inserted += cursor.rowcount
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
                media_path = msg.get("file") or msg.get("photo")
                reply_to_id = msg.get("reply_to_message_id")
                forwarded_from = msg.get("forwarded_from")
                
                messages_batch.append((
                    msg_id, chat_id, sender, sender_id, ts_iso, ts_unix,
                    text, media_type, media_path, reply_to_id, forwarded_from
                ))
                
                if len(messages_batch) >= batch_size:
                    cursor.executemany(
                        """INSERT OR IGNORE INTO messages 
                        (message_id, chat_id, sender, sender_id, timestamp, timestamp_unix, 
                        text, media_type, media_path, reply_to_id, forwarded_from) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        messages_batch
                    )
                    total_inserted += cursor.rowcount
                    messages_batch.clear()
                    conn.commit()
                    
        if messages_batch:
            cursor.executemany(
                """INSERT OR IGNORE INTO messages 
                (message_id, chat_id, sender, sender_id, timestamp, timestamp_unix, 
                text, media_type, media_path, reply_to_id, forwarded_from) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                messages_batch
            )
            total_inserted += cursor.rowcount
            messages_batch.clear()
            conn.commit()

        return total_inserted


def parse_html_file(path: str, conn: sqlite3.Connection, batch_size: int = 50000) -> int:
    """
    Ingest a Telegram HTML message file (e.g. messages.html, messages2.html).
    Infers the chat_id from parent directory and extracts the chat_name from header.
    """
    # Infer chat_id from parent directory name (e.g. "chat_001")
    parent_dir = os.path.dirname(path)
    chat_id = os.path.basename(parent_dir)
    
    # If the file is at the root level (single chat), fall back to "single_chat"
    # and infer name from the container directory.
    if chat_id.lower() in ("", "chats") or not os.path.isdir(os.path.join(parent_dir, "css")):
        # Check if this is a structured backup organized in a subfolder under a chat folder.
        # e.g., <BackupRoot>/<ChatName>/<DateRange>/messages.html
        grandparent_dir = os.path.dirname(parent_dir)
        grandparent_name = os.path.basename(grandparent_dir)
        great_grandparent_name = os.path.basename(os.path.dirname(grandparent_dir))
        
        if grandparent_name and (great_grandparent_name in ("Telegram Backup", "Backup", "Telegram (unofficial)") or "Telegram Backup" in parent_dir or "Z__" in chat_id):
            # Keep the unique date-range/folder name as the chat_id and use grandparent_name as the chat_name
            chat_name = grandparent_name
        else:
            # Check if single_chat directory contains messages.html directly
            chat_id = "single_chat"
            chat_name = os.path.basename(parent_dir) or "Single Chat"
    else:
        # Load the chat name
        messages_root = os.path.join(parent_dir, "messages.html")
        chat_name = _extract_chat_name_from_messages_html(messages_root) or chat_id

    cursor = conn.cursor()
    
    # Ensure chat is registered in database
    backup_path = os.path.abspath(os.path.dirname(path))
    cursor.execute(
        "INSERT OR IGNORE INTO chats (chat_id, chat_name, chat_type, backup_path) VALUES (?, ?, ?, ?)",
        (chat_id, chat_name, "personal_chat" if chat_id == "single_chat" else "unknown", backup_path)
    )
    cursor.execute(
        "UPDATE chats SET backup_path = ? WHERE chat_id = ? AND backup_path IS NULL",
        (backup_path, chat_id)
    )
    
    # Parse HTML messages
    parser = TelegramHTMLParser()
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            parser.feed(f.read())
    except Exception as e:
        print(f"Error parsing HTML {path}: {e}", file=sys.stderr)
        return 0

    messages_batch = []
    total_inserted = 0
    
    for msg in parser.messages:
        messages_batch.append((
            msg["message_id"], chat_id, msg["sender"], msg["sender_id"],
            msg["timestamp"], msg["timestamp_unix"], msg["text"],
            msg["media_type"], msg["media_path"], msg["reply_to_id"],
            msg["forwarded_from"]
        ))
        
        if len(messages_batch) >= batch_size:
            cursor.executemany(
                """INSERT OR IGNORE INTO messages 
                (message_id, chat_id, sender, sender_id, timestamp, timestamp_unix, 
                text, media_type, media_path, reply_to_id, forwarded_from) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                messages_batch
            )
            total_inserted += cursor.rowcount
            messages_batch.clear()
            conn.commit()
            
    if messages_batch:
        cursor.executemany(
            """INSERT OR IGNORE INTO messages 
            (message_id, chat_id, sender, sender_id, timestamp, timestamp_unix, 
            text, media_type, media_path, reply_to_id, forwarded_from) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            messages_batch
        )
        total_inserted += cursor.rowcount
        messages_batch.clear()
        conn.commit()
        
    return total_inserted


def parse_sqlite_backup_file(path: str, conn: sqlite3.Connection, batch_size: int = 50000) -> int:
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
            return 0
    except Exception:
        src_conn.close()
        return 0

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
        return 0

    has_media_cols = "media_type" in m_cols and "media_file" in m_cols
    if has_media_cols:
        src_cursor.execute("SELECT message_id, source_id, source_type, sender_id, text, time, media_type, media_file FROM messages")
    else:
        src_cursor.execute("SELECT message_id, source_id, source_type, sender_id, text, time, NULL, NULL FROM messages")
    
    messages_batch = []
    total_inserted = 0
    for row in src_cursor:
        mid, sid, stype, sender_id, text, t_val = row[0], row[1], row[2], row[3], row[4], row[5]
        mtype = row[6] if has_media_cols else None
        mfile = row[7] if has_media_cols else None
        
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
                
        mpath = f"files/{mfile}" if mfile else None
        
        messages_batch.append((
            int(mid), chat_id, sender, sender_id_s, ts_iso, ts_unix,
            str(text).strip() if text else "", mapped_mtype, mpath, None, None
        ))
        
        if len(messages_batch) >= batch_size:
            dest_cursor.executemany(
                """INSERT INTO messages 
                (message_id, chat_id, sender, sender_id, timestamp, timestamp_unix, 
                text, media_type, media_path, reply_to_id, forwarded_from) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, message_id) DO UPDATE SET
                  media_type = COALESCE(messages.media_type, excluded.media_type),
                  media_path = COALESCE(messages.media_path, excluded.media_path)""",
                messages_batch
            )
            total_inserted += dest_cursor.rowcount
            messages_batch.clear()
            conn.commit()
            
    if messages_batch:
        dest_cursor.executemany(
            """INSERT INTO messages 
            (message_id, chat_id, sender, sender_id, timestamp, timestamp_unix, 
            text, media_type, media_path, reply_to_id, forwarded_from) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id, message_id) DO UPDATE SET
              media_type = COALESCE(messages.media_type, excluded.media_type),
              media_path = COALESCE(messages.media_path, excluded.media_path)""",
            messages_batch
        )
        total_inserted += dest_cursor.rowcount
        messages_batch.clear()
        conn.commit()
        
    src_conn.close()
    return total_inserted


def index_backup_folder(root_path: str, db_path: str, log_fn=print) -> Tuple[int, int]:
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
    
    # Process JSON Exports first
    for p in sorted(json_files):
        processed_count += 1
        log_fn(f"[{processed_count}/{total_files}] Ingesting JSON: {os.path.basename(p)}...")
        try:
            n = parse_json_file(p, conn)
            total_messages += n
            log_fn(f"  -> Ingested {n} messages.")
        except Exception as e:
            log_fn(f"  -> ERROR ingesting JSON: {e}")
            
    # Process HTML Exports second
    for p in sorted(html_files):
        processed_count += 1
        log_fn(f"[{processed_count}/{total_files}] Ingesting HTML: {os.path.relpath(p, root_path)}...")
        try:
            n = parse_html_file(p, conn)
            total_messages += n
            log_fn(f"  -> Ingested {n} messages.")
        except Exception as e:
            log_fn(f"  -> ERROR ingesting HTML: {e}")

    # Process Unofficial SQLite Exports third
    for p in sorted(sqlite_files):
        processed_count += 1
        log_fn(f"[{processed_count}/{total_files}] Ingesting SQLite: {os.path.basename(p)}...")
        try:
            n = parse_sqlite_backup_file(p, conn)
            total_messages += n
            log_fn(f"  -> Ingested {n} messages.")
        except Exception as e:
            log_fn(f"  -> ERROR ingesting SQLite: {e}")

    log_fn("Generating post-load indexes, search tables, and synchronization triggers...")
    create_post_load_indexes(conn)
    
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM chats;")
    chats_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT count(*) FROM messages;")
    msgs_count = cursor.fetchone()[0]
    
    conn.close()
    
    log_fn("Database optimization completed successfully.")
    return chats_count, msgs_count


if __name__ == "__main__":
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
    ap.add_argument("path", help="Telegram export folder (will be scanned recursively)")
    ap.add_argument(
        "--export-db",
        required=True,
        help="Path to a SQLite database where all messages will be imported.",
    )
    args = ap.parse_args()

    root = os.path.abspath(args.path)
    if not os.path.exists(root):
        print(f"Path does not exist: {root}", file=sys.stderr)
        sys.exit(2)

    db_path = os.path.abspath(args.export_db)
    try:
        chats_count, msgs_count = index_backup_folder(
            root, db_path, log_fn=lambda msg: print(_c(msg, _Ansi.CYAN))
        )
        print(_c(f"Successfully indexed {msgs_count} messages across {chats_count} chats.", _Ansi.GREEN))
        sys.exit(0)
    except Exception as e:
        print(_c(f"Database ingestion failed: {e}", _Ansi.RED), file=sys.stderr)
        sys.exit(3)

