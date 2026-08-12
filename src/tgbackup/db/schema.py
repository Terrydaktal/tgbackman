"""Canonical SQLite schema and migrations.

The importer and the Telegram exporter share this module.  Schema ownership
therefore stays below both services and no database connection code needs to
import an application-level importer.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3

from ..config import (
    BLACKLIST_TABLE,
    DIALOGS_TABLE,
    EXPORTS_TABLE,
    PURGES_TABLE,
    RUN_ARCHIVE_TABLE,
    RUN_ATTEMPTS_TABLE,
    RUN_MESSAGES_TABLE,
    RUNS_TABLE,
    TARGET_CHAT_LINKS_TABLE,
    TARGETS_TABLE,
)

ARCHIVAL_MESSAGE_COLUMNS = {
    "message_type": "TEXT", "edit_timestamp": "TEXT", "edit_timestamp_unix": "INTEGER",
    "media_size": "INTEGER", "media_sha256": "TEXT", "media_status": "TEXT",
    "grouped_id": "TEXT", "entities_json": "TEXT", "reactions_json": "TEXT",
    "reply_markup_json": "TEXT", "action_json": "TEXT", "forward_json": "TEXT",
    "reply_to_chat_id": "TEXT", "reply_to_peer_kind": "TEXT", "reply_to_peer_id": "INTEGER",
    "reply_to_top_id": "INTEGER", "reply_to_story_id": "INTEGER", "reply_quote_text": "TEXT",
    "reply_quote_entities_json": "TEXT", "reply_quote_offset": "INTEGER", "reply_media_json": "TEXT",
    "raw_tl_payload": "BLOB", "raw_tl_sha256": "TEXT", "raw_tl_layer": "INTEGER",
    "raw_tl_library": "TEXT", "expanded_metadata_json": "TEXT",
    "extra_json": "TEXT", "raw_payload": "BLOB", "source_key": "TEXT",
    "source_format": "TEXT", "is_deleted": "INTEGER NOT NULL DEFAULT 0", "deleted_unix": "INTEGER",
}


def _add_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, sql_type in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")


def ensure_archive_schema(conn: sqlite3.Connection) -> None:
    """Create/migrate chats, messages, and source-provenance tables."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS chats (
            chat_id TEXT PRIMARY KEY, chat_name TEXT, chat_type TEXT, backup_path TEXT,
            is_active INTEGER DEFAULT 0, last_backup_unix INTEGER,
            last_backup_source TEXT, last_backup_confidence TEXT, last_backup_evidence TEXT,
            last_backup_run_unix INTEGER, last_backup_run_status TEXT
        )"""
    )
    _add_columns(conn, "chats", {
        "backup_path": "TEXT", "is_active": "INTEGER DEFAULT 0", "last_backup_unix": "INTEGER",
        "last_backup_source": "TEXT", "last_backup_confidence": "TEXT", "last_backup_evidence": "TEXT",
        "last_backup_run_unix": "INTEGER", "last_backup_run_status": "TEXT",
        "min_msg_id": "INTEGER", "max_msg_id": "INTEGER", "msg_count": "INTEGER",
        "min_timestamp": "TEXT", "max_timestamp": "TEXT", "min_timestamp_unix": "INTEGER",
        "max_timestamp_unix": "INTEGER",
    })
    conn.execute(
        """CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, message_id INTEGER NOT NULL, chat_id TEXT NOT NULL,
            sender TEXT, sender_id TEXT, timestamp TEXT, timestamp_unix INTEGER, text TEXT,
            media_type TEXT, media_path TEXT, reply_to_id INTEGER, forwarded_from TEXT,
            UNIQUE(chat_id, message_id), FOREIGN KEY(chat_id) REFERENCES chats(chat_id)
        )"""
    )
    _add_columns(conn, "messages", ARCHIVAL_MESSAGE_COLUMNS)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS backup_sources (
            source_key TEXT PRIMARY KEY, source_format TEXT NOT NULL, original_path TEXT NOT NULL,
            content_sha256 TEXT NOT NULL, content_size INTEGER NOT NULL, compressed_size INTEGER NOT NULL,
            compression TEXT NOT NULL, payload BLOB NOT NULL, imported_unix INTEGER NOT NULL,
            message_count INTEGER, UNIQUE(original_path, content_sha256)
        );
        CREATE TABLE IF NOT EXISTS backup_imports (
            source_key TEXT PRIMARY KEY, source_format TEXT NOT NULL, original_path TEXT NOT NULL,
            chat_id TEXT, expected_messages INTEGER NOT NULL, imported_messages INTEGER NOT NULL,
            skipped_records INTEGER NOT NULL DEFAULT 0, completed_unix INTEGER NOT NULL,
            FOREIGN KEY(source_key) REFERENCES backup_sources(source_key)
        );
        CREATE TABLE IF NOT EXISTS backup_import_files (
            source_key TEXT NOT NULL, original_path TEXT NOT NULL, source_format TEXT NOT NULL,
            chat_id TEXT, expected_messages INTEGER NOT NULL, imported_messages INTEGER NOT NULL,
            skipped_records INTEGER NOT NULL DEFAULT 0, completed_unix INTEGER NOT NULL,
            PRIMARY KEY(source_key, original_path), FOREIGN KEY(source_key) REFERENCES backup_sources(source_key)
        );
        CREATE TABLE IF NOT EXISTS message_sources (
            chat_id TEXT NOT NULL, message_id INTEGER NOT NULL, source_key TEXT NOT NULL,
            PRIMARY KEY(chat_id, message_id, source_key), FOREIGN KEY(source_key) REFERENCES backup_sources(source_key)
        );
        CREATE TABLE IF NOT EXISTS message_source_media (
          source_key TEXT NOT NULL, chat_id TEXT NOT NULL, message_id INTEGER NOT NULL,
          media_path TEXT, media_size INTEGER, media_sha256 TEXT, media_status TEXT,
          checked_unix INTEGER NOT NULL, PRIMARY KEY(source_key, chat_id, message_id)
        );
        CREATE TABLE IF NOT EXISTS archive_schema_migrations (
          migration_name TEXT PRIMARY KEY, applied_unix INTEGER NOT NULL,
          affected_rows INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS telegram_entity_snapshots (
          snapshot_sha256 TEXT PRIMARY KEY, peer_kind TEXT, peer_id INTEGER,
          entity_type TEXT NOT NULL, entity_json TEXT NOT NULL, tl_payload BLOB,
          tl_sha256 TEXT, telethon_layer INTEGER, telethon_version TEXT,
          first_captured_unix INTEGER NOT NULL, last_captured_unix INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS telegram_message_entity_refs (
          chat_id TEXT NOT NULL, message_id INTEGER NOT NULL, role TEXT NOT NULL,
          snapshot_sha256 TEXT NOT NULL,
          PRIMARY KEY(chat_id, message_id, role),
          FOREIGN KEY(snapshot_sha256) REFERENCES telegram_entity_snapshots(snapshot_sha256)
        );
        CREATE TABLE IF NOT EXISTS telegram_chat_entity_refs (
          chat_id TEXT NOT NULL, snapshot_sha256 TEXT NOT NULL,
          captured_unix INTEGER NOT NULL, source_key TEXT,
          role TEXT NOT NULL DEFAULT 'entity',
          PRIMARY KEY(chat_id, snapshot_sha256),
          FOREIGN KEY(snapshot_sha256) REFERENCES telegram_entity_snapshots(snapshot_sha256)
        );
        CREATE TABLE IF NOT EXISTS telegram_chat_snapshot_sources (
          chat_id TEXT NOT NULL, snapshot_sha256 TEXT NOT NULL,
          source_key TEXT NOT NULL, role TEXT NOT NULL,
          captured_unix INTEGER NOT NULL,
          PRIMARY KEY(chat_id, snapshot_sha256, source_key, role),
          FOREIGN KEY(snapshot_sha256) REFERENCES telegram_entity_snapshots(snapshot_sha256),
          FOREIGN KEY(source_key) REFERENCES backup_sources(source_key)
        );
        CREATE INDEX IF NOT EXISTS idx_messages_source_key ON messages(source_key);
        CREATE INDEX IF NOT EXISTS idx_messages_reply_target
            ON messages(reply_to_chat_id, reply_to_id);
        CREATE INDEX IF NOT EXISTS idx_message_entity_refs_snapshot
            ON telegram_message_entity_refs(snapshot_sha256);
        CREATE INDEX IF NOT EXISTS idx_chat_entity_refs_snapshot
            ON telegram_chat_entity_refs(snapshot_sha256);
        CREATE INDEX IF NOT EXISTS idx_chat_snapshot_sources_source
            ON telegram_chat_snapshot_sources(source_key);
        CREATE INDEX IF NOT EXISTS idx_chat_snapshot_sources_chat_source
            ON telegram_chat_snapshot_sources(chat_id, source_key);
        """
    )
    _add_columns(
        conn,
        "telegram_chat_entity_refs",
        {"role": "TEXT NOT NULL DEFAULT 'entity'"},
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_chat_entity_refs_chat_role
           ON telegram_chat_entity_refs(chat_id, role)"""
    )


def ensure_search_schema(conn: sqlite3.Connection) -> None:
    """Create B-tree/FTS indexes and synchronization triggers."""
    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_messages_chat_ts ON messages(chat_id, timestamp_unix);
        CREATE INDEX IF NOT EXISTS idx_messages_chat_ts_id
            ON messages(chat_id, timestamp_unix, message_id);
        CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages(sender);
        CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(timestamp_unix);
        """
    )
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(messages_fts)")}
    if columns and "media_path" not in columns:
        conn.execute("DROP TABLE messages_fts")
        columns = set()
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING "
        "fts5(text, media_path, content='messages', content_rowid='id')"
    )
    conn.executescript(
        """
        DROP TRIGGER IF EXISTS messages_ai;
        CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
          INSERT INTO messages_fts(rowid, text, media_path) VALUES (new.id, new.text, new.media_path);
        END;
        DROP TRIGGER IF EXISTS messages_ad;
        CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
          INSERT INTO messages_fts(messages_fts, rowid, text, media_path)
          VALUES('delete', old.id, old.text, old.media_path);
        END;
        DROP TRIGGER IF EXISTS messages_au;
        CREATE TRIGGER messages_au AFTER UPDATE ON messages BEGIN
          INSERT INTO messages_fts(messages_fts, rowid, text, media_path)
          VALUES('delete', old.id, old.text, old.media_path);
          INSERT INTO messages_fts(rowid, text, media_path) VALUES (new.id, new.text, new.media_path);
        END;
        """
    )
    if not columns:
        conn.execute("INSERT INTO messages_fts(messages_fts) VALUES ('rebuild')")


def ensure_targets_schema(conn: sqlite3.Connection) -> None:
    """Create exporter ledger, mapping, blacklist, and purge tables."""
    ensure_archive_schema(conn)
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS {TARGETS_TABLE} (
          target_key TEXT PRIMARY KEY, source_name TEXT NOT NULL, chat_id TEXT NOT NULL UNIQUE,
          peer_kind TEXT NOT NULL, peer_id INTEGER NOT NULL, access_hash INTEGER, title TEXT NOT NULL,
          username TEXT, enabled INTEGER NOT NULL DEFAULT 1, output_dir TEXT, last_message_id INTEGER,
          last_message_unix INTEGER, last_export_unix INTEGER, created_unix INTEGER NOT NULL,
          updated_unix INTEGER NOT NULL, UNIQUE(peer_kind, peer_id)
        );
        CREATE TABLE IF NOT EXISTS {TARGET_CHAT_LINKS_TABLE} (
          target_key TEXT NOT NULL, chat_id TEXT NOT NULL, match_method TEXT NOT NULL,
          linked_unix INTEGER NOT NULL, PRIMARY KEY(target_key, chat_id), UNIQUE(chat_id)
        );
        CREATE TABLE IF NOT EXISTS {DIALOGS_TABLE} (
          peer_kind TEXT NOT NULL, peer_id INTEGER NOT NULL, access_hash INTEGER, title TEXT NOT NULL,
          username TEXT, entity_type TEXT NOT NULL, last_seen_unix INTEGER NOT NULL,
          PRIMARY KEY(peer_kind, peer_id)
        );
        CREATE TABLE IF NOT EXISTS {RUNS_TABLE} (
          run_key TEXT PRIMARY KEY, target_key TEXT NOT NULL, chat_id TEXT NOT NULL,
          baseline_message_id INTEGER, baseline_unix INTEGER, full_rescan INTEGER NOT NULL DEFAULT 0,
          status TEXT NOT NULL, started_unix INTEGER NOT NULL, completed_unix INTEGER, error TEXT
        );
        CREATE TABLE IF NOT EXISTS {RUN_MESSAGES_TABLE} (
          run_key TEXT NOT NULL, message_id INTEGER NOT NULL, record_json TEXT NOT NULL,
          media_error TEXT, PRIMARY KEY(run_key, message_id),
          FOREIGN KEY(run_key) REFERENCES {RUNS_TABLE}(run_key) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS {RUN_ARCHIVE_TABLE} (
          run_key TEXT NOT NULL, message_id INTEGER NOT NULL, record_json TEXT NOT NULL,
          media_error TEXT, PRIMARY KEY(run_key, message_id)
        );
        CREATE TABLE IF NOT EXISTS {RUN_ATTEMPTS_TABLE} (
          attempt_key TEXT PRIMARY KEY, run_key TEXT NOT NULL, started_unix INTEGER NOT NULL,
          completed_unix INTEGER, status TEXT NOT NULL, error TEXT
        );
        CREATE TABLE IF NOT EXISTS {EXPORTS_TABLE} (
          export_key TEXT PRIMARY KEY, target_key TEXT NOT NULL, source_name TEXT NOT NULL,
          chat_id TEXT NOT NULL, output_path TEXT NOT NULL UNIQUE, message_count INTEGER NOT NULL DEFAULT 0,
          first_message_id INTEGER, last_message_id INTEGER, first_message_unix INTEGER,
          last_message_unix INTEGER, created_unix INTEGER NOT NULL, indexed_unix INTEGER, applied_unix INTEGER
        );
        CREATE TABLE IF NOT EXISTS {PURGES_TABLE} (
          purge_key TEXT PRIMARY KEY, target_key TEXT NOT NULL, title TEXT NOT NULL, chat_ids_json TEXT NOT NULL,
          manifest_json TEXT NOT NULL, status TEXT NOT NULL, created_unix INTEGER NOT NULL,
          completed_unix INTEGER, error TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_{PURGES_TABLE}_target ON {PURGES_TABLE}(target_key, created_unix);
        CREATE TABLE IF NOT EXISTS {BLACKLIST_TABLE} (
          target_key TEXT PRIMARY KEY, peer_kind TEXT NOT NULL, peer_id INTEGER NOT NULL, title TEXT NOT NULL,
          reason TEXT, created_unix INTEGER NOT NULL, UNIQUE(peer_kind, peer_id)
        );
        """
    )
    # Every application connection must have the same searchable schema.  An
    # older database can have all archival tables but still be missing FTS
    # triggers, so this cannot be limited to fresh database creation.
    ensure_search_schema(conn)
    conn.execute(
        f"""INSERT OR IGNORE INTO {TARGET_CHAT_LINKS_TABLE}(target_key, chat_id, match_method, linked_unix)
            SELECT target_key, chat_id, 'canonical', strftime('%s','now') FROM {TARGETS_TABLE}
            WHERE EXISTS (SELECT 1 FROM chats WHERE chats.chat_id={TARGETS_TABLE}.chat_id)"""
    )
    # This one-time local migration reads only already-archived API reply rows;
    # it never contacts Telegram or duplicates referenced message bodies.
    from .archive import backfill_reply_metadata

    backfill_reply_metadata(conn)
    conn.commit()


def refresh_chat_statistics(conn: sqlite3.Connection, chat_id: str) -> None:
    row = conn.execute(
        """SELECT MIN(message_id), MAX(message_id), COUNT(*), MIN(timestamp), MAX(timestamp),
                  MIN(timestamp_unix), MAX(timestamp_unix) FROM messages
           WHERE chat_id=? AND COALESCE(is_deleted, 0)=0""", (chat_id,)
    ).fetchone()
    conn.execute(
        """UPDATE chats SET min_msg_id=?, max_msg_id=?, msg_count=?, min_timestamp=?, max_timestamp=?,
                  min_timestamp_unix=?, max_timestamp_unix=? WHERE chat_id=?""", (*row, chat_id)
    )


def setup_database(db_path: str | os.PathLike[str]) -> sqlite3.Connection:
    """Open a canonical database and apply all schema migrations."""
    path = os.fspath(db_path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(parent, 0o700)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-2000000")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA foreign_keys=ON")
    ensure_targets_schema(conn)
    ensure_search_schema(conn)
    conn.commit()
    for candidate in (path, f"{path}-wal", f"{path}-shm"):
        with contextlib.suppress(OSError):
            os.chmod(candidate, 0o600)
    return conn
