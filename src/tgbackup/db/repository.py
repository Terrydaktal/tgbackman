"""Target and chat selection queries.

The exporter and GUI share these queries so active/blacklisted semantics stay
consistent regardless of which front end starts a backup.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Optional

from ..config import BLACKLIST_TABLE, TARGETS_TABLE, TARGET_CHAT_LINKS_TABLE
from ..models import DatabaseChat, Target


def database_chats(conn: sqlite3.Connection, *, active_only: bool = False) -> list[DatabaseChat]:
    predicate = "COALESCE(is_active, 0) = 1 AND " if active_only else ""
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(chats)").fetchall()}
    timestamp_expression = "max_timestamp_unix" if "max_timestamp_unix" in columns else "NULL"
    rows = conn.execute(
        f"""SELECT chat_id, chat_name, COALESCE(is_active, 0) AS is_active,
                   {timestamp_expression} AS max_timestamp_unix
            FROM chats WHERE {predicate}chat_name IS NOT NULL AND trim(chat_name) <> ''
            ORDER BY lower(trim(chat_name)), chat_id"""
    ).fetchall()
    return [DatabaseChat(
        chat_id=str(row["chat_id"]), name=str(row["chat_name"]).strip(),
        is_active=bool(row["is_active"]),
        max_timestamp_unix=(int(row["max_timestamp_unix"])
                            if row["max_timestamp_unix"] is not None else None),
    ) for row in rows]


def active_chats(conn: sqlite3.Connection) -> list[DatabaseChat]:
    return database_chats(conn, active_only=True)


def row_to_target(row: sqlite3.Row) -> Target:
    return Target(
        target_key=str(row["target_key"]), source_name=str(row["source_name"]),
        chat_id=str(row["chat_id"]), peer_kind=str(row["peer_kind"]), peer_id=int(row["peer_id"]),
        access_hash=int(row["access_hash"]) if row["access_hash"] is not None else None,
        title=str(row["title"]), username=str(row["username"]) if row["username"] else None,
        enabled=bool(row["enabled"]), output_dir=str(row["output_dir"]) if row["output_dir"] else None,
        last_message_id=int(row["last_message_id"]) if row["last_message_id"] is not None else None,
        last_message_unix=int(row["last_message_unix"]) if row["last_message_unix"] is not None else None,
        last_export_unix=int(row["last_export_unix"]) if row["last_export_unix"] is not None else None,
    )


def load_targets(conn: sqlite3.Connection, active_only: bool = False) -> list[Target]:
    rows = conn.execute(f"SELECT * FROM {TARGETS_TABLE} ORDER BY lower(source_name), target_key").fetchall()
    if not active_only:
        return [row_to_target(row) for row in rows]
    active_keys = {str(row[0]) for row in conn.execute(f"""
        SELECT DISTINCT links.target_key FROM {TARGET_CHAT_LINKS_TABLE} AS links
        JOIN chats ON chats.chat_id = links.chat_id
        WHERE COALESCE(chats.is_active, 0) = 1
          AND (links.match_method <> 'telegram-migrated-from' OR EXISTS
               (SELECT 1 FROM messages WHERE messages.chat_id = links.chat_id))
        UNION SELECT targets.target_key FROM {TARGETS_TABLE} AS targets
        JOIN chats ON chats.chat_id = targets.chat_id
        WHERE COALESCE(chats.is_active, 0) = 1""").fetchall()}
    return [row_to_target(row) for row in rows if bool(row["enabled"]) and row["target_key"] in active_keys]


def blacklisted_target_keys(conn: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in conn.execute(f"""
        SELECT targets.target_key FROM {TARGETS_TABLE} AS targets
        WHERE EXISTS (SELECT 1 FROM {BLACKLIST_TABLE} AS blacklist
                      WHERE blacklist.target_key = targets.target_key
                         OR (blacklist.peer_kind = targets.peer_kind
                             AND blacklist.peer_id = targets.peer_id))""").fetchall()}


def runnable_targets(conn: sqlite3.Connection, *, include_inactive: bool = False) -> list[Target]:
    blocked = blacklisted_target_keys(conn)
    source = load_targets(conn, active_only=not include_inactive)
    return [target for target in source if target.enabled and target.target_key not in blocked]


def set_target_blacklisted(conn: sqlite3.Connection, target: Target, *, blacklisted: bool,
                           reason: Optional[str] = None) -> None:
    if blacklisted:
        conn.execute(f"""INSERT INTO {BLACKLIST_TABLE}
            (target_key, peer_kind, peer_id, title, reason, created_unix)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(peer_kind, peer_id) DO UPDATE SET target_key=excluded.target_key,
                title=excluded.title, reason=COALESCE(excluded.reason, {BLACKLIST_TABLE}.reason)""",
                     (target.target_key, target.peer_kind, target.peer_id, target.title, reason, int(time.time())))
        conn.execute(f"""UPDATE chats SET is_active=0 WHERE chat_id=? OR chat_id IN
            (SELECT chat_id FROM {TARGET_CHAT_LINKS_TABLE} WHERE target_key=?)""",
                     (target.chat_id, target.target_key))
    else:
        conn.execute(f"DELETE FROM {BLACKLIST_TABLE} WHERE target_key=? OR (peer_kind=? AND peer_id=?)",
                     (target.target_key, target.peer_kind, target.peer_id))
    conn.commit()
