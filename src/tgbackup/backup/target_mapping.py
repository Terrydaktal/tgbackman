"""Telegram dialog-to-database target mapping service."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from ..config import (
    BLACKLIST_TABLE, DIALOGS_TABLE, TARGETS_TABLE, TARGET_CHAT_LINKS_TABLE, safe_component
)
from ..db.repository import row_to_target
from ..errors import ExportError
from ..models import DatabaseChat, Target
from .targets import (
    database_peer_hint, entity_description, generated_chat_id, normalized_chat_name, target_key
)


def unix_now() -> int:
    return int(time.time())

def legacy_chat_id(conn: sqlite3.Connection, source_name: str, peer_kind: str, peer_id: int) -> str:
    """Reuse an existing tgbackman ID where it clearly matches this peer."""
    try:
        rows = conn.execute(
            """
            SELECT chat_id FROM chats
            WHERE lower(trim(chat_name)) = lower(trim(?))
            ORDER BY COALESCE(max_timestamp_unix, 0) DESC
            """,
            (source_name,),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = conn.execute(
            "SELECT chat_id FROM chats WHERE lower(trim(chat_name)) = lower(trim(?))",
            (source_name,),
        ).fetchall()
    suffix = str(peer_id)
    for row in rows:
        value = str(row["chat_id"])
        if value == suffix or value.endswith(f"_{suffix}"):
            return value
    if len(rows) == 1:
        return str(rows[0]["chat_id"])
    prefix = "dialog" if peer_kind == "user" else "group" if peer_kind == "group" else "channel"
    return f"{prefix}_{peer_id}"


def cache_dialog(
    conn: sqlite3.Connection,
    description: tuple[str, str, Optional[str], int, Optional[int], str],
) -> None:
    kind, title, username, peer_id, access_hash, entity_type = description
    conn.execute(
        f"""
        INSERT INTO {DIALOGS_TABLE} (
            peer_kind, peer_id, access_hash, title, username, entity_type, last_seen_unix
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(peer_kind, peer_id) DO UPDATE SET
            access_hash = excluded.access_hash,
            title = excluded.title,
            username = excluded.username,
            entity_type = excluded.entity_type,
            last_seen_unix = excluded.last_seen_unix
        """,
        (kind, peer_id, access_hash, title, username, entity_type, unix_now()),
    )


def migrated_peer_destination(entity: Any) -> Optional[tuple[str, int]]:
    """Return Telegram's authoritative destination for a migrated basic group."""
    migrated_to = getattr(entity, "migrated_to", None)
    channel_id = getattr(migrated_to, "channel_id", None)
    if channel_id is None:
        return None
    return "channel", abs(int(channel_id))


def consolidate_migrated_targets(
    conn: sqlite3.Connection,
    candidates: list[tuple[Any, tuple[str, str, Optional[str], int, Optional[int], str]]],
) -> list[tuple[str, str]]:
    """Move archive links to the current supergroup target and disable its predecessor.

    Telegram's ``Chat.migrated_to`` field is a stable peer relationship, so it
    is safe to use where title/name matching would not be.  Watermarks are not
    copied because basic-group and supergroup message-ID spaces are unrelated.
    """
    known_peers = {(description[0], description[3]) for _, description in candidates}
    consolidated: list[tuple[str, str]] = []
    for entity, description in candidates:
        old_peer = (description[0], description[3])
        new_peer = migrated_peer_destination(entity)
        if new_peer is None or new_peer not in known_peers:
            continue
        old_target = conn.execute(
            f"SELECT target_key FROM {TARGETS_TABLE} WHERE peer_kind=? AND peer_id=?",
            old_peer,
        ).fetchone()
        new_target = conn.execute(
            f"SELECT target_key FROM {TARGETS_TABLE} WHERE peer_kind=? AND peer_id=?",
            new_peer,
        ).fetchone()
        if old_target is None or new_target is None:
            continue
        old_key = str(old_target["target_key"])
        new_key = str(new_target["target_key"])
        if old_key == new_key:
            continue
        conn.execute(
            f"""UPDATE {TARGET_CHAT_LINKS_TABLE}
                SET target_key=?, match_method='telegram-migrated-from', linked_unix=?
                WHERE target_key=?""",
            (new_key, unix_now(), old_key),
        )
        conn.execute(
            f"UPDATE {TARGETS_TABLE} SET enabled=0, updated_unix=? WHERE target_key=?",
            (unix_now(), old_key),
        )
        consolidated.append((old_key, new_key))
    return consolidated


def link_target_chat(
    conn: sqlite3.Connection,
    target_key_value: str,
    chat_id: str,
    match_method: str,
) -> None:
    conflict = conn.execute(
        f"SELECT target_key FROM {TARGET_CHAT_LINKS_TABLE} WHERE chat_id = ? AND target_key <> ?",
        (chat_id, target_key_value),
    ).fetchone()
    if conflict:
        raise ExportError(
            f"Database chat {chat_id!r} is already linked to target {conflict['target_key']!r}"
        )
    conn.execute(
        f"""
        INSERT INTO {TARGET_CHAT_LINKS_TABLE} (target_key, chat_id, match_method, linked_unix)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(target_key, chat_id) DO UPDATE SET
            match_method = excluded.match_method,
            linked_unix = excluded.linked_unix
        """,
        (target_key_value, chat_id, match_method, unix_now()),
    )


def materialize_unbacked_target_chats(conn: sqlite3.Connection) -> tuple[int, int]:
    """Expose runnable or blacklisted Telegram-only targets as zero-message chats.

    ``map --all`` used to cache unmatched Telegram dialogs only in the target
    tables.  The GUI inventories ``chats``, so those valid targets remained
    invisible and could not be selected for backup.  Create one stable
    placeholder for each enabled target that has no archive link yet. Blacklisted
    disabled tombstones are also retained so the viewer can display and unlist
    them. Once a user activates a normal placeholder, the ordinary active-target
    query schedules it; the first successful export updates the same row in place.
    """
    rows = conn.execute(
        f"""
        SELECT target_key, chat_id, title, peer_kind, output_dir
        FROM {TARGETS_TABLE} AS targets
        WHERE (
                COALESCE(enabled, 1) = 1
                OR EXISTS (
                    SELECT 1 FROM {BLACKLIST_TABLE} AS blacklist
                    WHERE blacklist.target_key = targets.target_key
                       OR (blacklist.peer_kind = targets.peer_kind AND blacklist.peer_id = targets.peer_id)
                )
              )
          AND NOT EXISTS (
              SELECT 1 FROM {TARGET_CHAT_LINKS_TABLE} AS links
              WHERE links.target_key = targets.target_key
          )
        ORDER BY lower(title), target_key
        """
    ).fetchall()
    inserted = 0
    linked = 0
    for row in rows:
        chat_type = "personal_chat" if row["peer_kind"] == "user" else str(row["peer_kind"])
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO chats (
                chat_id, chat_name, chat_type, backup_path, is_active,
                last_backup_unix, last_backup_run_unix, last_backup_run_status, msg_count
            ) VALUES (?, ?, ?, ?, 0, NULL, NULL, NULL, 0)
            """,
            (
                str(row["chat_id"]),
                str(row["title"]),
                chat_type,
                str(row["output_dir"]) if row["output_dir"] else None,
            ),
        )
        inserted += max(cursor.rowcount, 0)
        link_target_chat(
            conn,
            str(row["target_key"]),
            str(row["chat_id"]),
            "telegram-discovered",
        )
        linked += 1
    return inserted, linked


def upsert_target(
    conn: sqlite3.Connection,
    source_name: str,
    entity: Any,
    output_root: Path,
    explicit_chat_id: Optional[str] = None,
    *,
    commit: bool = True,
) -> Target:
    kind, title, username, peer_id, access_hash, _ = entity_description(entity)
    key = target_key(source_name, kind, peer_id)
    existing = conn.execute(
        f"SELECT * FROM {TARGETS_TABLE} WHERE target_key = ?", (key,)
    ).fetchone()
    if existing is None:
        existing = conn.execute(
            f"SELECT * FROM {TARGETS_TABLE} WHERE peer_kind = ? AND peer_id = ?",
            (kind, peer_id),
        ).fetchone()
        if existing is not None:
            key = str(existing["target_key"])
    if existing:
        chat_id = str(existing["chat_id"])
        if explicit_chat_id and explicit_chat_id != chat_id:
            conflict = conn.execute(
                f"SELECT target_key FROM {TARGETS_TABLE} WHERE chat_id = ? AND target_key <> ?",
                (explicit_chat_id, key),
            ).fetchone()
            if conflict:
                raise ExportError(f"Chat ID {explicit_chat_id!r} is already mapped to {conflict['target_key']!r}")
            chat_id = explicit_chat_id
        output_dir = existing["output_dir"] or str(output_root / safe_component(title))
        conn.execute(
            f"""
            UPDATE {TARGETS_TABLE}
            SET source_name = ?, chat_id = ?, title = ?, username = ?, access_hash = ?,
                output_dir = ?, updated_unix = ?
            WHERE target_key = ?
            """,
            (source_name, chat_id, title, username, access_hash, output_dir, unix_now(), key),
        )
    else:
        chat_id = explicit_chat_id or legacy_chat_id(conn, source_name, kind, peer_id)
        conflict = conn.execute(
            f"SELECT target_key FROM {TARGETS_TABLE} WHERE chat_id = ? AND target_key <> ?",
            (chat_id, key),
        ).fetchone()
        if conflict:
            raise ExportError(
                f"Chat ID {chat_id!r} is already mapped to target {conflict['target_key']!r}; "
                "use --chat-id to choose a stable unique ID."
            )
        output_dir = str(output_root / safe_component(title))
        conn.execute(
            f"""
            INSERT INTO {TARGETS_TABLE} (
                target_key, source_name, chat_id, peer_kind, peer_id, access_hash,
                title, username, enabled, output_dir, created_unix, updated_unix
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            """,
            (key, source_name, chat_id, kind, peer_id, access_hash, title, username,
             output_dir, unix_now(), unix_now()),
        )
    if commit:
        conn.commit()
    row = conn.execute(f"SELECT * FROM {TARGETS_TABLE} WHERE target_key = ?", (key,)).fetchone()
    assert row is not None
    return row_to_target(row)


def match_dialogs_to_database_chats(
    chats: list[DatabaseChat],
    candidates: list[tuple[Any, tuple[str, str, Optional[str], int, Optional[int], str]]],
) -> tuple[
    dict[tuple[str, int], list[tuple[DatabaseChat, str]]],
    list[tuple[DatabaseChat, str]],
]:
    """Match stable peer hints first, then unique normalized Telegram titles."""
    by_peer: dict[tuple[str, int], list[tuple[Any, tuple[str, str, Optional[str], int, Optional[int], str]]]] = {}
    by_name: dict[str, list[tuple[Any, tuple[str, str, Optional[str], int, Optional[int], str]]]] = {}
    for candidate in candidates:
        description = candidate[1]
        by_peer.setdefault((description[0], description[3]), []).append(candidate)
        by_name.setdefault(normalized_chat_name(description[1]), []).append(candidate)

    assignments: dict[tuple[str, int], list[tuple[DatabaseChat, str]]] = {}
    unresolved: list[tuple[DatabaseChat, str]] = []
    assigned_chat_ids: set[str] = set()

    for chat in chats:
        hint = database_peer_hint(chat.chat_id)
        if hint is None:
            continue
        allowed_kinds, peer_id = hint
        matches = [
            candidate
            for kind in allowed_kinds
            for candidate in by_peer.get((kind, peer_id), [])
        ]
        if len(matches) == 1:
            description = matches[0][1]
            assignments.setdefault((description[0], description[3]), []).append((chat, "peer-id"))
            assigned_chat_ids.add(chat.chat_id)
        elif len(matches) > 1:
            unresolved.append((chat, "peer ID matched more than one Telegram dialog"))
            assigned_chat_ids.add(chat.chat_id)

    for chat in chats:
        if chat.chat_id in assigned_chat_ids:
            continue
        matches = by_name.get(normalized_chat_name(chat.name), [])
        if len(matches) == 1:
            description = matches[0][1]
            assignments.setdefault((description[0], description[3]), []).append((chat, "exact-title"))
        elif len(matches) > 1:
            unresolved.append((chat, "title matches more than one Telegram dialog"))
        else:
            unresolved.append((chat, "no exact Telegram dialog match"))
    return assignments, unresolved


def auto_map_database_chats(
    conn: sqlite3.Connection,
    chats: list[DatabaseChat],
    candidates: list[tuple[Any, tuple[str, str, Optional[str], int, Optional[int], str]]],
    output_root: Path,
    *,
    include_unmatched_dialogs: bool,
) -> tuple[int, int, int, list[tuple[DatabaseChat, str]]]:
    """Cache every dialog and create/link safe target mappings in one transaction."""
    unique_candidates: dict[
        tuple[str, int],
        tuple[Any, tuple[str, str, Optional[str], int, Optional[int], str]],
    ] = {}
    for candidate in candidates:
        description = candidate[1]
        unique_candidates[(description[0], description[3])] = candidate
        cache_dialog(conn, description)

    # Existing explicit and previously verified links are authoritative.  A
    # later title change must not silently remap the same archive row.
    linked_chat_ids = {
        str(row[0])
        for row in conn.execute(f"SELECT chat_id FROM {TARGET_CHAT_LINKS_TABLE}")
    }
    assignments, unresolved = match_dialogs_to_database_chats(
        [chat for chat in chats if chat.chat_id not in linked_chat_ids],
        list(unique_candidates.values()),
    )
    target_count = 0
    linked_count = 0
    try:
        for peer_key, (entity, description) in unique_candidates.items():
            linked_chats = assignments.get(peer_key, [])
            existing = conn.execute(
                f"SELECT * FROM {TARGETS_TABLE} WHERE peer_kind = ? AND peer_id = ?",
                peer_key,
            ).fetchone()
            if existing is not None:
                source_name = str(existing["source_name"])
                explicit_chat_id = None
            elif linked_chats:
                def canonical_score(item: tuple[DatabaseChat, str]) -> tuple[int, int, int, str]:
                    chat, method = item
                    return (
                        1 if method == "peer-id" else 0,
                        1 if chat.is_active else 0,
                        chat.max_timestamp_unix or 0,
                        chat.chat_id,
                    )

                canonical, _ = max(linked_chats, key=canonical_score)
                source_name = canonical.name
                explicit_chat_id = canonical.chat_id
            elif include_unmatched_dialogs:
                source_name = description[1]
                explicit_chat_id = generated_chat_id(description[0], description[3])
            else:
                continue

            target = upsert_target(
                conn,
                source_name,
                entity,
                output_root,
                explicit_chat_id,
                commit=False,
            )
            target_count += 1
            for chat, method in linked_chats:
                link_target_chat(conn, target.target_key, chat.chat_id, method)
                linked_count += 1
        migrated_targets = consolidate_migrated_targets(
            conn, list(unique_candidates.values())
        )
        for old_key, new_key in migrated_targets:
            print(
                f"Consolidated migrated Telegram target {old_key} into {new_key}; "
                "the old basic-group target is disabled."
            )
        discovered_count, discovered_links = materialize_unbacked_target_chats(conn)
        linked_count += discovered_links
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return target_count, linked_count, discovered_count, unresolved
