"""SQLite repositories used by the backup services."""

from .connection import open_database
from .archive import (
    archival_message_values,
    flatten_telegram_text,
    resolve_local_media_path,
    upsert_archival_message,
    upsert_chat_entity_snapshot,
)
from .repository import (
    active_chats,
    blacklisted_target_keys,
    database_chats,
    load_targets,
    row_to_target,
    runnable_targets,
    set_target_blacklisted,
)
from .schema import (
    ensure_archive_schema,
    ensure_search_schema,
    ensure_targets_schema,
    refresh_chat_statistics,
    setup_database,
)
from .sources import archive_source_file, backfill_source_media_integrity, record_import

__all__ = [
    "open_database", "database_chats", "active_chats", "row_to_target", "load_targets",
    "runnable_targets", "blacklisted_target_keys", "set_target_blacklisted",
    "archival_message_values", "flatten_telegram_text", "resolve_local_media_path",
    "upsert_archival_message", "upsert_chat_entity_snapshot", "ensure_archive_schema", "ensure_search_schema",
    "ensure_targets_schema", "refresh_chat_statistics", "setup_database",
    "archive_source_file", "backfill_source_media_integrity", "record_import",
]
