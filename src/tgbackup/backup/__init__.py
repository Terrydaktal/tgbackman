"""Backup-domain services with narrow, independently testable interfaces."""

from .media import (
    download_media,
    flood_wait_seconds,
    media_download_plan,
    media_filename,
    media_type_for,
    sha256_file,
)
from .records import (
    database_run_key,
    json_safe,
    range_dir_name,
    range_dir_name_from_stats,
    sender_label,
)
from .targets import (
    database_peer_hint,
    direct_target_output_dir,
    entity_description,
    generated_chat_id,
    normalized_chat_name,
    path_is_under,
    target_key,
    target_output_dir,
)
from .target_mapping import (
    auto_map_database_chats,
    cache_dialog,
    link_target_chat,
    match_dialogs_to_database_chats,
    materialize_unbacked_target_chats,
    migrated_peer_destination,
)

__all__ = [
    "database_peer_hint", "database_run_key", "direct_target_output_dir",
    "auto_map_database_chats", "cache_dialog", "download_media", "entity_description", "flood_wait_seconds", "generated_chat_id",
    "json_safe", "media_download_plan", "media_filename", "media_type_for",
    "normalized_chat_name", "path_is_under", "range_dir_name", "range_dir_name_from_stats",
    "link_target_chat", "match_dialogs_to_database_chats", "materialize_unbacked_target_chats",
    "migrated_peer_destination", "sender_label", "sha256_file", "target_key", "target_output_dir",
]
