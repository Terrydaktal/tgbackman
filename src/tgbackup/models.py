"""Typed domain objects shared by backup, database, and maintenance services."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ExportStats:
    """Streaming export statistics used for watermarks and manifests."""

    message_count: int = 0
    first_message_id: Optional[int] = None
    last_message_id: Optional[int] = None
    first_message_unix: Optional[int] = None
    last_message_unix: Optional[int] = None
    media_count: int = 0
    skipped_media_count: int = 0
    media_errors: list[str] = field(default_factory=list)

    def observe(self, record: dict[str, Any], error: Optional[str]) -> None:
        message_id = int(record["id"]) if record.get("id") is not None else None
        timestamp = record.get("date_unixtime")
        message_unix = (
            int(timestamp)
            if timestamp is not None and str(timestamp).lstrip("-").isdigit()
            else None
        )
        self.message_count += 1
        if message_id is not None:
            self.first_message_id = (
                message_id if self.first_message_id is None else min(self.first_message_id, message_id)
            )
            self.last_message_id = (
                message_id if self.last_message_id is None else max(self.last_message_id, message_id)
            )
        if message_unix is not None:
            self.first_message_unix = (
                message_unix
                if self.first_message_unix is None
                else min(self.first_message_unix, message_unix)
            )
            self.last_message_unix = (
                message_unix
                if self.last_message_unix is None
                else max(self.last_message_unix, message_unix)
            )
        if record.get("media_type"):
            self.media_count += 1
        if record.get("media_skipped"):
            self.skipped_media_count += 1
        if error:
            self.media_errors.append(error)


@dataclass(frozen=True)
class DatabaseChat:
    chat_id: str
    name: str
    is_active: bool
    max_timestamp_unix: Optional[int]


@dataclass(frozen=True)
class MediaDownloadPlan:
    """One exact Telegram media object and the representation to download."""

    primary: Any
    media_type: str
    filename: str
    expected_size: Optional[int]
    thumb_type: Optional[str] = None


@dataclass
class Target:
    target_key: str
    source_name: str
    chat_id: str
    peer_kind: str
    peer_id: int
    access_hash: Optional[int]
    title: str
    username: Optional[str]
    enabled: bool
    output_dir: Optional[str]
    last_message_id: Optional[int]
    last_message_unix: Optional[int]
    last_export_unix: Optional[int]


@dataclass
class PurgePlan:
    target: Target
    chat_ids: list[str]
    chat_names: list[str]
    message_count: int
    backup_root: Optional[Path]
    media_files: list[Path]
    media_directories: list[Path]
    shared_media: list[Path]
    missing_media: list[str]
    unsafe_media: list[str]
    exclusive_source_keys: list[str]
    retained_source_keys: list[str]
    retained_source_paths: list[str]
    bytes_to_delete: int

    def manifest(self) -> dict[str, Any]:
        return {
            "target_key": self.target.target_key,
            "title": self.target.title,
            "peer": f"{self.target.peer_kind}:{self.target.peer_id}",
            "chat_ids": self.chat_ids,
            "chat_names": self.chat_names,
            "message_count": self.message_count,
            "backup_root": str(self.backup_root) if self.backup_root else None,
            "media_files": [str(path) for path in self.media_files],
            "media_directories": [str(path) for path in self.media_directories],
            "shared_media": [str(path) for path in self.shared_media],
            "missing_media": self.missing_media,
            "unsafe_media": self.unsafe_media,
            "exclusive_source_keys": self.exclusive_source_keys,
            "retained_source_keys": self.retained_source_keys,
            "retained_source_paths": self.retained_source_paths,
            "bytes_to_delete": self.bytes_to_delete,
        }


@dataclass(frozen=True)
class BackupDateDecision:
    chat_id: str
    timestamp: Optional[int]
    source: str
    confidence: str
    evidence: str
