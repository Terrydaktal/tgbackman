"""Pure message-record and export-range helpers."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from ..models import ExportStats, Target


def sender_label(sender: Any, sender_id: int | None, outgoing: bool) -> str:
    if outgoing:
        return "Me"
    if sender is None:
        return str(sender_id) if sender_id is not None else "Unknown"
    title = getattr(sender, "title", None)
    if title:
        return str(title)
    name = " ".join(str(value) for value in (getattr(sender, "first_name", None), getattr(sender, "last_name", None)) if value).strip()
    return name or getattr(sender, "username", None) or str(sender_id or "Unknown")


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def tl_object_envelope(value: Any, *, require_binary: bool = False) -> dict[str, Any] | None:
    """Serialize one Telethon object as presentation JSON plus exact TL bytes."""
    if value is None or not hasattr(value, "to_dict"):
        return None
    from telethon import __version__ as telethon_version
    from telethon.tl.alltlobjects import LAYER

    payload = json_safe(value)
    try:
        binary = bytes(value)
    except Exception as exc:
        if require_binary:
            raise ValueError(f"cannot serialize {type(value).__name__} as TL bytes") from exc
        binary = b""
    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest_payload = binary or canonical_json
    envelope: dict[str, Any] = {
        "type": type(value).__name__,
        "json": payload,
        "telethon_layer": int(LAYER),
        "telethon_version": str(telethon_version),
        "snapshot_sha256": hashlib.sha256(digest_payload).hexdigest(),
    }
    if binary:
        envelope.update(
            {
                "tl_encoding": "base64",
                "tl_data": base64.b64encode(binary).decode("ascii"),
                "tl_size": len(binary),
                "tl_sha256": hashlib.sha256(binary).hexdigest(),
            }
        )
    identity = telegram_entity_identity(value)
    if identity is not None:
        envelope["peer_kind"], envelope["peer_id"] = identity
    return envelope


def telegram_entity_identity(entity: Any) -> tuple[str, int] | None:
    """Return the stable peer identity for a full User, Chat, or Channel."""
    if entity is None or getattr(entity, "id", None) is None:
        return None
    type_name = type(entity).__name__.casefold()
    if "channel" in type_name:
        return "channel", abs(int(entity.id))
    if type_name == "chat" or type_name.endswith("forbidden") and "chat" in type_name:
        return "group", abs(int(entity.id))
    if "user" in type_name:
        return "user", abs(int(entity.id))
    return None


def telegram_peer_identity(peer: Any) -> tuple[str, int] | None:
    """Return the stable exporter identity for a Telethon peer object."""
    if peer is None:
        return None
    for attribute, kind in (
        ("user_id", "user"),
        ("chat_id", "group"),
        ("channel_id", "channel"),
    ):
        peer_id = getattr(peer, attribute, None)
        if peer_id is not None:
            return kind, abs(int(peer_id))
    return None


def reply_metadata(message: Any) -> dict[str, Any]:
    """Normalize reply references without copying the referenced message."""
    header = getattr(message, "reply_to", None)
    message_id = getattr(message, "reply_to_msg_id", None)
    if message_id is None and header is not None:
        message_id = getattr(header, "reply_to_msg_id", None)

    metadata: dict[str, Any] = {}
    if message_id is not None:
        metadata["reply_to_message_id"] = int(message_id)
    if header is None:
        return metadata

    peer = getattr(header, "reply_to_peer_id", None) or getattr(header, "peer", None)
    identity = telegram_peer_identity(peer)
    if identity is not None:
        metadata["reply_to_peer_kind"], metadata["reply_to_peer_id"] = identity

    scalar_fields = (
        ("reply_to_top_id", "reply_to_top_id", int),
        ("story_id", "reply_to_story_id", int),
        ("quote_text", "reply_quote_text", str),
        ("quote_offset", "reply_quote_offset", int),
    )
    for source_name, record_name, converter in scalar_fields:
        value = getattr(header, source_name, None)
        if value is not None:
            metadata[record_name] = converter(value)

    for source_name, record_name in (
        ("quote_entities", "reply_quote_entities"),
        ("reply_media", "reply_media"),
    ):
        value = getattr(header, source_name, None)
        if value is not None:
            metadata[record_name] = json_safe(value)
    return metadata


def _range_name(start: int | None, end: int | None, now: datetime) -> str:
    first = datetime.fromtimestamp(start, timezone.utc) if start is not None else now
    last = datetime.fromtimestamp(end, timezone.utc) if end is not None else now
    return f"{first.strftime('%Y-%m-%dT%H-%M-%SZ')}__{last.strftime('%Y-%m-%dT%H-%M-%SZ')}"


def range_dir_name(records: list[dict[str, Any]], now: datetime) -> str:
    dates = [int(record["date_unixtime"]) for record in records if record.get("date_unixtime") is not None and str(record["date_unixtime"]).isdigit()]
    return _range_name(min(dates) if dates else None, max(dates) if dates else None, now)


def range_dir_name_from_stats(stats: ExportStats, now: datetime) -> str:
    return _range_name(stats.first_message_unix, stats.last_message_unix, now)


def database_run_key(
    target: Target,
    baseline_id: int | None,
    baseline_unix: int | None,
    args: Any,
    *,
    effective_media: str | None = None,
    effective_max_file_size: int | None = None,
) -> str:
    identity = {
        "target_key": target.target_key, "chat_id": target.chat_id,
        "baseline_id": baseline_id, "baseline_unix": baseline_unix,
        "full_rescan": bool(args.full_rescan), "overlap_ids": args.overlap_ids,
        "overlap_days": args.overlap_days, "discard_overlap": bool(args.discard_overlap),
        "media": effective_media if effective_media is not None else args.media,
        "max_file_size": effective_max_file_size if effective_max_file_size is not None else args.max_file_size,
        # Schema 3 requires exact TL payloads plus an explicit completeness
        # ledger.  It must not resume older staged rows that cannot satisfy
        # those guarantees without asking Telegram for the message again.
        "staging_schema": 3,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
