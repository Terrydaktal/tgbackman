"""Pure message-record and export-range helpers."""

from __future__ import annotations

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


def _range_name(start: int | None, end: int | None, now: datetime) -> str:
    first = datetime.fromtimestamp(start, timezone.utc) if start is not None else now
    last = datetime.fromtimestamp(end, timezone.utc) if end is not None else now
    return f"{first.strftime('%Y-%m-%dT%H-%M-%SZ')}__{last.strftime('%Y-%m-%dT%H-%M-%SZ')}"


def range_dir_name(records: list[dict[str, Any]], now: datetime) -> str:
    dates = [int(record["date_unixtime"]) for record in records if record.get("date_unixtime") is not None and str(record["date_unixtime"]).isdigit()]
    return _range_name(min(dates) if dates else None, max(dates) if dates else None, now)


def range_dir_name_from_stats(stats: ExportStats, now: datetime) -> str:
    return _range_name(stats.first_message_unix, stats.last_message_unix, now)


def database_run_key(target: Target, baseline_id: int | None, baseline_unix: int | None, args: Any) -> str:
    identity = {
        "target_key": target.target_key, "chat_id": target.chat_id,
        "baseline_id": baseline_id, "baseline_unix": baseline_unix,
        "full_rescan": bool(args.full_rescan), "overlap_ids": args.overlap_ids,
        "overlap_days": args.overlap_days, "discard_overlap": bool(args.discard_overlap),
        "media": args.media, "max_file_size": args.max_file_size,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
