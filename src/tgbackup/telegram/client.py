"""Small Telethon adapter kept separate from backup/database orchestration."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from ..config import credentials, ensure_private_dir, secure_session_file
from ..errors import ExportError


def require_telethon() -> Any:
    try:
        import telethon  # type: ignore
    except ImportError as exc:
        raise ExportError("Telethon is not installed. Install the package's telethon dependency.") from exc
    return telethon


async def connect_client(args: argparse.Namespace) -> Any:
    require_telethon()
    from telethon import TelegramClient
    api_id, api_hash = credentials(Path(args.config).expanduser())
    session_path = Path(args.session).expanduser()
    ensure_private_dir(session_path.parent)
    client = TelegramClient(str(session_path), api_id, api_hash)
    await client.connect()
    if not await client.is_user_authorized():
        phone = input("Telegram phone number (international format): ").strip()
        await client.start(phone=phone)
    secure_session_file(session_path)
    return client


def target_input_peer(target: Any) -> Any:
    from telethon.tl.types import InputPeerChannel, InputPeerChat, InputPeerUser
    if target.peer_kind == "user":
        if target.access_hash is None:
            raise ExportError(f"Target {target.source_name!r} has no stored user access hash; remap it")
        return InputPeerUser(target.peer_id, target.access_hash)
    if target.peer_kind == "group":
        return InputPeerChat(target.peer_id)
    if target.peer_kind == "channel":
        if target.access_hash is None:
            raise ExportError(f"Target {target.source_name!r} has no stored channel access hash; remap it")
        return InputPeerChannel(target.peer_id, target.access_hash)
    raise ExportError(f"Unknown target peer kind: {target.peer_kind}")


async def resolve_peer(client: Any, value: str) -> Any:
    """Resolve usernames/links directly and numeric IDs through dialogs."""
    value = value.strip()
    if not re.fullmatch(r"-?\d+", value):
        return await client.get_entity(value)
    numeric = int(value)
    wanted = int(value[4:]) if value.startswith("-100") and len(value) > 4 else abs(numeric)
    for dialog in await client.get_dialogs(limit=None):
        entity = dialog.entity
        if getattr(entity, "id", None) == wanted:
            return entity
    raise ExportError(f"Telegram numeric peer {value} was not found in your dialogs; use a username/link or run `dialogs` to verify the account.")
