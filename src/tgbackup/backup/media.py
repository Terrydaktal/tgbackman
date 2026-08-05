"""Telegram media selection, download, integrity checks, and reuse."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import sys
from pathlib import Path
from typing import Any, Optional

from ..config import safe_component
from ..errors import ExportError
from ..models import MediaDownloadPlan
from ..progress import ProgressReporter


def _legacy_media_type_for(message: Any) -> Optional[str]:
    for attribute, media_type in (
        ("photo", "photo"), ("voice", "voice_message"), ("video", "video"),
        ("video_note", "video"), ("audio", "audio_file"), ("sticker", "sticker"),
        ("gif", "animation"), ("document", "file"),
    ):
        if getattr(message, attribute, None) is not None:
            return media_type
    return None


def _document_media_type(document: Any) -> str:
    from telethon.tl import types
    attributes = list(getattr(document, "attributes", None) or [])
    audio = next((item for item in attributes if isinstance(item, types.DocumentAttributeAudio)), None)
    video = next((item for item in attributes if isinstance(item, types.DocumentAttributeVideo)), None)
    if audio is not None and bool(getattr(audio, "voice", False)):
        return "voice_message"
    if video is not None and bool(getattr(video, "round_message", False)):
        return "video"
    if audio is not None:
        return "audio_file"
    if any(isinstance(item, types.DocumentAttributeSticker) for item in attributes):
        return "sticker"
    if any(isinstance(item, types.DocumentAttributeAnimated) for item in attributes):
        return "animation"
    return "video" if video is not None else "file"


def _photo_size_sort_key(size: Any) -> tuple[int, int]:
    from telethon.tl import types
    if isinstance(size, (types.PhotoStrippedSize, types.PhotoCachedSize)):
        return 1, len(size.bytes)
    if isinstance(size, types.PhotoSize):
        return 1, int(size.size)
    if isinstance(size, types.PhotoSizeProgressive):
        return 1, max((int(item) for item in size.sizes), default=0)
    if isinstance(size, types.VideoSize):
        return 2, int(size.size)
    return 0, 0


def _photo_size_expected_size(size: Any) -> Optional[int]:
    from telethon import utils as telethon_utils
    from telethon.tl import types
    if isinstance(size, types.PhotoStrippedSize):
        return len(telethon_utils.stripped_photo_to_jpg(size.bytes))
    if isinstance(size, types.PhotoCachedSize):
        return len(size.bytes)
    if isinstance(size, types.PhotoSize):
        return int(size.size)
    if isinstance(size, types.PhotoSizeProgressive):
        return max((int(item) for item in size.sizes), default=0) or None
    if isinstance(size, types.VideoSize):
        return int(size.size)
    return None


def _filename_from_file(message_id: int, file_obj: Any, fallback_ext: str) -> str:
    name = getattr(file_obj, "name", None) if file_obj else None
    ext = getattr(file_obj, "ext", None) if file_obj else None
    if not name:
        name = f"{message_id}{ext or fallback_ext}"
    name = safe_component(str(name), f"{message_id}{ext or fallback_ext}")
    if not Path(name).suffix and ext:
        name += str(ext)
    elif not Path(name).suffix and fallback_ext:
        name += fallback_ext
    return f"{message_id}_{name}"


def media_download_plan(message: Any) -> Optional[MediaDownloadPlan]:
    """Select the same primary downloadable object Telethon will retrieve."""
    from telethon.tl import types
    from telethon.tl.custom.file import File
    message_id = int(message.id)
    media = getattr(message, "media", None)
    primary: Any = None
    if isinstance(media, types.MessageMediaWebPage):
        webpage = getattr(media, "webpage", None)
        if isinstance(webpage, types.WebPage):
            primary = webpage.document or webpage.photo
    elif isinstance(media, types.MessageMediaDocument) and isinstance(media.document, types.Document):
        primary = media.document
    elif isinstance(media, types.MessageMediaPhoto) and isinstance(media.photo, types.Photo):
        primary = media.photo
    action = getattr(message, "action", None)
    if primary is None and isinstance(action, types.MessageActionChatEditPhoto) and isinstance(action.photo, types.Photo):
        primary = action.photo
    if isinstance(primary, types.Document):
        file_obj = File(primary)
        return MediaDownloadPlan(primary=primary, media_type=_document_media_type(primary),
                                 filename=_filename_from_file(message_id, file_obj, file_obj.ext or ""),
                                 expected_size=int(primary.size) if primary.size is not None else None)
    if isinstance(primary, types.Photo):
        sizes = [item for item in list(primary.sizes or []) + list(primary.video_sizes or [])
                 if not isinstance(item, types.PhotoPathSize)]
        if not sizes:
            return None
        selected = sorted(sizes, key=_photo_size_sort_key)[-1]
        if isinstance(selected, types.PhotoSizeEmpty):
            return None
        is_video = isinstance(selected, types.VideoSize)
        extension = ".mp4" if is_video else ".jpg"
        return MediaDownloadPlan(primary=primary, media_type="video" if is_video else "photo",
                                 filename=f"{message_id}_{message_id}{extension}",
                                 expected_size=_photo_size_expected_size(selected),
                                 thumb_type=str(selected.type))
    media_type = _legacy_media_type_for(message)
    if media_type is None:
        return None
    file_obj = getattr(message, "file", None)
    expected_size = getattr(file_obj, "size", None) if file_obj else None
    return MediaDownloadPlan(primary=getattr(message, "media", None), media_type=media_type,
                             filename=_filename_from_file(message_id, file_obj, getattr(file_obj, "ext", None) or ""),
                             expected_size=int(expected_size) if expected_size is not None else None)


def media_type_for(message: Any) -> Optional[str]:
    plan = media_download_plan(message)
    return plan.media_type if plan else None


def media_filename(message: Any, plan: Optional[MediaDownloadPlan] = None) -> str:
    selected = plan or media_download_plan(message)
    return selected.filename if selected is not None else f"{int(message.id)}_{int(message.id)}"


def flood_wait_seconds(exc: BaseException) -> Optional[int]:
    if exc.__class__.__name__ != "FloodWaitError":
        return None
    try:
        return max(1, int(getattr(exc, "seconds", None)))
    except (TypeError, ValueError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def download_media(message: Any, media_root: Path, media_type: str, retries: int,
                         expected_size: Optional[int] = None, progress: Optional[ProgressReporter] = None,
                         plan: Optional[MediaDownloadPlan] = None) -> Optional[str]:
    selected_plan = plan or media_download_plan(message)
    if selected_plan is not None:
        media_type, expected_size = selected_plan.media_type, selected_plan.expected_size
    media_dir = media_root / media_type
    media_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        media_dir.chmod(0o700)
    filename = media_filename(message, selected_plan)
    destination = media_dir / filename
    if destination.exists() and destination.is_file() and destination.stat().st_size > 0:
        if expected_size is None or destination.stat().st_size == expected_size:
            with contextlib.suppress(OSError):
                destination.chmod(0o600)
            if progress:
                progress.note_reused_media()
            return destination.relative_to(media_root.parent).as_posix()
        destination.unlink()
    last_error: Optional[Exception] = None
    flood_waits = 0
    attempt = 0
    while attempt <= max(0, retries):
        try:
            if destination.exists():
                destination.unlink()
            reported_total = 0
            kwargs: dict[str, Any] = {"file": str(destination)}
            if selected_plan is not None and selected_plan.thumb_type is not None:
                kwargs["thumb"] = selected_plan.thumb_type
            if progress:
                def report_download(received: int, total: int) -> None:
                    nonlocal reported_total
                    reported_total = max(reported_total, int(total or 0))
                    progress.media_download_progress(int(message.id), filename, int(received), int(total or 0))
                kwargs["progress_callback"] = report_download
            downloaded = await message.download_media(**kwargs)
            if downloaded and destination.is_file() and destination.stat().st_size > 0:
                actual_size = destination.stat().st_size
                if expected_size is not None and reported_total and expected_size != reported_total:
                    last_error = ExportError(f"media for message {message.id} has inconsistent Telegram sizes ({expected_size} metadata, {reported_total} transfer)")
                elif expected_size is not None and actual_size != expected_size:
                    last_error = ExportError(f"media for message {message.id} has size {actual_size}, expected {expected_size}")
                elif expected_size is None and reported_total and actual_size != reported_total:
                    last_error = ExportError(f"media for message {message.id} has size {actual_size}, expected transfer size {reported_total}")
                else:
                    with contextlib.suppress(OSError):
                        destination.chmod(0o600)
                    return destination.relative_to(media_root.parent).as_posix()
            elif downloaded:
                last_error = ExportError(f"media for message {message.id} was reported downloaded but is missing")
        except Exception as exc:
            wait_seconds = flood_wait_seconds(exc)
            if wait_seconds is not None:
                flood_waits += 1
                if flood_waits > 5:
                    raise
                print(f"Telegram requested a {wait_seconds}s media wait; sleeping before retry", file=sys.stderr)
                await asyncio.sleep(wait_seconds)
                continue
            last_error = exc
        attempt += 1
        if attempt <= retries:
            await asyncio.sleep(min(30, 2**attempt))
    if last_error:
        raise last_error
    return None
