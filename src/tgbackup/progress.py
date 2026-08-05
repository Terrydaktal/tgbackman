"""Terminal progress reporting and single-run locking."""

from __future__ import annotations

import contextlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - Linux is the supported deployment.
    fcntl = None  # type: ignore[assignment]

from .errors import ExportError


def human_bytes(value: int) -> str:
    amount = float(max(0, value))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TiB"


def human_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


class ProgressReporter:
    """Periodic line-oriented progress suitable for terminals and journals."""

    def __init__(self, target_name: str, *, interval: float = 5.0, every: int = 100,
                 enabled: bool = True, output: Callable[[str], None] = print,
                 chat_position: Optional[int] = None, chat_total: Optional[int] = None) -> None:
        self.target_name = target_name
        self.interval = max(0.1, interval)
        self.every = max(1, every)
        self.enabled = enabled
        self.output = output
        self.chat_position = chat_position
        self.chat_total = chat_total
        self.started = time.monotonic()
        self.last_report = self.started
        self.processed = 0
        self.latest_message_id: Optional[int] = None
        self.latest_message_unix: Optional[int] = None
        self.media_seen = 0
        self.media_ready = 0
        self.media_skipped = 0
        self.media_errors = 0
        self.media_bytes = 0
        self.reused_media = 0
        self.current_media_started = self.started
        self.current_media_id: Optional[int] = None
        self.current_media_name = ""

    def emit(self, text: str) -> None:
        if not self.enabled:
            return
        position = (f"chat {self.chat_position}/{self.chat_total}; "
                    if self.chat_position is not None and self.chat_total is not None else "")
        line = f"[{self.target_name}] progress: {position}{text}"
        if self.output is print:
            print(line, flush=True)
        else:
            self.output(line)

    def start(self, mode: str, resumed_messages: int = 0) -> None:
        resume = f"; resuming with {resumed_messages:,} staged message(s)" if resumed_messages else ""
        self.emit(f"started {mode}{resume}")

    def phase(self, text: str) -> None:
        self.emit(f"{text}; elapsed {human_duration(time.monotonic() - self.started)}")

    def note_reused_media(self) -> None:
        self.reused_media += 1

    def media_download_progress(self, message_id: int, filename: str, received: int, total: int) -> None:
        now = time.monotonic()
        if self.current_media_id != message_id:
            self.current_media_id = message_id
            self.current_media_name = filename
            self.current_media_started = now
        if now - self.last_report < self.interval and received < total:
            return
        percent = (received * 100 / total) if total else 0.0
        elapsed = max(0.001, now - self.current_media_started)
        speed = int(received / elapsed)
        total_text = human_bytes(total) if total else "unknown"
        self.emit(f"media message {message_id} {filename}: {human_bytes(received)}/{total_text} "
                  f"({percent:.1f}%) at {human_bytes(speed)}/s; total elapsed "
                  f"{human_duration(now - self.started)}")
        self.last_report = now

    def observe(self, record: dict[str, Any], error: Optional[str]) -> None:
        self.processed += 1
        if record.get("id") is not None:
            self.latest_message_id = int(record["id"])
        timestamp = record.get("date_unixtime")
        if timestamp is not None and str(timestamp).lstrip("-").isdigit():
            self.latest_message_unix = int(timestamp)
        if record.get("media_type"):
            self.media_seen += 1
            if record.get("file"):
                self.media_ready += 1
                self.media_bytes += int(record.get("media_size") or 0)
            elif record.get("media_skipped"):
                self.media_skipped += 1
            elif record.get("media_error") or error:
                self.media_errors += 1
        now = time.monotonic()
        if self.processed == 1 or self.processed % self.every == 0 or now - self.last_report >= self.interval:
            self.report(now)

    def report(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        elapsed = max(0.001, now - self.started)
        latest_date = (datetime.fromtimestamp(self.latest_message_unix, timezone.utc).isoformat()
                       if self.latest_message_unix is not None else "unknown")
        self.emit(f"{self.processed:,} processed; latest id={self.latest_message_id or 'unknown'} "
                  f"date={latest_date}; {self.processed / elapsed:.1f} msg/s; media seen={self.media_seen:,} "
                  f"ready={self.media_ready:,} reused={self.reused_media:,} skipped={self.media_skipped:,} "
                  f"errors={self.media_errors:,} bytes={human_bytes(self.media_bytes)}; "
                  f"elapsed {human_duration(elapsed)}")
        self.last_report = now

    def finish(self, outcome: str) -> None:
        if self.processed:
            self.report()
        self.phase(outcome)

    def fail(self, exc: BaseException) -> None:
        self.phase(f"failed after {self.processed:,} processed: {exc}")


class ExportLock:
    """Non-blocking process lock preventing duplicate runs for one output root."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Optional[Any] = None

    def acquire(self) -> None:
        if fcntl is None:
            raise ExportError("process locking is unavailable on this platform")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            self.path.chmod(0o600)
        except OSError:
            pass
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise ExportError(f"another tgbackman export is already running for {self.path.parent}") from exc
        self.handle = handle

    def release(self) -> None:
        if self.handle is None:
            return
        with contextlib.suppress(OSError):
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


__all__ = ["human_bytes", "human_duration", "ProgressReporter", "ExportLock"]
