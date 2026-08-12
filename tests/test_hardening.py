"""Regression tests for the hardening paths added after the initial exporter tests."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import tgbackup.exporter as exporter
from tgbackup.backup.media import download_media
from tgbackup.backup.records import tl_object_envelope
from tgbackup.backup.targets import target_output_dir
from tgbackup.database.importer import (
    _iter_multi_chat_json,
    parse_json_file_archival,
    resolve_local_media_path,
)
from tgbackup.db.schema import setup_database
from tgbackup.db.sources import archive_source_file
from tgbackup.media_dedupe import deduplicate
from tgbackup.models import MediaDownloadPlan


class _MediaMessage:
    def __init__(self, payload: bytes = b"media", *, flood_wait: int | None = None):
        self.id = 42
        self.payload = payload
        self.flood_wait = flood_wait
        self.calls = 0

    async def download_media(self, *, file: str, progress_callback=None, **_kwargs):
        self.calls += 1
        if self.flood_wait is not None and self.calls == 1:
            error_type = type("FloodWaitError", (Exception,), {})
            error = error_type()
            error.seconds = self.flood_wait
            raise error
        if progress_callback:
            progress_callback(len(self.payload), len(self.payload))
        Path(file).write_bytes(self.payload)
        return file


def _plan(expected_size: int | None = None) -> MediaDownloadPlan:
    return MediaDownloadPlan(
        primary=object(),
        media_type="file",
        filename="42_payload.bin",
        expected_size=expected_size,
    )


class MediaHardeningTests(unittest.TestCase):
    def test_reuse_requires_a_matching_hash_sidecar(self):
        async def run() -> None:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                message = _MediaMessage(b"media")
                first = await download_media(
                    message,
                    root / "media",
                    "file",
                    0,
                    expected_size=5,
                    plan=_plan(5),
                )
                self.assertEqual(message.calls, 1)
                self.assertIsNotNone(first)
                second = await download_media(
                    message,
                    root / "media",
                    "file",
                    0,
                    expected_size=5,
                    plan=_plan(5),
                )
                self.assertEqual(message.calls, 1)
                self.assertEqual(second, first)

                sidecar = root / "media" / "file" / "42_payload.bin.sha256"
                sidecar.write_text("0" * 64 + "\n", encoding="ascii")
                await download_media(
                    message,
                    root / "media",
                    "file",
                    0,
                    expected_size=5,
                    plan=_plan(5),
                )
                self.assertEqual(message.calls, 2)

        asyncio.run(run())

    def test_flood_wait_delay_is_respected_before_retry(self):
        async def run() -> None:
            with tempfile.TemporaryDirectory() as directory:
                message = _MediaMessage(b"ok", flood_wait=17)
                waits: list[float] = []

                async def fake_sleep(seconds: float) -> None:
                    waits.append(seconds)

                with mock.patch("tgbackup.backup.media.asyncio.sleep", fake_sleep):
                    result = await download_media(
                        message,
                        Path(directory) / "media",
                        "file",
                        0,
                        expected_size=2,
                        plan=_plan(2),
                    )
                self.assertIsNotNone(result)
                self.assertEqual(message.calls, 2)
                self.assertEqual(waits, [17])

        asyncio.run(run())

    def test_max_size_rejects_unknown_size_downloads(self):
        async def run() -> None:
            with tempfile.TemporaryDirectory() as directory:
                message = _MediaMessage(b"too large")
                with self.assertRaises(exporter.ExportError):
                    await download_media(
                        message,
                        Path(directory) / "media",
                        "file",
                        0,
                        expected_size=None,
                        plan=_plan(None),
                        max_file_size=3,
                    )
                self.assertFalse(list((Path(directory) / "media").rglob("*.part-*")))

        asyncio.run(run())


class DatabaseHardeningTests(unittest.TestCase):
    def test_fts_triggers_repair_insert_update_and_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            conn = setup_database(Path(directory) / "archive.db")
            conn.execute("INSERT INTO chats(chat_id, chat_name) VALUES ('chat', 'Chat')")
            conn.execute("INSERT INTO messages(message_id, chat_id, text) VALUES (1, 'chat', 'alpha')")
            self.assertEqual(conn.execute("SELECT count(*) FROM messages_fts WHERE messages_fts MATCH 'alpha'").fetchone()[0], 1)
            conn.execute("UPDATE messages SET text='beta' WHERE message_id=1")
            self.assertEqual(conn.execute("SELECT count(*) FROM messages_fts WHERE messages_fts MATCH 'alpha'").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT count(*) FROM messages_fts WHERE messages_fts MATCH 'beta'").fetchone()[0], 1)
            conn.execute("DELETE FROM messages WHERE message_id=1")
            self.assertEqual(conn.execute("SELECT count(*) FROM messages_fts WHERE messages_fts MATCH 'beta'").fetchone()[0], 0)
            conn.close()

    def test_source_payload_is_repaired_even_when_size_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "export.json"
            source.write_bytes(b"aaaa")
            conn = setup_database(root / "archive.db")
            key = archive_source_file(conn, str(source), "json", archive_payload=True)
            conn.commit()
            conn.execute("UPDATE backup_sources SET payload=? WHERE source_key=?", (sqlite3.Binary(b"corrupt"), key))
            conn.commit()
            self.assertEqual(archive_source_file(conn, str(source), "json", archive_payload=True), key)
            conn.commit()
            payload = conn.execute("SELECT compression, payload FROM backup_sources WHERE source_key=?", (key,)).fetchone()
            self.assertEqual(payload[0], "zlib")
            self.assertEqual(zlib.decompress(payload[1]), b"aaaa")
            conn.close()

    def test_media_paths_cannot_escape_declared_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "export"
            root.mkdir()
            self.assertEqual(resolve_local_media_path(str(root), "media/x.bin"), str(root / "media/x.bin"))
            with self.assertRaises(ValueError):
                resolve_local_media_path(str(root), "../outside.bin")
            outside = Path(directory) / "outside.bin"
            outside.write_bytes(b"secret")
            link = root / "link"
            try:
                link.symlink_to(outside)
            except OSError:
                self.skipTest("symbolic links are unavailable")
            with self.assertRaises(ValueError):
                resolve_local_media_path(str(root), "link")

    def test_multi_chat_json_streams_headers_and_individual_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "result.json"
            source.write_text(json.dumps({
                "chats": {"list": [
                    {"id": 1, "name": "Empty", "messages": []},
                    {"id": 2, "name": "Chat", "messages": [
                        {"id": 10, "text": [{"type": "plain", "text": "hello"}]},
                        {"id": 11, "text": "world"},
                    ]},
                ]}
            }), encoding="utf-8")
            events = list(_iter_multi_chat_json(str(source)))
            self.assertEqual([(header.get("id"), message and message.get("id")) for header, message in events], [(1, None), (2, None), (2, 10), (2, 11)])
            conn = setup_database(root / "archive.db")
            key = archive_source_file(conn, str(source), "json", archive_payload=False)
            imported, chats = parse_json_file_archival(str(source), conn, root_path=str(root), source_key=key)
            self.assertEqual(imported, 2)
            self.assertEqual(chats, {"1", "2"})
            self.assertEqual(conn.execute("SELECT count(*) FROM chats").fetchone()[0], 2)
            self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 2)
            conn.close()

    def test_full_rescan_tombstones_absent_messages_and_updates_watermark(self):
        async def run() -> None:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                db = root / "archive.db"
                conn = setup_database(db)
                target = exporter.Target(
                    target_key="target", source_name="Chat", chat_id="chat", peer_kind="user", peer_id=1,
                    access_hash=None, title="Chat", username=None, enabled=True, output_dir=None,
                    last_message_id=2, last_message_unix=200, last_export_unix=None,
                )
                conn.execute("INSERT INTO chats(chat_id, chat_name) VALUES ('chat', 'Chat')")
                conn.execute("INSERT INTO messages(message_id, chat_id, timestamp_unix, text) VALUES (1, 'chat', 100, 'gone')")
                conn.execute("INSERT INTO messages(message_id, chat_id, timestamp_unix, text) VALUES (2, 'chat', 200, 'kept')")
                conn.execute("""INSERT INTO telegram_backup_targets
                    (target_key, source_name, chat_id, peer_kind, peer_id, title, enabled, created_unix, updated_unix, last_message_id, last_message_unix)
                    VALUES ('target', 'Chat', 'chat', 'user', 1, 'Chat', 1, 1, 1, 2, 200)""")
                conn.commit()

                async def records():
                    yield {"id": 2, "date_unixtime": "200", "text": "kept"}, None

                stats = await exporter.write_database_stream(conn, target, records(), root / "chat", "full-run", 2, 200, True)
                self.assertEqual(stats.message_count, 1)
                self.assertEqual(conn.execute("SELECT is_deleted FROM messages WHERE message_id=1").fetchone()[0], 1)
                self.assertEqual(conn.execute("SELECT is_deleted FROM messages WHERE message_id=2").fetchone()[0], 0)
                self.assertEqual(conn.execute("SELECT last_message_id FROM telegram_backup_targets").fetchone()[0], 2)
                self.assertEqual(conn.execute("SELECT count(*) FROM telegram_backup_run_records WHERE run_key='full-run'").fetchone()[0], 1)
                self.assertEqual(conn.execute("SELECT status FROM telegram_backup_run_attempts WHERE run_key='full-run'").fetchone()[0], "completed")
                conn.close()

        asyncio.run(run())

    def test_empty_full_rescan_is_authoritative(self):
        async def run() -> None:
            from telethon.tl import types

            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                conn = setup_database(root / "archive.db")
                target = exporter.Target(
                    target_key="target", source_name="Chat", chat_id="chat", peer_kind="user", peer_id=1,
                    access_hash=None, title="Chat", username=None, enabled=True, output_dir=None,
                    last_message_id=99, last_message_unix=999, last_export_unix=None,
                )
                conn.execute("INSERT INTO chats(chat_id, chat_name) VALUES ('chat', 'Chat')")
                conn.execute("INSERT INTO messages(message_id, chat_id, timestamp_unix, text) VALUES (99, 'chat', 999, 'removed')")
                conn.execute("""INSERT INTO telegram_backup_targets
                    (target_key, source_name, chat_id, peer_kind, peer_id, title, enabled, created_unix, updated_unix, last_message_id, last_message_unix)
                    VALUES ('target', 'Chat', 'chat', 'user', 1, 'Chat', 1, 1, 1, 99, 999)""")
                conn.commit()

                async def no_records():
                    if False:
                        yield {}, None

                stats = await exporter.write_database_stream(
                    conn,
                    target,
                    no_records(),
                    root / "chat",
                    "empty-full",
                    99,
                    999,
                    True,
                    chat_entity_snapshot=tl_object_envelope(
                        types.User(id=1, access_hash=2, first_name="Chat"),
                        require_binary=True,
                    ),
                    chat_full_snapshot=tl_object_envelope(
                        types.users.UserFull(
                            full_user=types.UserFull(
                                id=1,
                                settings=types.PeerSettings(),
                                notify_settings=types.PeerNotifySettings(),
                                common_chats_count=0,
                            ),
                            chats=[],
                            users=[types.User(id=1, access_hash=2, first_name="Chat")],
                        ),
                        require_binary=True,
                    ),
                )
                self.assertEqual(stats.message_count, 0)
                self.assertEqual(conn.execute("SELECT is_deleted FROM messages WHERE message_id=99").fetchone()[0], 1)
                self.assertIsNone(conn.execute("SELECT last_message_id FROM telegram_backup_targets").fetchone()[0])
                conn.close()

        asyncio.run(run())


class SafetyAndMaintenanceTests(unittest.TestCase):
    def test_target_marker_collision_gets_a_peer_specific_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            occupied = root / "Chat"
            occupied.mkdir()
            (occupied / ".tgbackman_target.json").write_text(json.dumps({"chat_id": "other"}), encoding="utf-8")
            target = exporter.Target(
                target_key="target", source_name="Chat", chat_id="chat_1", peer_kind="user", peer_id=1,
                access_hash=None, title="Chat", username=None, enabled=True, output_dir=None,
                last_message_id=None, last_message_unix=None, last_export_unix=None,
            )
            selected = target_output_dir(root, target)
            self.assertEqual(selected, root / "Chat__chat_1")
            self.assertEqual(json.loads((selected / ".tgbackman_target.json").read_text())["chat_id"], "chat_1")
            self.assertEqual(json.loads((occupied / ".tgbackman_target.json").read_text())["chat_id"], "other")

    def test_dedupe_report_is_read_only_and_counts_duplicate_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.bin").write_bytes(b"same")
            (root / "b.bin").write_bytes(b"same")
            (root / "b.bin.sha256").write_text("ignored", encoding="ascii")
            self.assertEqual(deduplicate(root, apply=False), (2, 1, 0))
            self.assertEqual((root / "a.bin").read_bytes(), (root / "b.bin").read_bytes())

    def test_purge_recovery_rejects_symlinked_manifest_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "media"
            media.mkdir()
            outside = Path(directory) / "outside"
            outside.mkdir()
            try:
                link = media / "link"
                link.symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("symbolic links are unavailable")
            with self.assertRaises(exporter.ExportError):
                exporter.delete_manifest_media({"media_files": [str(link / "x.bin")]}, root)


if __name__ == "__main__":
    unittest.main()
