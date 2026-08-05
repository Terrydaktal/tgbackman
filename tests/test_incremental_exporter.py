import asyncio
import hashlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import db_indexer
import telegram_incremental_backup as exporter


class FakeMessage:
    def __init__(self, message_id: int, *, text: str | None = None, fail_media: bool = False):
        self.id = message_id
        self.date = datetime.fromtimestamp(message_id, timezone.utc)
        self.sender = None
        self.sender_id = None
        self.out = False
        self.raw_text = text or f"message {message_id}"
        self.action = None
        self.photo = None
        self.voice = None
        self.video = None
        self.video_note = None
        self.audio = None
        self.sticker = None
        self.gif = None
        self.document = None
        self.file = None
        self.reply_to_msg_id = None
        self.forward = None
        self.edit_date = None
        self.entities = None
        self.reactions = None
        self.reply_markup = None
        self.media = None
        self.grouped_id = None
        self.fail_media = fail_media

    async def download_media(self, file: str):
        if self.fail_media:
            raise RuntimeError("test media failure")
        Path(file).write_bytes(b"media")
        return file


class FakeClient:
    def __init__(self, messages):
        self.messages = messages

    def iter_messages(self, entity, **kwargs):
        async def generator():
            for message in self.messages:
                yield message

        return generator()


class ExportClient(FakeClient):
    async def get_input_entity(self, peer):
        return object()

    async def disconnect(self):
        return None


def make_target() -> exporter.Target:
    return exporter.Target(
        target_key="chat-key",
        source_name="Chat",
        chat_id="chat_1",
        peer_kind="user",
        peer_id=1,
        access_hash=2,
        title="Chat",
        username=None,
        enabled=True,
        output_dir=None,
        last_message_id=None,
        last_message_unix=None,
        last_export_unix=None,
    )


class IncrementalExporterTests(unittest.TestCase):
    def test_cli_has_positive_index_and_exact_date_default(self):
        args = exporter.build_parser().parse_args(["run", "--index"])
        self.assertTrue(args.index)
        self.assertFalse(args.run_all)
        self.assertEqual(args.overlap_ids, 0)
        self.assertEqual(args.overlap_days, 0)
        self.assertEqual(args.progress_interval, 5.0)
        self.assertEqual(args.progress_every, 100)
        self.assertFalse(args.no_progress)

        all_args = exporter.build_parser().parse_args(["run", "--all"])
        self.assertTrue(all_args.run_all)

    def test_progress_reporter_includes_counts_media_rate_and_outcome(self):
        lines = []
        reporter = exporter.ProgressReporter(
            "Alexandra", interval=60, every=2, output=lines.append
        )
        reporter.start("direct-to-database backup", resumed_messages=4500)
        reporter.observe(
            {
                "id": 10,
                "date_unixtime": "100",
                "media_type": "photo",
                "file": "media/photo/10.jpg",
                "media_size": 1024,
            },
            None,
        )
        reporter.observe(
            {"id": 11, "date_unixtime": "101", "media_type": "file", "media_skipped": "policy"},
            None,
        )
        reporter.finish("commit complete; watermark advanced to message 11")
        output = "\n".join(lines)
        self.assertIn("resuming with 4,500 staged", output)
        self.assertIn("2 processed", output)
        self.assertIn("latest id=11", output)
        self.assertIn("media seen=2", output)
        self.assertIn("ready=1", output)
        self.assertIn("skipped=1", output)
        self.assertIn("1.0 KiB", output)
        self.assertIn("watermark advanced", output)

    def test_media_download_progress_callback_reports_file_completion(self):
        class ProgressMessage(FakeMessage):
            async def download_media(self, file: str, progress_callback=None):
                if progress_callback:
                    progress_callback(2, 5)
                Path(file).write_bytes(b"media")
                if progress_callback:
                    progress_callback(5, 5)
                return file

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                lines = []
                reporter = exporter.ProgressReporter(
                    "Chat", interval=60, every=100, output=lines.append
                )
                result = await exporter.download_media(
                    ProgressMessage(12), Path(directory) / "media", "file", 0, 5, reporter
                )
                self.assertIsNotNone(result)
                output = "\n".join(lines)
                self.assertIn("media message 12", output)
                self.assertIn("100.0%", output)
                self.assertIn("5 B/5 B", output)

        asyncio.run(run())

    def test_session_file_name_is_telethon_name(self):
        self.assertEqual(exporter.telethon_session_file(Path("/tmp/session")), Path("/tmp/session.session"))
        self.assertEqual(exporter.telethon_session_file(Path("/tmp/session.session")), Path("/tmp/session.session"))

    def test_flood_wait_seconds_uses_telegram_delay(self):
        FloodWaitError = type("FloodWaitError", (Exception,), {})
        error = FloodWaitError()
        error.seconds = 123
        self.assertEqual(exporter.flood_wait_seconds(error), 123)
        self.assertIsNone(exporter.flood_wait_seconds(RuntimeError("not a Telegram wait")))

    def test_export_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".lock"
            first = exporter.ExportLock(path)
            second = exporter.ExportLock(path)
            first.acquire()
            try:
                with self.assertRaises(exporter.ExportError):
                    second.acquire()
            finally:
                first.release()

    def test_allowed_media_error_is_verifiable(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                message = FakeMessage(1, fail_media=True)
                message.document = object()
                message.file = type("File", (), {"size": 5, "name": "file.bin", "ext": ".bin"})()
                record, error = await exporter.message_record(
                    message,
                    Path(directory) / "media",
                    True,
                    set(exporter.MEDIA_TYPES),
                    0,
                    True,
                    0,
                )
                self.assertIn("media_error", record)
                self.assertNotIn("file", record)
                self.assertIsNotNone(error)

        asyncio.run(run())

    def test_streamed_export_has_range_and_metadata(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "output"
                target = make_target()
                staging = root / "Chat" / ".partial-chat-key-none"
                staging.mkdir(parents=True)

                async def records():
                    yield {"id": 1, "date_unixtime": "100", "text": "one"}, None
                    yield {"id": 2, "date_unixtime": "200", "text": "two"}, None

                final, stats = await exporter.write_export_stream(target, records(), root, staging)
                self.assertEqual(stats.message_count, 2)
                self.assertIsNotNone(final)
                payload = json.loads((final / "result.json").read_text(encoding="utf-8"))
                self.assertEqual([row["id"] for row in payload["chats"]["list"][0]["messages"]], [1, 2])
                self.assertTrue((final / ".backman_export_meta.json").is_file())

        asyncio.run(run())

    def test_direct_chat_output_adds_only_date_range(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                chat_dir = Path(directory) / "alex2"
                chat_dir.mkdir()
                target = make_target()
                target.chat_id = "alex2"
                staging = chat_dir / ".partial-chat-key-none"
                staging.mkdir()

                async def records():
                    yield {"id": 3, "date_unixtime": "300", "text": "three"}, None

                selected = exporter.direct_target_output_dir(chat_dir, target, write_marker=True)
                final, stats = await exporter.write_export_stream(
                    target,
                    records(),
                    chat_dir,
                    staging,
                    target_dir_override=selected,
                )
                self.assertEqual(stats.message_count, 1)
                self.assertEqual(final.parent, chat_dir)
                self.assertTrue((final / "result.json").is_file())
                self.assertFalse((chat_dir / target.title).exists())

        asyncio.run(run())

    def test_download_rejects_truncated_media(self):
        class WrongSizeMessage(FakeMessage):
            async def download_media(self, file: str):
                Path(file).write_bytes(b"bad")
                return file

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                message = WrongSizeMessage(1)
                message.file = type("File", (), {"size": 5, "name": "file.bin", "ext": ".bin"})()
                with self.assertRaises(exporter.ExportError):
                    await exporter.download_media(message, Path(directory) / "media", "file", 0, 5)

        asyncio.run(run())

    def test_web_preview_document_uses_document_type_name_size_and_hash(self):
        from telethon.tl import types

        class CompoundPreviewMessage(FakeMessage):
            async def download_media(self, file: str, thumb=None, progress_callback=None):
                self.asserted_thumb = thumb
                payload = b"mp4data"
                Path(file).write_bytes(payload)
                if progress_callback:
                    progress_callback(len(payload), len(payload))
                return file

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                document = types.Document(
                    id=10,
                    access_hash=11,
                    file_reference=b"",
                    date=datetime.now(timezone.utc),
                    mime_type="video/mp4",
                    size=7,
                    dc_id=1,
                    attributes=[
                        types.DocumentAttributeVideo(duration=1, w=320, h=240),
                        types.DocumentAttributeFilename("preview.mp4"),
                    ],
                )
                photo = types.Photo(
                    id=20,
                    access_hash=21,
                    file_reference=b"",
                    date=datetime.now(timezone.utc),
                    sizes=[types.PhotoSize(type="x", w=10, h=10, size=3)],
                    dc_id=1,
                )
                webpage = types.WebPage(
                    id=30,
                    url="https://example.test/video",
                    display_url="example.test/video",
                    hash=31,
                    photo=photo,
                    document=document,
                )
                message = CompoundPreviewMessage(2011206)
                message.media = types.MessageMediaWebPage(webpage)
                record, error = await exporter.message_record(
                    message,
                    Path(directory) / "media",
                    False,
                    set(exporter.MEDIA_TYPES),
                    0,
                    True,
                    0,
                )
                self.assertIsNone(error)
                self.assertEqual(record["media_type"], "video")
                self.assertEqual(record["media_size"], 7)
                self.assertTrue(record["file"].endswith("/video/2011206_preview.mp4"))
                self.assertEqual(record["media_sha256"], hashlib.sha256(b"mp4data").hexdigest())
                self.assertIsNone(message.asserted_thumb)

        asyncio.run(run())

    def test_web_preview_photo_uses_the_exact_cached_representation_size(self):
        from telethon.tl import types
        from telethon.tl.custom.file import File

        class CachedPreviewMessage(FakeMessage):
            async def download_media(self, file: str, thumb=None, progress_callback=None):
                self.asserted_thumb = thumb
                payload = b"j" * 547
                Path(file).write_bytes(payload)
                if progress_callback:
                    progress_callback(len(payload), len(payload))
                return file

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                stripped = types.PhotoStrippedSize(type="i", bytes=b"\x01\x01\x01" + b"s" * 70)
                cached = types.PhotoCachedSize(type="m", w=43, h=33, bytes=b"j" * 547)
                photo = types.Photo(
                    id=40,
                    access_hash=41,
                    file_reference=b"",
                    date=datetime.now(timezone.utc),
                    sizes=[stripped, cached],
                    dc_id=1,
                )
                # Telethon's Message.file calculation prefers the reconstructed
                # stripped size (695), even though download_media selects the
                # 547-byte cached representation.  This was the original bug.
                self.assertEqual(File(photo).size, 695)
                webpage = types.WebPage(
                    id=50,
                    url="https://example.test/photo",
                    display_url="example.test/photo",
                    hash=51,
                    photo=photo,
                )
                message = CachedPreviewMessage(58301)
                message.media = types.MessageMediaWebPage(webpage)
                record, error = await exporter.message_record(
                    message,
                    Path(directory) / "media",
                    False,
                    set(exporter.MEDIA_TYPES),
                    0,
                    True,
                    0,
                )
                self.assertIsNone(error)
                self.assertEqual(message.asserted_thumb, "m")
                self.assertEqual(record["media_type"], "photo")
                self.assertEqual(record["media_size"], 547)
                self.assertTrue(record["file"].endswith("/photo/58301_58301.jpg"))
                self.assertEqual(record["media_sha256"], hashlib.sha256(b"j" * 547).hexdigest())

        asyncio.run(run())

    def test_generated_systemd_keeps_paths_and_uses_database_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit_dir = root / "units"
            db = root / "custom db.sqlite"
            config = root / "secret.env"
            session = root / "custom session"
            output = root / "backup output"
            args = exporter.build_parser().parse_args(
                [
                    "--db", str(db),
                    "install-systemd",
                    "--config", str(config),
                    "--session", str(session),
                    "--output", str(output),
                    "--unit-dir", str(unit_dir),
                    "--mount-point", "/media/lewis/1b",
                ]
            )
            exporter.install_systemd_example(args)
            service = (unit_dir / "tgbackman-telegram-backup.service").read_text(encoding="utf-8")
            self.assertNotIn("--index", service)
            self.assertNotIn("--legacy-json-export", service)
            self.assertIn(f"--config {config}", service)
            self.assertIn(str(session), service)
            self.assertIn("ConditionPathIsMountPoint=/media/lewis/1b", service)

    def test_default_run_indexes_before_watermark(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                db_path = root / "db.sqlite"
                output = root / "output"
                db_indexer.setup_database(str(db_path)).close()
                target = make_target()
                conn = exporter.open_db(db_path)
                conn.execute("INSERT INTO chats(chat_id, chat_name, is_active) VALUES ('chat_1', 'Chat', 1)")
                conn.execute(
                    """INSERT INTO telegram_backup_targets
                    (target_key, source_name, chat_id, peer_kind, peer_id, access_hash, title, enabled, created_unix, updated_unix)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1, 1, 1)""",
                    (target.target_key, target.source_name, target.chat_id, target.peer_kind, target.peer_id, target.access_hash, target.title),
                )
                conn.commit()
                conn.close()
                args = exporter.build_parser().parse_args(
                    [
                        "--db", str(db_path),
                        "--config", str(root / "credentials.env"),
                        "--session", str(root / "session"),
                        "run", "--output", str(output),
                    ]
                )
                with mock.patch.object(
                    exporter,
                    "connect_client",
                    new=mock.AsyncMock(return_value=ExportClient([FakeMessage(1), FakeMessage(2)])),
                ):
                    self.assertEqual(await exporter.run_exports(args), 0)
                conn = exporter.open_db(db_path)
                ledger = conn.execute(
                    "SELECT indexed_unix, applied_unix FROM telegram_backup_exports"
                ).fetchone()
                watermark = conn.execute(
                    "SELECT last_message_id FROM telegram_backup_targets WHERE target_key = ?",
                    (target.target_key,),
                ).fetchone()[0]
                message_count = conn.execute("SELECT count(*) FROM messages").fetchone()[0]
                chat_stats = conn.execute(
                    "SELECT min_msg_id, max_msg_id, msg_count, min_timestamp_unix, max_timestamp_unix "
                    "FROM chats WHERE chat_id = ?",
                    (target.chat_id,),
                ).fetchone()
                conn.close()
                self.assertIsNotNone(ledger[0])
                self.assertIsNotNone(ledger[1])
                self.assertEqual(watermark, 2)
                self.assertEqual(message_count, 2)
                self.assertEqual(tuple(chat_stats), (1, 2, 2, 1, 2))

        asyncio.run(run())

    def test_bulk_run_accepts_duplicate_display_names_when_targets_are_mapped(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                db_path = root / "db.sqlite"
                output = root / "output"
                db_indexer.setup_database(str(db_path)).close()
                conn = exporter.open_db(db_path)
                conn.executemany(
                    "INSERT INTO chats(chat_id, chat_name, is_active) VALUES (?, 'Shared', 1)",
                    [("chat_1",), ("chat_2",)],
                )
                conn.executemany(
                    """INSERT INTO telegram_backup_targets
                    (target_key, source_name, chat_id, peer_kind, peer_id, access_hash,
                     title, enabled, created_unix, updated_unix)
                    VALUES (?, 'Shared', ?, 'user', ?, ?, 'Shared', 1, 1, 1)""",
                    [("shared-one", "chat_1", 1, 2), ("shared-two", "chat_2", 2, 3)],
                )
                conn.commit()
                conn.close()
                args = exporter.build_parser().parse_args(
                    [
                        "--db", str(db_path),
                        "--config", str(root / "credentials.env"),
                        "--session", str(root / "session"),
                        "run", "--output", str(output),
                    ]
                )
                captured = []
                with mock.patch.object(
                    exporter,
                    "connect_client",
                    new=mock.AsyncMock(return_value=ExportClient([FakeMessage(1)])),
                ), redirect_stdout(io.StringIO()) as stream:
                    self.assertEqual(await exporter.run_exports(args), 0)
                    captured.append(stream.getvalue())
                conn = exporter.open_db(db_path)
                self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 2)
                self.assertEqual(
                    conn.execute("SELECT count(*) FROM telegram_backup_targets WHERE last_message_id=1").fetchone()[0],
                    2,
                )
                conn.close()
                self.assertIn("All-chat summary: 2/2 processed", captured[0])
                self.assertIn("chat 1/2", captured[0])
                self.assertIn("chat 2/2", captured[0])

        asyncio.run(run())

    def test_run_all_backs_up_enabled_inactive_targets_without_activating_them(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                db_path = root / "db.sqlite"
                output = root / "output"
                db_indexer.setup_database(str(db_path)).close()
                conn = exporter.open_db(db_path)
                conn.executemany(
                    "INSERT INTO chats(chat_id, chat_name, is_active) VALUES (?, ?, ?)",
                    [
                        ("chat_active", "Active", 1),
                        ("chat_inactive", "Inactive", 0),
                        ("chat_disabled", "Disabled", 0),
                        ("chat_blacklisted", "Blacklisted", 1),
                    ],
                )
                conn.executemany(
                    """INSERT INTO telegram_backup_targets
                    (target_key, source_name, chat_id, peer_kind, peer_id, access_hash,
                     title, enabled, created_unix, updated_unix)
                    VALUES (?, ?, ?, 'user', ?, ?, ?, ?, 1, 1)""",
                    [
                        ("active-key", "Active", "chat_active", 1, 11, "Active", 1),
                        ("inactive-key", "Inactive", "chat_inactive", 2, 22, "Inactive", 1),
                        ("disabled-key", "Disabled", "chat_disabled", 3, 33, "Disabled", 0),
                        ("blacklisted-key", "Blacklisted", "chat_blacklisted", 4, 44, "Blacklisted", 1),
                    ],
                )
                conn.execute(
                    """INSERT INTO telegram_backup_blacklist
                    (target_key, peer_kind, peer_id, title, reason, created_unix)
                    VALUES ('retired-key', 'user', 4, 'Blacklisted', 'test', 1)"""
                )
                conn.commit()
                conn.close()
                args = exporter.build_parser().parse_args(
                    [
                        "--db", str(db_path),
                        "--config", str(root / "credentials.env"),
                        "--session", str(root / "session"),
                        "run", "--all", "--output", str(output),
                    ]
                )
                with mock.patch.object(
                    exporter,
                    "connect_client",
                    new=mock.AsyncMock(return_value=ExportClient([FakeMessage(1)])),
                ), redirect_stdout(io.StringIO()) as stream:
                    self.assertEqual(await exporter.run_exports(args), 0)

                conn = exporter.open_db(db_path)
                backed_up = conn.execute(
                    "SELECT chat_id, count(*) FROM messages GROUP BY chat_id ORDER BY chat_id"
                ).fetchall()
                active_flags = conn.execute(
                    "SELECT chat_id, is_active FROM chats ORDER BY chat_id"
                ).fetchall()
                conn.close()
                self.assertEqual(
                    [tuple(row) for row in backed_up],
                    [("chat_active", 1), ("chat_inactive", 1)],
                )
                self.assertEqual(
                    [tuple(row) for row in active_flags],
                    [
                        ("chat_active", 1),
                        ("chat_blacklisted", 1),
                        ("chat_disabled", 0),
                        ("chat_inactive", 0),
                    ],
                )
                self.assertIn("including inactive chats", stream.getvalue())
                self.assertIn("All-chat summary: 2/2 processed", stream.getvalue())

        asyncio.run(run())

    def test_blacklist_command_deactivates_and_blocks_explicit_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "db.sqlite"
            db_indexer.setup_database(str(db_path)).close()
            conn = exporter.open_db(db_path)
            conn.execute(
                "INSERT INTO chats(chat_id, chat_name, is_active) VALUES ('chat_1', 'Chat', 1)"
            )
            target = make_target()
            conn.execute(
                """INSERT INTO telegram_backup_targets
                (target_key, source_name, chat_id, peer_kind, peer_id, access_hash,
                 title, enabled, created_unix, updated_unix)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, 1, 1)""",
                (
                    target.target_key, target.source_name, target.chat_id, target.peer_kind,
                    target.peer_id, target.access_hash, target.title,
                ),
            )
            conn.commit()
            conn.close()

            blacklist_args = exporter.build_parser().parse_args(
                ["--db", str(db_path), "blacklist-chat", "--target", "chat-key"]
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(exporter.blacklist_chat_command(blacklist_args), 0)

            conn = exporter.open_db(db_path)
            self.assertEqual(exporter.runnable_targets(conn, include_inactive=True), [])
            self.assertEqual(
                conn.execute("SELECT is_active FROM chats WHERE chat_id='chat_1'").fetchone()[0],
                0,
            )
            self.assertEqual(
                conn.execute("SELECT target_key FROM telegram_backup_blacklist").fetchone()[0],
                "chat-key",
            )
            conn.close()

            run_args = exporter.build_parser().parse_args(
                [
                    "--db", str(db_path),
                    "--config", str(root / "credentials.env"),
                    "--session", str(root / "session"),
                    "run", "--target", "chat-key", "--output", str(root / "output"),
                ]
            )
            with self.assertRaisesRegex(exporter.ExportError, "blacklisted target"):
                asyncio.run(exporter.run_exports(run_args))

            remove_args = exporter.build_parser().parse_args(
                [
                    "--db", str(db_path), "blacklist-chat", "--target", "chat-key",
                    "--remove",
                ]
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(exporter.blacklist_chat_command(remove_args), 0)
            conn = exporter.open_db(db_path)
            self.assertEqual(conn.execute("SELECT count(*) FROM telegram_backup_blacklist").fetchone()[0], 0)
            self.assertEqual(
                conn.execute("SELECT is_active FROM chats WHERE chat_id='chat_1'").fetchone()[0],
                0,
            )
            conn.close()

    def test_backup_date_repair_uses_export_batch_snapshot_and_api_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "db.sqlite"
            backup_root = root / "backups"
            converted_a = backup_root / "Converted A"
            converted_b = backup_root / "Converted B"
            unofficial = backup_root / "Unofficial"
            api_path = backup_root / "API"
            never_path = backup_root / "Never"
            for path in (converted_a, converted_b, unofficial, api_path, never_path):
                path.mkdir(parents=True)

            meta = {
                "kind": "html_single_chat_export_converted",
                "converted_from": {"export_root": "/old/export"},
                "created_utc": datetime.fromtimestamp(4000, timezone.utc).isoformat(),
            }
            for path in (converted_a, converted_b):
                (path / ".backman_export_meta.json").write_text(json.dumps(meta))
                generated = path / "messages.html"
                generated.write_text("converted")
                os.utime(generated, (5000, 5000))
            asset_a = converted_a / "style.css"
            asset_b = converted_b / "photo.jpg"
            asset_a.write_text("a")
            asset_b.write_text("b")
            os.utime(asset_a, (1000, 1000))
            os.utime(asset_b, (2000, 2000))
            snapshot = unofficial / "database.sqlite"
            sqlite3.connect(snapshot).close()
            os.utime(snapshot, (3000, 3000))
            (api_path / ".tgbackman_target.json").write_text("{}")
            (never_path / ".tgbackman_target.json").write_text("{}")

            db_indexer.setup_database(str(db_path)).close()
            conn = exporter.open_db(db_path)
            conn.executemany(
                """INSERT INTO chats(
                       chat_id, chat_name, backup_path, max_timestamp_unix, last_backup_unix
                   ) VALUES (?, ?, ?, ?, 9999)""",
                [
                    ("converted-a", "Converted A", str(converted_a), 1500),
                    ("converted-b", "Converted B", str(converted_b), 1500),
                    ("unofficial", "Unofficial", str(unofficial), 2500),
                    ("api", "API", str(api_path), 3400),
                    ("never", "Never", str(never_path), None),
                ],
            )
            conn.execute(
                """INSERT INTO telegram_backup_exports(
                       export_key, target_key, source_name, chat_id, output_path,
                       message_count, created_unix, indexed_unix, applied_unix
                   ) VALUES ('api-run', 'api-key', 'API', 'api', 'sqlite:api-run',
                             2, 3500, 3500, 3500)"""
            )
            conn.commit()

            decisions = {
                item.chat_id: item
                for item in exporter.calculate_backup_date_repairs(
                    conn, backup_root=backup_root.resolve()
                )
            }
            self.assertEqual(decisions["converted-a"].timestamp, 2000)
            self.assertEqual(decisions["converted-b"].timestamp, 2000)
            self.assertEqual(
                decisions["converted-a"].source,
                "converted_desktop_export_asset_batch",
            )
            self.assertEqual(decisions["unofficial"].timestamp, 3000)
            self.assertEqual(decisions["api"].timestamp, 3500)
            self.assertIsNone(decisions["never"].timestamp)
            self.assertEqual(decisions["never"].source, "telegram_api_no_content_commit")
            conn.close()

            args = exporter.build_parser().parse_args(
                [
                    "--db", str(db_path), "repair-backup-dates",
                    "--backup-root", str(backup_root),
                ]
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(exporter.repair_backup_dates_command(args), 0)
            conn = exporter.open_db(db_path)
            repaired = conn.execute(
                "SELECT last_backup_unix, last_backup_source FROM chats WHERE chat_id='converted-a'"
            ).fetchone()
            self.assertEqual(tuple(repaired), (2000, "converted_desktop_export_asset_batch"))
            conn.close()

    def test_run_all_rejects_single_target_scope(self):
        args = exporter.build_parser().parse_args(["run", "--all", "--target", "chat-key"])
        with self.assertRaisesRegex(exporter.ExportError, "--all cannot be combined with --target"):
            asyncio.run(exporter.run_exports(args))

    def test_purge_chat_dry_run_then_removes_aliases_and_only_unshared_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "db.sqlite"
            backup_root = root / "backups"
            target_dir = backup_root / "Purge Me"
            media_dir = target_dir / "media" / "file"
            retained_dir = backup_root / "Retained"
            media_dir.mkdir(parents=True)
            retained_dir.mkdir(parents=True)
            owned_file = media_dir / "owned.bin"
            shared_file = backup_root / "shared.bin"
            owned_file.write_bytes(b"owned media")
            shared_file.write_bytes(b"shared media")
            (target_dir / ".tgbackman_target.json").write_text(
                json.dumps(
                    {"target_key": "purge-key", "chat_id": "chat_purge", "title": "Purge Me"}
                ),
                encoding="utf-8",
            )

            db_indexer.setup_database(str(db_path)).close()
            conn = exporter.open_db(db_path)
            conn.executemany(
                "INSERT INTO chats(chat_id, chat_name, backup_path, is_active) VALUES (?, ?, ?, ?)",
                [
                    ("chat_purge", "Purge Me", str(target_dir), 1),
                    ("chat_alias", "Old Purge Name", str(target_dir), 0),
                    ("chat_keep", "Keep Me", str(retained_dir), 1),
                ],
            )
            conn.execute(
                """INSERT INTO telegram_backup_targets
                (target_key, source_name, chat_id, peer_kind, peer_id, access_hash,
                 title, enabled, output_dir, last_message_id, last_message_unix,
                 last_export_unix, created_unix, updated_unix)
                VALUES ('purge-key', 'Purge Me', 'chat_purge', 'user', 42, 84,
                        'Purge Me', 1, ?, 20, 200, 300, 1, 1)""",
                (str(target_dir),),
            )
            conn.executemany(
                """INSERT INTO telegram_backup_target_chats
                (target_key, chat_id, match_method, linked_unix)
                VALUES ('purge-key', ?, ?, 1)""",
                [("chat_purge", "peer-id"), ("chat_alias", "exact-title")],
            )
            conn.executemany(
                """INSERT INTO backup_sources
                (source_key, source_format, original_path, content_sha256, content_size,
                 compressed_size, compression, payload, imported_unix, message_count)
                VALUES (?, ?, ?, ?, 1, 1, 'zlib', ?, 1, ?)""",
                [
                    ("exclusive-source", "telegram_api", "telegram://user/42/run", "a" * 64, b"x", 1),
                    ("shared-source", "json", str(backup_root / "shared-result.json"), "b" * 64, b"y", 2),
                ],
            )
            conn.executemany(
                """INSERT INTO messages(message_id, chat_id, text, media_type, media_path,
                                          source_key, source_format)
                   VALUES (?, ?, ?, 'file', ?, ?, ?)""",
                [
                    (10, "chat_purge", "owned", str(owned_file), "exclusive-source", "telegram_api"),
                    (20, "chat_alias", "shared old", str(shared_file), "shared-source", "json"),
                    (20, "chat_keep", "shared kept", str(shared_file), "shared-source", "json"),
                ],
            )
            conn.executemany(
                "INSERT INTO message_sources(chat_id, message_id, source_key) VALUES (?, ?, ?)",
                [
                    ("chat_purge", 10, "exclusive-source"),
                    ("chat_alias", 20, "shared-source"),
                    ("chat_keep", 20, "shared-source"),
                ],
            )
            conn.executemany(
                """INSERT INTO backup_imports
                (source_key, source_format, original_path, chat_id, expected_messages,
                 imported_messages, skipped_records, completed_unix)
                VALUES (?, ?, ?, ?, 1, 1, 0, 1)""",
                [
                    ("exclusive-source", "telegram_api", "telegram://user/42/run", "chat_purge"),
                    ("shared-source", "json", str(backup_root / "shared-result.json"), None),
                ],
            )
            conn.execute(
                """INSERT INTO telegram_backup_runs
                (run_key, target_key, chat_id, status, started_unix)
                VALUES ('run-key', 'purge-key', 'chat_purge', 'failed', 1)"""
            )
            conn.execute(
                """INSERT INTO telegram_backup_run_messages
                (run_key, message_id, record_json) VALUES ('run-key', 99, '{}')"""
            )
            conn.execute(
                """INSERT INTO telegram_backup_exports
                (export_key, target_key, source_name, chat_id, output_path,
                 message_count, created_unix)
                VALUES ('export-key', 'purge-key', 'Purge Me', 'chat_purge', ?, 1, 1)""",
                (str(target_dir / "range"),),
            )
            conn.commit()
            conn.close()

            dry_args = exporter.build_parser().parse_args(
                [
                    "--db", str(db_path), "purge-chat", "--target", "purge-key",
                    "--delete-media", "--backup-root", str(backup_root), "--dry-run",
                ]
            )
            with redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(exporter.purge_chat_command(dry_args), 0)
            self.assertIn("Messages to remove: 2", stream.getvalue())
            self.assertIn("shared media retained", stream.getvalue())
            self.assertTrue(owned_file.is_file())
            self.assertTrue(shared_file.is_file())
            conn = exporter.open_db(db_path)
            self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 3)
            conn.close()

            execute_args = exporter.build_parser().parse_args(
                [
                    "--db", str(db_path), "purge-chat", "--target", "purge-key",
                    "--delete-media", "--backup-root", str(backup_root),
                    "--confirm", "purge-key",
                ]
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(exporter.purge_chat_command(execute_args), 0)

            self.assertFalse(target_dir.exists())
            self.assertTrue(shared_file.is_file())
            conn = exporter.open_db(db_path)
            self.assertEqual(
                [tuple(row) for row in conn.execute("SELECT chat_id, message_id FROM messages")],
                [("chat_keep", 20)],
            )
            self.assertEqual(
                [row[0] for row in conn.execute("SELECT chat_id FROM chats")],
                ["chat_keep"],
            )
            target_row = conn.execute(
                """SELECT enabled, last_message_id, last_message_unix, last_export_unix
                   FROM telegram_backup_targets WHERE target_key='purge-key'"""
            ).fetchone()
            self.assertEqual(tuple(target_row), (0, None, None, None))
            with mock.patch.object(
                exporter,
                "entity_description",
                return_value=("user", "Purge Me", None, 42, 84, "FakeEntity"),
            ):
                remapped = exporter.upsert_target(
                    conn, "Purge Me", object(), backup_root, commit=False
                )
            self.assertFalse(remapped.enabled)
            self.assertEqual(exporter.materialize_unbacked_target_chats(conn), (0, 0))
            self.assertEqual(conn.execute("SELECT count(*) FROM telegram_backup_target_chats").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT count(*) FROM telegram_backup_runs").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT count(*) FROM telegram_backup_exports").fetchone()[0], 0)
            self.assertIsNone(
                conn.execute(
                    "SELECT source_key FROM backup_sources WHERE source_key='exclusive-source'"
                ).fetchone()
            )
            self.assertIsNotNone(
                conn.execute(
                    "SELECT source_key FROM backup_sources WHERE source_key='shared-source'"
                ).fetchone()
            )
            purge_status = conn.execute(
                "SELECT status FROM telegram_backup_purges WHERE target_key='purge-key'"
            ).fetchone()[0]
            self.assertEqual(purge_status, "completed")
            self.assertEqual(conn.execute("SELECT count(*) FROM messages_fts").fetchone()[0], 1)
            self.assertEqual(conn.execute("PRAGMA foreign_key_check").fetchall(), [])
            conn.close()

    def test_purge_directory_scan_refuses_symbolic_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            outside = root / "outside.bin"
            target.mkdir()
            outside.write_bytes(b"outside")
            (target / "link.bin").symlink_to(outside)
            with self.assertRaisesRegex(exporter.ExportError, "symbolic link"):
                exporter.scan_directory_without_links(target, root)
            self.assertTrue(outside.is_file())
            self.assertTrue((target / "link.bin").is_symlink())

    def test_purge_chat_requires_exact_confirmation_before_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "db.sqlite"
            db_indexer.setup_database(str(db_path)).close()
            conn = exporter.open_db(db_path)
            conn.execute("INSERT INTO chats(chat_id, chat_name, is_active) VALUES ('chat_1', 'Chat', 1)")
            target = make_target()
            conn.execute(
                """INSERT INTO telegram_backup_targets
                (target_key, source_name, chat_id, peer_kind, peer_id, access_hash,
                 title, enabled, created_unix, updated_unix)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, 1, 1)""",
                (
                    target.target_key, target.source_name, target.chat_id, target.peer_kind,
                    target.peer_id, target.access_hash, target.title,
                ),
            )
            conn.commit()
            conn.close()
            args = exporter.build_parser().parse_args(
                [
                    "--db", str(db_path), "purge-chat", "--target", "chat-key",
                    "--confirm", "wrong-key",
                ]
            )
            with self.assertRaisesRegex(exporter.ExportError, "Confirmation mismatch"):
                exporter.purge_chat_command(args)
            conn = exporter.open_db(db_path)
            self.assertEqual(conn.execute("SELECT count(*) FROM chats").fetchone()[0], 1)
            self.assertEqual(
                conn.execute(
                    "SELECT enabled FROM telegram_backup_targets WHERE target_key='chat-key'"
                ).fetchone()[0],
                1,
            )
            conn.close()

        args = exporter.build_parser().parse_args(
            ["run", "--all", "--chat-output-dir", "/tmp/single-chat"]
        )
        with self.assertRaisesRegex(
            exporter.ExportError, "--all cannot be combined with --chat-output-dir"
        ):
            asyncio.run(exporter.run_exports(args))

    def test_bulk_target_output_dir_is_honored_only_inside_output_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = make_target()
            target.output_dir = str(root / "output" / "Alexandra")
            selected = exporter.target_output_dir(root / "output", target)
            self.assertEqual(selected, (root / "output" / "Alexandra").resolve())
            target.output_dir = str(root / "elsewhere" / "Alexandra")
            selected = exporter.target_output_dir(root / "output", target)
            self.assertEqual(selected, (root / "output" / "Chat").resolve())

    def test_map_all_exposes_unbacked_dialogs_as_inactive_runnable_placeholders(self):
        class MappingEntity:
            def __init__(self, kind, title, peer_id, access_hash):
                self.description = (kind, title, None, peer_id, access_hash, "FakeEntity")

        class Dialog:
            def __init__(self, entity):
                self.entity = entity

        class MappingClient:
            def __init__(self, entities):
                self.entities = entities

            async def get_dialogs(self, limit=None):
                return [Dialog(entity) for entity in self.entities]

            async def disconnect(self):
                return None

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                db_path = root / "db.sqlite"
                db_indexer.setup_database(str(db_path)).close()
                conn = exporter.open_db(db_path)
                conn.executemany(
                    "INSERT INTO chats(chat_id, chat_name, is_active) VALUES (?, ?, ?)",
                    [
                        ("dialog_101", "Active", 1),
                        ("inactive-fragment", "Inactive", 0),
                    ],
                )
                conn.commit()
                conn.close()
                entities = [
                    MappingEntity("user", "Active", 101, 1001),
                    MappingEntity("user", "Inactive", 202, 2002),
                    MappingEntity("user", "Not in archive", 303, 3003),
                ]
                args = exporter.build_parser().parse_args(
                    [
                        "--db", str(db_path),
                        "--config", str(root / "credentials.env"),
                        "--session", str(root / "session"),
                        "map", "--all", "--output", str(root / "output"),
                    ]
                )
                with mock.patch.object(
                    exporter,
                    "connect_client",
                    new=mock.AsyncMock(return_value=MappingClient(entities)),
                ), mock.patch.object(
                    exporter,
                    "entity_description",
                    side_effect=lambda entity: entity.description,
                ):
                    await exporter.map_targets(args)

                conn = exporter.open_db(db_path)
                self.assertEqual(
                    conn.execute(f"SELECT count(*) FROM {exporter.DIALOGS_TABLE}").fetchone()[0],
                    3,
                )
                self.assertEqual(
                    conn.execute(f"SELECT count(*) FROM {exporter.TARGETS_TABLE}").fetchone()[0],
                    3,
                )
                links = conn.execute(
                    f"SELECT chat_id, match_method FROM {exporter.TARGET_CHAT_LINKS_TABLE} ORDER BY chat_id"
                ).fetchall()
                self.assertEqual(
                    [(row[0], row[1]) for row in links],
                    [
                        ("dialog_101", "peer-id"),
                        ("dialog_303", "telegram-discovered"),
                        ("inactive-fragment", "exact-title"),
                    ],
                )
                placeholder = conn.execute(
                    """SELECT chat_name, chat_type, backup_path, is_active, msg_count,
                              min_msg_id, max_msg_id
                       FROM chats WHERE chat_id = 'dialog_303'"""
                ).fetchone()
                self.assertEqual(
                    tuple(placeholder),
                    (
                        "Not in archive",
                        "personal_chat",
                        str(root / "output" / "Not in archive"),
                        0,
                        0,
                        None,
                        None,
                    ),
                )
                runnable = exporter.load_targets(conn, active_only=True)
                self.assertEqual([(target.title, target.peer_id) for target in runnable], [("Active", 101)])
                conn.execute("UPDATE chats SET is_active=1 WHERE chat_id='dialog_303'")
                runnable = exporter.load_targets(conn, active_only=True)
                self.assertEqual(
                    [(target.title, target.peer_id) for target in runnable],
                    [("Active", 101), ("Not in archive", 303)],
                )
                conn.close()

        asyncio.run(run())

    def test_automatic_mapping_does_not_guess_duplicate_dialog_titles(self):
        chat = exporter.DatabaseChat("chat-1", "Shared", True, None)
        first = (object(), ("user", "Shared", None, 1, 10, "Fake"))
        second = (object(), ("user", "Shared", None, 2, 20, "Fake"))
        assignments, unresolved = exporter.match_dialogs_to_database_chats(
            [chat], [first, second]
        )
        self.assertEqual(assignments, {})
        self.assertEqual(unresolved[0][0].chat_id, "chat-1")
        self.assertIn("more than one", unresolved[0][1])

    def test_migrated_group_is_consolidated_into_current_supergroup_target(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "db.sqlite"
            db_indexer.setup_database(str(db_path)).close()
            conn = exporter.open_db(db_path)
            conn.executemany(
                "INSERT INTO chats(chat_id, chat_name, is_active) VALUES (?, ?, 1)",
                [("old-chat", "Old name"), ("new-chat", "New name")],
            )
            conn.executemany(
                """INSERT INTO telegram_backup_targets
                (target_key, source_name, chat_id, peer_kind, peer_id, title,
                 enabled, last_message_id, created_unix, updated_unix)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, 1, 1)""",
                [
                    ("old-target", "Old name", "old-chat", "group", 100, "Old name", 9000),
                    ("new-target", "New name", "new-chat", "channel", 200, "New name", 25),
                ],
            )
            conn.executemany(
                """INSERT INTO telegram_backup_target_chats
                (target_key, chat_id, match_method, linked_unix) VALUES (?, ?, 'peer-id', 1)""",
                [("old-target", "old-chat"), ("new-target", "new-chat")],
            )
            conn.commit()

            old_entity = type(
                "OldGroup",
                (),
                {"migrated_to": type("Destination", (), {"channel_id": 200})()},
            )()
            new_entity = type("NewSupergroup", (), {"migrated_to": None})()
            candidates = [
                (old_entity, ("group", "Old name", None, 100, None, "Chat")),
                (new_entity, ("channel", "New name", None, 200, 300, "Channel")),
            ]
            self.assertEqual(
                exporter.consolidate_migrated_targets(conn, candidates),
                [("old-target", "new-target")],
            )
            old_enabled = conn.execute(
                "SELECT enabled FROM telegram_backup_targets WHERE target_key='old-target'"
            ).fetchone()[0]
            new_watermark = conn.execute(
                "SELECT last_message_id FROM telegram_backup_targets WHERE target_key='new-target'"
            ).fetchone()[0]
            links = conn.execute(
                "SELECT target_key, chat_id, match_method FROM telegram_backup_target_chats ORDER BY chat_id"
            ).fetchall()
            self.assertEqual(old_enabled, 0)
            self.assertEqual(new_watermark, 25)
            self.assertEqual(
                [tuple(row) for row in links],
                [
                    ("new-target", "new-chat", "peer-id"),
                    ("new-target", "old-chat", "telegram-migrated-from"),
                ],
            )
            conn.execute("UPDATE chats SET is_active=0")
            conn.execute("UPDATE chats SET is_active=1 WHERE chat_id='old-chat'")
            self.assertEqual(exporter.load_targets(conn, active_only=True), [])
            conn.execute(
                "INSERT INTO messages(message_id, chat_id, text) VALUES (1, 'old-chat', 'history')"
            )
            self.assertEqual(
                [target.target_key for target in exporter.load_targets(conn, active_only=True)],
                ["new-target"],
            )
            conn.close()

    def test_database_stream_preserves_failed_staging_and_resumes_atomically(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                db_path = root / "db.sqlite"
                db_indexer.setup_database(str(db_path)).close()
                conn = exporter.open_db(db_path)
                target = make_target()
                conn.execute("INSERT INTO chats(chat_id, chat_name, is_active) VALUES ('chat_1', 'Chat', 1)")
                conn.execute(
                    """INSERT INTO telegram_backup_targets
                    (target_key, source_name, chat_id, peer_kind, peer_id, access_hash,
                     title, enabled, created_unix, updated_unix)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1, 1, 1)""",
                    (target.target_key, target.source_name, target.chat_id, target.peer_kind,
                     target.peer_id, target.access_hash, target.title),
                )
                conn.commit()

                async def interrupted():
                    yield {"id": 1, "date_unixtime": "100", "text": "one"}, None
                    raise RuntimeError("interrupted")

                with self.assertRaises(RuntimeError):
                    await exporter.write_database_stream(
                        conn, target, interrupted(), root / "chat", "run", None, None, False
                    )
                self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 0)
                self.assertIsNone(
                    conn.execute("SELECT last_message_id FROM telegram_backup_targets").fetchone()[0]
                )
                self.assertEqual(
                    conn.execute("SELECT count(*) FROM telegram_backup_run_messages").fetchone()[0], 1
                )

                async def resumed():
                    yield {"id": 1, "date_unixtime": "100", "text": "one edited"}, None
                    yield {"id": 2, "date_unixtime": "200", "text": "two"}, None

                stats = await exporter.write_database_stream(
                    conn, target, resumed(), root / "chat", "run", None, None, False
                )
                self.assertEqual(stats.message_count, 2)
                self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 2)
                self.assertEqual(
                    conn.execute("SELECT text FROM messages WHERE message_id=1").fetchone()[0],
                    "one edited",
                )
                self.assertEqual(
                    conn.execute("SELECT last_message_id FROM telegram_backup_targets").fetchone()[0], 2
                )
                self.assertEqual(
                    conn.execute("SELECT count(*) FROM telegram_backup_run_messages").fetchone()[0],
                    0,
                )
                async def no_new_messages():
                    if False:
                        yield {}, None

                no_new_stats = await exporter.write_database_stream(
                    conn, target, no_new_messages(), root / "chat", "run-no-new", 2, 200, False
                )
                self.assertEqual(no_new_stats.message_count, 0)
                run_timestamp, run_status = conn.execute(
                    "SELECT last_backup_run_unix, last_backup_run_status FROM chats WHERE chat_id='chat_1'"
                ).fetchone()
                self.assertIsNotNone(run_timestamp)
                self.assertEqual(run_status, "completed_no_new_messages")
                conn.close()

        asyncio.run(run())

    def test_staged_resume_boundary_skips_persisted_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "db.sqlite"
            db_indexer.setup_database(str(db_path)).close()
            conn = exporter.open_db(db_path)
            conn.execute(
                """INSERT INTO telegram_backup_runs
                (run_key, target_key, chat_id, status, started_unix)
                VALUES ('run', 'target', 'chat', 'failed', 1)"""
            )
            conn.executemany(
                "INSERT INTO telegram_backup_run_messages(run_key, message_id, record_json) VALUES (?, ?, ?)",
                [
                    ("run", 10, json.dumps({"id": 10, "text": "one"})),
                    ("run", 11, json.dumps({"id": 11, "text": "two"})),
                ],
            )
            conn.commit()
            resume_after, count = exporter.staged_resume_after_id(
                conn, "run", Path(directory) / "Chat"
            )
            self.assertEqual((resume_after, count), (11, 2))
            conn.close()

    def test_prune_completed_staging_keeps_failed_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "db.sqlite"
            db_indexer.setup_database(str(db_path)).close()
            conn = exporter.open_db(db_path)
            conn.executemany(
                """INSERT INTO telegram_backup_runs
                (run_key, target_key, chat_id, status, started_unix, completed_unix)
                VALUES (?, 'target', 'chat', ?, 1, ?)""",
                [
                    ("completed", "completed", 2),
                    ("failed", "failed", None),
                ],
            )
            conn.executemany(
                """INSERT INTO telegram_backup_run_messages
                (run_key, message_id, record_json) VALUES (?, ?, ?)""",
                [
                    ("completed", 1, json.dumps({"id": 1})),
                    ("failed", 2, json.dumps({"id": 2})),
                ],
            )
            conn.commit()
            self.assertEqual(exporter.prune_completed_staging(conn), 1)
            self.assertEqual(
                [tuple(row) for row in conn.execute(
                    "SELECT run_key, message_id FROM telegram_backup_run_messages"
                )],
                [("failed", 2)],
            )
            conn.close()

    def test_exact_and_id_overlap_boundaries(self):
        async def collect(overlap_ids, overlap_seconds=0, resume_after_id=None):
            with tempfile.TemporaryDirectory() as directory:
                messages = [FakeMessage(i) for i in range(1, 13)]
                records = []
                async for record, _ in exporter.iter_message_records(
                    FakeClient(messages),
                    object(),
                    10,
                    10,
                    Path(directory) / "media",
                    overlap_ids,
                    overlap_seconds,
                    False,
                    set(exporter.MEDIA_TYPES),
                    0,
                    False,
                    0,
                    False,
                    False,
                    None,
                    resume_after_id,
                ):
                    records.append(record["id"])
                return records

        self.assertEqual(asyncio.run(collect(0)), [11, 12])
        self.assertEqual(asyncio.run(collect(2)), [9, 10, 11, 12])
        self.assertEqual(asyncio.run(collect(0, 2)), [8, 9, 10, 11, 12])
        self.assertEqual(asyncio.run(collect(0, 0, 10)), [11, 12])

    def test_date_only_boundary_re_reads_same_second(self):
        async def collect():
            with tempfile.TemporaryDirectory() as directory:
                records = []
                async for record, _ in exporter.iter_message_records(
                    FakeClient([FakeMessage(9), FakeMessage(10), FakeMessage(11)]),
                    object(), None, 10, Path(directory) / "media", 0, 0, False,
                    set(exporter.MEDIA_TYPES), 0, False, 0, False, False, None,
                ):
                    records.append(record["id"])
                return records

        self.assertEqual(asyncio.run(collect()), [10, 11])

    def test_ledger_applies_watermark_after_index(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "db.sqlite"
            db_indexer.setup_database(str(db_path)).close()
            conn = exporter.open_db(db_path)
            conn.execute("INSERT INTO chats(chat_id, chat_name, is_active) VALUES ('chat_1', 'Chat', 1)")
            conn.execute(
                """INSERT INTO telegram_backup_targets
                (target_key, source_name, chat_id, peer_kind, peer_id, access_hash, title, created_unix, updated_unix)
                VALUES ('chat-key', 'Chat', 'chat_1', 'user', 1, 2, 'Chat', 1, 1)"""
            )
            target = make_target()
            stats = exporter.ExportStats(message_count=2, first_message_id=1, last_message_id=2)
            export_path = Path(directory) / "Chat" / "range"
            export_path.mkdir(parents=True)
            (export_path / "result.json").write_text("{}", encoding="utf-8")
            exporter.record_export(conn, target, export_path, stats)
            self.assertIsNone(conn.execute("SELECT last_message_id FROM telegram_backup_targets").fetchone()[0])
            # A normal reconciliation must not apply an unindexed export.
            exporter.apply_export_watermarks(conn, output_root=Path(directory))
            self.assertIsNone(conn.execute("SELECT last_message_id FROM telegram_backup_targets").fetchone()[0])
            exporter.mark_exports_indexed(conn, Path(directory))
            exporter.apply_export_watermarks(conn, require_indexed=True, output_root=Path(directory))
            self.assertEqual(conn.execute("SELECT last_message_id FROM telegram_backup_targets").fetchone()[0], 2)
            conn.close()


class IndexerTests(unittest.TestCase):
    def test_html_parser_keeps_final_message_after_br_and_skips_day_separator(self):
        parser = db_indexer.TelegramHTMLParser()
        parser.feed(
            '<div class="message service" id="message-1"><div class="text">Day</div></div>'
            '<div class="message default clearfix" id="message42">'
            '<div class="from_name">Alex</div><div class="text">line one<br>line two</div></div>'
        )
        parser.close()
        parser.flush_current()
        self.assertEqual(len(parser.messages), 1)
        self.assertEqual(parser.messages[0]["message_id"], 42)
        self.assertEqual(parser.messages[0]["text"], "line one\nline two")

    def test_rich_json_fields_and_exact_source_are_archived(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / "Alex" / "2024-01-01T00-00-00Z__2024-01-01T00-00-01Z"
            export.mkdir(parents=True)
            payload = {
                "messages": [{
                    "id": 7,
                    "type": "message",
                    "date": "2024-01-01T00:00:00Z",
                    "text": [{"type": "link", "text": "site", "href": "https://example.test"}],
                    "entities": [{"type": "url", "offset": 0, "length": 4}],
                    "reactions": [{"emoji": "👍", "count": 2}],
                    "grouped_id": 123,
                }]
            }
            result = export / "result.json"
            result.write_text(json.dumps(payload), encoding="utf-8")
            db_path = root / "archive.db"
            db_indexer.index_backup_folder(
                str(root), str(db_path), log_fn=lambda _: None, archive_sources=True
            )
            conn = sqlite3.connect(db_path)
            row = conn.execute(
                "SELECT text, entities_json, reactions_json, grouped_id, raw_payload FROM messages"
            ).fetchone()
            source = conn.execute(
                "SELECT compression, length(payload) FROM backup_sources"
            ).fetchone()
            conn.close()
            self.assertEqual(row[0], "site")
            self.assertIn("url", row[1])
            self.assertIn("👍", row[2])
            self.assertEqual(row[3], "123")
            self.assertTrue(row[4])
            self.assertEqual(source[0], "zlib")
            self.assertGreater(source[1], 0)
            self.assertEqual(
                db_indexer.verify_database_archive(
                    str(db_path), require_archived_sources=True
                ),
                [],
            )

    def test_unofficial_sqlite_import_keeps_rich_fields_and_skips_placeholders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "unofficial"
            source_dir.mkdir()
            source = source_dir / "database.sqlite"
            conn = sqlite3.connect(source)
            conn.executescript(
                """
                CREATE TABLE users(id INTEGER, first_name TEXT, last_name TEXT);
                CREATE TABLE chats(id INTEGER, title TEXT);
                CREATE TABLE messages(
                    id INTEGER PRIMARY KEY, message_id INTEGER, message_type TEXT,
                    source_type TEXT, source_id INTEGER, sender_id INTEGER,
                    fwd_from_id INTEGER, text TEXT, time INTEGER, has_media INTEGER,
                    media_type TEXT, media_file TEXT, media_size INTEGER,
                    media_json TEXT, markup_json TEXT, data BLOB, api_layer INTEGER
                );
                INSERT INTO users VALUES(9, 'Sender', 'Name');
                INSERT INTO messages VALUES(
                    1, 55, 'message', 'dialog', 99, 9, 8, 'hello', 1000, 1,
                    'document', 'file.bin', 4, '{"kind":"doc"}', '{"button":1}', X'0102', 201
                );
                INSERT INTO messages(id, message_id) VALUES(2, 56);
                """
            )
            conn.commit()
            conn.close()
            db_path = root / "archive.db"
            db_indexer.index_backup_folder(
                str(root), str(db_path), log_fn=lambda _: None, archive_sources=True
            )
            conn = sqlite3.connect(db_path)
            row = conn.execute(
                """SELECT message_type, forwarded_from, media_size, reply_markup_json,
                          extra_json, raw_payload FROM messages"""
            ).fetchone()
            count = conn.execute("SELECT count(*) FROM messages").fetchone()[0]
            conn.close()
            self.assertEqual(count, 1)
            self.assertEqual(row[0:3], ("message", "8", 4))
            self.assertIn("button", row[3])
            self.assertIn("unofficial_api_layer", row[4])
            self.assertTrue(row[5])

    def test_indexer_upserts_edits_and_resolves_media_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / "export"
            media = export / "media" / "file"
            media.mkdir(parents=True)
            (media / "x.bin").write_bytes(b"hello")
            result = export / "result.json"
            payload = {
                "chats": {
                    "list": [
                        {
                            "id": "chat_1",
                            "name": "Chat",
                            "messages": [
                                {
                                    "id": 1,
                                    "date": "2020-01-01T00:00:00Z",
                                    "date_unixtime": "1577836800",
                                    "text": "old",
                                    "media_type": "file",
                                    "file": "media/file/x.bin",
                                }
                            ],
                        }
                    ]
                }
            }
            result.write_text(json.dumps(payload), encoding="utf-8")
            db_path = root / "db.sqlite"
            db_indexer.index_backup_folder(str(root), str(db_path), log_fn=lambda _: None)
            payload["chats"]["list"][0]["messages"][0]["text"] = "edited"
            result.write_text(json.dumps(payload), encoding="utf-8")
            db_indexer.index_backup_folder(str(root), str(db_path), log_fn=lambda _: None)
            conn = sqlite3.connect(db_path)
            row = conn.execute("SELECT text, media_path FROM messages WHERE chat_id='chat_1' AND message_id=1").fetchone()
            self.assertEqual(row[0], "edited")
            self.assertEqual(row[1], str(media / "x.bin"))
            conn.close()

    def test_indexer_raises_when_json_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / "export"
            export.mkdir()
            (export / "result.json").write_text("{not valid json", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                db_indexer.index_backup_folder(str(root), str(root / "db.sqlite"), log_fn=lambda _: None)


if __name__ == "__main__":
    unittest.main()
