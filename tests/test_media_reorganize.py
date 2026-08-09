from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tgbackup.db.schema import setup_database
from tgbackup.media_reorganize import apply_plan, build_plan


class MediaReorganisationTests(unittest.TestCase):
    def _database(self, path: Path, source: Path) -> None:
        conn = setup_database(path)
        payload = b"same media"
        digest = hashlib.sha256(payload).hexdigest()
        conn.executemany(
            "INSERT INTO chats(chat_id, chat_name, chat_type, backup_path, is_active) VALUES (?, ?, ?, ?, ?)",
            [
                ("chat-one", "Example Group", "group", str(source / "old-range"), 1),
                ("chat-two", "Example Group", "group", str(source / "old-range"), 1),
            ],
        )
        conn.executemany(
            """INSERT INTO telegram_backup_targets
               (target_key, source_name, chat_id, peer_kind, peer_id, title, created_unix, updated_unix)
               VALUES (?, ?, ?, ?, ?, ?, 1, 1)""",
            [
                ("one", "Example Group", "chat-one", "group", 10, "Example Group"),
                ("two", "Example Group", "chat-two", "group", 11, "Example Group"),
            ],
        )
        conn.executemany(
            """INSERT INTO messages
               (chat_id, message_id, timestamp_unix, media_type, media_path, media_size, media_sha256, media_status)
               VALUES (?, ?, 1, 'photo', 'media/photo/pic.jpg', ?, ?, 'downloaded')""",
            [("chat-one", 1, len(payload), digest), ("chat-two", 1, len(payload), digest)],
        )
        conn.commit()
        conn.close()

    def test_planner_uses_peer_identity_and_keeps_duplicate_chat_paths_visible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy"
            destination = root / "canonical"
            media = source / "old-range" / "media" / "photo"
            media.mkdir(parents=True)
            (media / "pic.jpg").write_bytes(b"same media")
            db = root / "backup.db"
            self._database(db, source)

            plan = build_plan(db, source, destination)

            self.assertEqual(len(plan.chats), 2)
            self.assertEqual(len(plan.media), 2)
            self.assertNotEqual(plan.chats[0].destination, plan.chats[1].destination)
            self.assertTrue(all(item.relative == "media/photo/pic.jpg" for item in plan.media))
            self.assertFalse(destination.exists())

    def test_apply_updates_database_only_after_verified_reflinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy"
            destination = root / "canonical"
            media = source / "old-range" / "media" / "photo"
            media.mkdir(parents=True)
            (media / "pic.jpg").write_bytes(b"same media")
            db = root / "backup.db"
            self._database(db, source)
            plan = build_plan(db, source, destination)
            manifest = root / "manifest.json"
            copied_from: list[Path] = []

            def fake_reflink(source_path: Path, destination_path: Path) -> None:
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                copied_from.append(source_path)
                destination_path.write_bytes(source_path.read_bytes())

            with mock.patch("tgbackup.media_reorganize._filesystem", return_value="btrfs"), mock.patch(
                "tgbackup.media_reorganize._copy_reflink", side_effect=fake_reflink
            ):
                apply_plan(plan, manifest)

            self.assertTrue(manifest.is_file())
            self.assertTrue(json.loads(manifest.read_text(encoding="utf-8"))["db_committed"])
            conn = setup_database(db)
            rows = conn.execute("SELECT chat_id, backup_path FROM chats ORDER BY chat_id").fetchall()
            target_rows = conn.execute("SELECT chat_id, output_dir FROM telegram_backup_targets ORDER BY chat_id").fetchall()
            media_rows = conn.execute("SELECT chat_id, media_path FROM messages ORDER BY chat_id").fetchall()
            conn.close()
            self.assertTrue(all(Path(row[1]).is_dir() for row in rows))
            self.assertEqual([row[1] for row in target_rows], [row[1] for row in rows])
            self.assertEqual([row[1] for row in media_rows], ["media/photo/pic.jpg", "media/photo/pic.jpg"])
            for chat_id, media_path in media_rows:
                chat_path = next(Path(row[1]) for row in rows if row[0] == chat_id)
                self.assertEqual((chat_path / media_path).read_bytes(), b"same media")
            self.assertTrue((source / "old-range" / "media" / "photo" / "pic.jpg").is_file())
            self.assertEqual(copied_from[1], Path(rows[0][1]) / "media/photo/pic.jpg")

    def test_metadata_mismatch_is_reported_without_a_plan_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy"
            media = source / "old-range" / "media" / "photo"
            media.mkdir(parents=True)
            (media / "pic.jpg").write_bytes(b"actual")
            db = root / "backup.db"
            self._database(db, source)
            conn = setup_database(db)
            conn.execute("UPDATE messages SET media_size=999 WHERE chat_id='chat-one'")
            conn.commit()
            conn.close()

            plan = build_plan(db, source, root / "canonical")
            self.assertTrue(any("chat-one:1" in error for error in plan.mismatched))
            self.assertEqual(len([item for item in plan.media if item.chat_id == "chat-one"]), 0)


if __name__ == "__main__":
    unittest.main()
