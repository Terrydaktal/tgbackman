import sqlite3
import tempfile
import unittest
from pathlib import Path

from tgbackup.database import range_repair as ranges


HTML = """
<div class="message default clearfix" id="message1">
  <div class="pull_right date details" title="01.02.2024 03:04:05 UTC+00:00"></div>
</div>
<div class="message default clearfix" id="message2">
  <div class="pull_right date details" title="06.02.2024 07:08:09 UTC+00:00"></div>
</div>
"""


class FlatRangeWrappingTests(unittest.TestCase):
    def test_flat_wrap_moves_export_but_preserves_chat_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            chat = Path(directory) / "example-chat"
            photos = chat / "photos"
            photos.mkdir(parents=True)
            (chat / "messages.html").write_text(HTML, encoding="utf-8")
            (photos / "photo.jpg").write_bytes(b"photo")
            marker = chat / ".tgbackman_target.json"
            marker.write_text("{}", encoding="utf-8")

            range_name = ranges._compute_range_dir(str(chat), recursive=False)
            self.assertEqual(range_name, "2024-02-01T03-04-05Z__2024-02-06T07-08-09Z")
            ranges._wrap_flat_backup(str(chat), range_name, apply=True)

            destination = chat / range_name
            self.assertTrue((destination / "messages.html").is_file())
            self.assertTrue((destination / "photos" / "photo.jpg").is_file())
            self.assertTrue(marker.is_file())
            self.assertFalse((chat / "messages.html").exists())

    def test_dry_run_does_not_move_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            chat = Path(directory) / "chat"
            chat.mkdir()
            (chat / "messages.html").write_text(HTML, encoding="utf-8")
            range_name = ranges._compute_range_dir(str(chat), recursive=False)

            ranges._wrap_flat_backup(str(chat), range_name, apply=False)

            self.assertTrue((chat / "messages.html").is_file())
            self.assertFalse((chat / range_name).exists())

    def test_wrap_migrates_relative_database_media_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chat = root / "example-chat"
            media = chat / "files" / "voice.ogg"
            media.parent.mkdir(parents=True)
            media.write_bytes(b"voice")
            (chat / "messages.html").write_text(HTML, encoding="utf-8")
            destination = chat / "2024-02-01T03-04-05Z__2024-02-06T07-08-09Z"

            conn = sqlite3.connect(root / "index.db")
            conn.executescript(
                """
                CREATE TABLE chats (
                    chat_id TEXT PRIMARY KEY,
                    chat_name TEXT,
                    backup_path TEXT
                );
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY,
                    chat_id TEXT,
                    media_path TEXT
                );
                """
            )
            conn.execute(
                "INSERT INTO chats VALUES (?, ?, ?)",
                ("example-chat", "Example Chat", str(chat)),
            )
            conn.execute(
                "INSERT INTO messages VALUES (?, ?, ?)",
                (1, "example-chat", "files/voice.ogg"),
            )
            conn.commit()

            chat_ids, updates, missing = ranges._plan_db_migration(
                conn, str(chat), str(destination)
            )
            self.assertEqual(chat_ids, ["example-chat"])
            self.assertEqual(missing, 0)
            self.assertEqual(updates, [(str(destination / "files" / "voice.ogg"), 1)])

            conn.execute("BEGIN IMMEDIATE")
            moved = ranges._wrap_flat_backup(
                str(chat), destination.name, apply=True, announce=False
            )
            ranges._apply_db_migration(conn, chat_ids, updates, str(destination))
            conn.commit()

            self.assertIn("files", moved)
            self.assertTrue((destination / "files" / "voice.ogg").is_file())
            self.assertEqual(
                conn.execute(
                    "SELECT backup_path FROM chats WHERE chat_id = 'example-chat'"
                ).fetchone()[0],
                str(destination),
            )
            self.assertEqual(
                conn.execute(
                    "SELECT media_path FROM messages WHERE id = 1"
                ).fetchone()[0],
                str(destination / "files" / "voice.ogg"),
            )
            conn.close()

    def test_rollback_restores_wrapped_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            chat = Path(directory) / "chat"
            chat.mkdir()
            (chat / "messages.html").write_text(HTML, encoding="utf-8")
            range_name = ranges._compute_range_dir(str(chat), recursive=False)
            destination = chat / range_name

            moved = ranges._wrap_flat_backup(
                str(chat), range_name, apply=True, announce=False
            )
            ranges._rollback_flat_backup(str(chat), str(destination), moved)

            self.assertTrue((chat / "messages.html").is_file())
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
