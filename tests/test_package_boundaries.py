import unittest
from datetime import datetime
from pathlib import Path

import tgbackup.config as config
from tgbackup.backup.media import media_filename
from tgbackup.backup.records import range_dir_name_from_stats
from tgbackup.backup.targets import target_key
from tgbackup.db.archive import archival_message_values
from tgbackup.db.schema import ensure_targets_schema
from tgbackup.models import ExportStats
from tgbackup.progress import ExportLock, ProgressReporter


class PackageBoundaryTests(unittest.TestCase):
    def test_console_package_exports_configuration_helpers(self):
        self.assertEqual(config.parse_size("4MiB"), 4 * 1024 * 1024)
        self.assertEqual(config.parse_size("1MB"), 1_000_000)
        self.assertTrue(config.telethon_session_file(Path("telegram")).name.endswith(".session"))

    def test_media_and_progress_services_are_importable_without_client(self):
        class Message:
            id = 42

        self.assertEqual(media_filename(Message()), "42_42")
        self.assertTrue(ExportLock(Path("/tmp/tgbackman-test.lock")))
        self.assertTrue(ProgressReporter("test", enabled=False))

    def test_database_and_target_services_have_stable_narrow_interfaces(self):
        self.assertIn("example_chat-", target_key("Example Chat", "user", 42))
        self.assertTrue(range_dir_name_from_stats(ExportStats(), datetime.now()))
        self.assertEqual(len(archival_message_values("chat", {"id": 1}, ".", "source", "json")), 27)
        self.assertTrue(callable(ensure_targets_schema))


if __name__ == "__main__":
    unittest.main()
