import unittest
from pathlib import Path

import tgbackup.config as config
from tgbackup.backup.media import media_filename
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


if __name__ == "__main__":
    unittest.main()
