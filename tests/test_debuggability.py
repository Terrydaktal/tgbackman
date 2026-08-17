import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from tgbackup.diagnostics import (
    DiagnosticEvent,
    DiagnosticRecorder,
    OperationRegistry,
    exception_details,
    build_identity,
    register_secret,
)
from tgbackup import exporter
from tgbackup.exporter import reap_stale_attempts_command
from tgbackup.progress import ProgressReporter
from tgbackup.db.schema import append_diagnostic_event, ensure_targets_schema, verify_diagnostic_event_chain


class DebuggabilityTests(unittest.TestCase):
    def test_exception_details_are_bounded_and_redact_secret_keys(self):
        register_secret("file-loaded-secret-1234")
        with mock.patch.dict(os.environ, {"TG_API_HASH": "abcdef0123456789"}):
            try:
                raise RuntimeError("failure with abcdef0123456789 and file-loaded-secret-1234")
            except RuntimeError as exc:
                details = exception_details(
                    exc,
                    phase="test",
                    fields={"api_hash": "must-not-appear", "safe": "yes"},
                )
        self.assertEqual(details["exception_type"], "RuntimeError")
        self.assertNotIn("must-not-appear", json.dumps(details))
        self.assertNotIn("file-loaded-secret-1234", json.dumps(details))
        self.assertIn("safe", details["fields"])
        self.assertLessEqual(len(details["traceback"]), 12_000)

    def test_operation_registry_writes_atomic_snapshot_and_removes_on_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            registry = OperationRegistry(path)
            registry.update("running", target_key="target-1", secret="not persisted")
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "running")
            self.assertEqual(saved["target_key"], "target-1")
            self.assertNotIn("secret", saved)
            registry.finish("completed")
            self.assertFalse(path.exists())

    def test_diagnostic_recorder_is_jsonl_and_best_effort(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            recorder = DiagnosticRecorder(path)
            recorder.emit(DiagnosticEvent(event="started", component="test"))
            event = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(event["event"], "started")
            self.assertEqual(event["schema"], 1)

    def test_diagnostic_sink_failure_never_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = DiagnosticRecorder(Path(directory))
            recorder.emit(DiagnosticEvent(event="sink_failure", component="test"))

    def test_no_progress_does_not_touch_hot_path_accounting(self):
        reporter = ProgressReporter("test", enabled=False)
        with mock.patch("tgbackup.progress.time.monotonic", side_effect=AssertionError("clock called")):
            reporter.observe({"id": 1, "date_unixtime": 1, "media_type": "photo"}, None)
            reporter.media_download_progress(1, "image.jpg", 1, 2)
        self.assertEqual(reporter.processed, 0)

    def test_run_ledgers_have_structured_failure_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "backup.db"
            conn = sqlite3.connect(path)
            ensure_targets_schema(conn)
            self.assertIsNotNone(
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='telegram_backup_diagnostic_events'"
                ).fetchone()
            )
            for table in ("telegram_backup_runs", "telegram_backup_run_attempts"):
                columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                self.assertTrue(
                    {
                        "error_type",
                        "error_phase",
                        "error_traceback",
                        "diagnostic_json",
                        "build_revision",
                        "operation_id",
                        "heartbeat_unix",
                    }
                    <= columns
                )
            event_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(telegram_backup_diagnostic_events)")
            }
            self.assertTrue({"actor", "writer_role", "reason", "outcome"} <= event_columns)
            append_diagnostic_event(
                conn,
                event_id="test-event",
                event_type="test",
                component="tests",
                level="info",
                operation_id=None,
                run_key=None,
                target_key="target",
                status="completed",
                details_json="{}",
                build_revision="test",
                event_unix=int(time.time()),
            )
            audit_row = conn.execute(
                "SELECT actor, writer_role, reason, outcome FROM telegram_backup_diagnostic_events "
                "WHERE event_id='test-event'"
            ).fetchone()
            self.assertEqual(tuple(audit_row)[1:], ("tests", "test", "completed"))
            self.assertTrue(audit_row[0])
            integrity = conn.execute(
                "SELECT previous_hash, integrity_sha256, host_name, process_id "
                "FROM telegram_backup_diagnostic_events WHERE event_id='test-event'"
            ).fetchone()
            self.assertIsNotNone(integrity[1])
            self.assertTrue(integrity[2])
            self.assertGreater(integrity[3], 0)
            valid, reason = verify_diagnostic_event_chain(conn)
            self.assertTrue(valid, reason)
            append_diagnostic_event(
                conn,
                event_id="large-event",
                event_type="large",
                component="tests",
                level="info",
                operation_id=None,
                run_key=None,
                target_key=None,
                status="completed",
                details_json=json.dumps({"value": "x" * 20_000}),
                build_revision="test",
                event_unix=int(time.time()),
            )
            bounded = conn.execute(
                "SELECT details_json FROM telegram_backup_diagnostic_events WHERE event_id='large-event'"
            ).fetchone()[0]
            self.assertLessEqual(len(bounded), 12_000)
            self.assertTrue(json.loads(bounded)["truncated"])
            conn.execute(
                "UPDATE telegram_backup_diagnostic_events SET outcome='tampered' WHERE event_id='test-event'"
            )
            valid, reason = verify_diagnostic_event_chain(conn)
            self.assertFalse(valid)
            self.assertIn("integrity", reason or "")
            conn.close()

    def test_stale_reaper_uses_heartbeat_and_does_not_reap_live_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "backup.db"
            conn = sqlite3.connect(path)
            ensure_targets_schema(conn)
            now = int(time.time())
            conn.execute(
                "INSERT INTO telegram_backup_runs(run_key,target_key,chat_id,status,started_unix,heartbeat_unix) "
                "VALUES ('stale-run','target','chat','running',?,?)",
                (now - 10_000, now - 10_000),
            )
            conn.execute(
                "INSERT INTO telegram_backup_run_attempts(attempt_key,run_key,started_unix,heartbeat_unix,status) "
                "VALUES ('stale-attempt','stale-run',?,?, 'running')",
                (now - 10_000, now - 10_000),
            )
            conn.execute(
                "INSERT INTO telegram_backup_runs(run_key,target_key,chat_id,status,started_unix,heartbeat_unix) "
                "VALUES ('live-run','target','chat','running',?,?)",
                (now - 10_000, now),
            )
            conn.execute(
                "INSERT INTO telegram_backup_run_attempts(attempt_key,run_key,started_unix,heartbeat_unix,status) "
                "VALUES ('live-attempt','live-run',?,?, 'running')",
                (now - 10_000, now),
            )
            conn.commit()
            conn.close()
            self.assertEqual(
                reap_stale_attempts_command(Namespace(db=str(path), older_than=3600)),
                0,
            )
            conn = sqlite3.connect(path)
            self.assertEqual(
                conn.execute(
                    "SELECT status FROM telegram_backup_run_attempts WHERE attempt_key='stale-attempt'"
                ).fetchone()[0],
                "failed",
            )
            self.assertEqual(
                conn.execute(
                    "SELECT status FROM telegram_backup_run_attempts WHERE attempt_key='live-attempt'"
                ).fetchone()[0],
                "running",
            )
            conn.close()

    def test_diagnostic_database_retention_keeps_hash_suffix_verifiable(self):
        import tgbackup.db.schema as schema

        with tempfile.TemporaryDirectory() as directory:
            conn = sqlite3.connect(Path(directory) / "backup.db")
            ensure_targets_schema(conn)
            with mock.patch.object(schema, "MAX_DATABASE_DIAGNOSTIC_EVENTS", 2):
                for index in range(4):
                    append_diagnostic_event(
                        conn,
                        event_id=f"retained-{index}",
                        event_type="retention-test",
                        component="tests",
                        level="info",
                        operation_id=None,
                        run_key=None,
                        target_key=None,
                        status="completed",
                        details_json=json.dumps({"index": index}),
                        build_revision="test",
                        event_unix=index + 1,
                    )
            self.assertEqual(
                conn.execute("SELECT count(*) FROM telegram_backup_diagnostic_events").fetchone()[0],
                2,
            )
            valid, reason = verify_diagnostic_event_chain(conn)
            self.assertTrue(valid, reason)
            conn.close()

    def test_crash_paths_are_unique(self):
        from tgbackup.diagnostics import _atomic_write_json, _crash_path

        self.assertNotEqual(_crash_path(), _crash_path())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crashes" / "one.json"
            _atomic_write_json(path, {"schema": 1, "status": "failed"})
            self.assertEqual(json.loads(path.read_text())["status"], "failed")
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_default_recorder_and_terminal_operation_are_durable(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"XDG_STATE_HOME": directory}, clear=False):
                recorder = DiagnosticRecorder()
                recorder.emit(DiagnosticEvent(event="default", component="tests"))
                self.assertTrue((Path(directory) / "tgbackman" / "events.jsonl").is_file())
                registry = OperationRegistry(Path(directory) / "tgbackman" / "active.json")
                registry.update("failed", error={"type": "TestFailure"})
                registry.finish("failed", failures=1)
                terminal_files = list((Path(directory) / "tgbackman" / "operations").glob("*.json"))
                self.assertEqual(len(terminal_files), 1)
                self.assertEqual(json.loads(terminal_files[0].read_text())["status"], "failed")

    def test_handled_run_failure_is_persisted_without_diagnostics_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.dict(
                os.environ,
                {"XDG_STATE_HOME": str(root / "state"), "TGBACKMAN_DIAGNOSTICS_FILE": ""},
                clear=False,
            ):
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "tgbackman-backup",
                        "--db",
                        str(root / "backup.db"),
                        "run",
                        "--output",
                        str(root / "output"),
                    ],
                ):
                    self.assertEqual(exporter.main(), 2)
                event_file = root / "state" / "tgbackman" / "events.jsonl"
                self.assertTrue(event_file.is_file())
                self.assertIn("cli_failed", event_file.read_text(encoding="utf-8"))
                terminal_files = list((root / "state" / "tgbackman" / "operations").glob("*.json"))
                self.assertEqual(len(terminal_files), 1)
                self.assertEqual(json.loads(terminal_files[0].read_text())["status"], "failed")

    def test_reaper_respects_the_export_lock(self):
        from tgbackup.progress import ExportLock

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "backup.db"
            conn = sqlite3.connect(path)
            ensure_targets_schema(conn)
            now = int(time.time()) - 10_000
            conn.execute(
                "INSERT INTO telegram_backup_runs(run_key,target_key,chat_id,status,started_unix,heartbeat_unix) "
                "VALUES ('locked-run','target','chat','running',?,?)",
                (now, now),
            )
            conn.execute(
                "INSERT INTO telegram_backup_run_attempts(attempt_key,run_key,started_unix,heartbeat_unix,status) "
                "VALUES ('locked-attempt','locked-run',?,?, 'running')",
                (now, now),
            )
            conn.commit()
            conn.close()
            lock = ExportLock(path.parent / ".backup.db.tgbackman.lock")
            lock.acquire()
            try:
                with self.assertRaises(Exception):
                    reap_stale_attempts_command(Namespace(db=str(path), older_than=3600))
            finally:
                lock.release()

    def test_build_script_watches_source_tree_for_dirty_identity(self):
        build_script = Path(__file__).parents[1] / "tgbackman" / "build.rs"
        text = build_script.read_text(encoding="utf-8")
        self.assertIn('rerun-if-changed=src', text)
        self.assertIn('rerun-if-changed=../.git/refs', text)
        self.assertIn('rerun-if-changed=../src', text)

    def test_build_identity_exposes_dirty_and_runtime_fingerprint(self):
        identity = build_identity()
        self.assertIn("revision", identity)
        self.assertIn("dirty", identity)
        self.assertIn("working_tree_status", identity)
        self.assertIn("python_implementation", identity)


if __name__ == "__main__":
    unittest.main()
