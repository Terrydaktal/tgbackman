"""Connection setup and validation for the shared SQLite database."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, Optional

from ..errors import ExportError


def open_database(
    path: Path,
    *,
    ensure_schema: Callable[[sqlite3.Connection], None],
    setup_database: Optional[Callable[[str], sqlite3.Connection]] = None,
) -> sqlite3.Connection:
    """Open, validate, and migrate a tgbackman database.

    ``ensure_schema`` is injected by the application layer to keep connection
    handling independent from migration DDL. ``setup_database`` is an optional
    migration hook for callers that need a custom database bootstrap policy;
    the canonical schema service is used by default.
    """
    if not path.exists():
        raise ExportError(f"Database does not exist: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    required = {str(row[0]) for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()}
    missing = {"chats", "messages"} - required
    if missing:
        conn.close()
        raise ExportError(f"Database is missing required table(s): {', '.join(sorted(missing))}")
    archival_tables = {"backup_sources", "backup_imports", "backup_import_files", "message_sources"}
    message_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(messages)").fetchall()}
    if not archival_tables.issubset(required) or "raw_payload" not in message_columns:
        conn.close()
        if setup_database is None:
            from .schema import setup_database as canonical_setup_database
            setup_database = canonical_setup_database
        conn = setup_database(str(path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
    ensure_schema(conn)
    return conn
