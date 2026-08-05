#!/usr/bin/env python3
"""Compatibility launcher for the database-aware range reconciler."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tgbackup.database.range_repair import *  # noqa: F401,F403
from tgbackup.database.range_repair import (
    _apply_db_migration,
    _compute_range_dir,
    _plan_db_migration,
    _rollback_flat_backup,
    _wrap_flat_backup,
)


if __name__ == "__main__":
    from tgbackup.database.range_repair import main

    raise SystemExit(main())
