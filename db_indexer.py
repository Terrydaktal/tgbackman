#!/usr/bin/env python3
"""Compatibility launcher for :mod:`tgbackup.database.importer`.

The canonical importer is database-facing and lives under ``src/tgbackup``;
this filename remains for existing scripts and tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

if __name__ == "__main__":
    from tgbackup.database.importer import main

    raise SystemExit(main())
else:
    from tgbackup.database.importer import *  # noqa: F401,F403
