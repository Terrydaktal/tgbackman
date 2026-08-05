#!/usr/bin/env python3
"""Compatibility launcher for the database rebuild command."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tgbackup.database.rebuild import main


if __name__ == "__main__":
    raise SystemExit(main())
