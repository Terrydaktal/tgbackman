#!/usr/bin/env python3
"""Compatibility launcher for the canonical-database overlap report."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


if __name__ == "__main__":
    from tgbackup.database.overlap_cli import main

    raise SystemExit(main())
