#!/usr/bin/env python3
"""Compatibility launcher for legacy-tree inspection."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tgbackup.legacy_tree.inspect_tree import *  # noqa: F401,F403
from tgbackup.legacy_tree.inspect_tree import main


if __name__ == "__main__":
    raise SystemExit(main())
