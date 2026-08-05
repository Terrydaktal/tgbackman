"""Command-line adapter for the canonical-database overlap report.

The historical implementation remains in ``overlap_report.py`` for
compatibility. This adapter gives it an explicit package entry point and a
normal ``--db`` option instead of requiring a machine-specific environment
variable or default path.
"""

from __future__ import annotations

import argparse
import os
import runpy
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze overlaps and containment between canonical database chat histories."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Canonical SQLite database (defaults to TGBACKMAN_DB or the legacy platform path).",
    )
    args = parser.parse_args(argv)
    if args.db is not None:
        database = args.db.expanduser().resolve()
        if not database.is_file():
            parser.error(f"database does not exist: {database}")
        os.environ["TGBACKMAN_DB"] = str(database)
    runpy.run_module("tgbackup.database.overlap_report", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
