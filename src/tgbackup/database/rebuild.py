#!/usr/bin/env python3
"""Build and verify a new lossless archive DB without touching the live DB."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .importer import index_backup_folder, verify_database_archive


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a side-by-side canonical SQLite archive, embed exact source files, "
            "and verify it. The output must not already exist."
        )
    )
    parser.add_argument("source", help="Root containing legacy HTML/JSON/SQLite backups")
    parser.add_argument("--output", required=True, help="New SQLite database path")
    parser.add_argument(
        "--check-media",
        action="store_true",
        help="Also verify every indexed local media file, size, and SHA-256",
    )
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.is_dir():
        parser.error(f"source directory does not exist: {source}")
    if output.exists() or Path(f"{output}-wal").exists() or Path(f"{output}-shm").exists():
        parser.error(f"refusing to overwrite an existing database or sidecar: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        chats, messages = index_backup_folder(
            str(source), str(output), archive_sources=True
        )
        errors = verify_database_archive(
            str(output), require_archived_sources=True, check_media=args.check_media
        )
    except Exception as exc:
        print(f"Rebuild failed; the live database was untouched: {exc}", file=sys.stderr)
        print(f"Incomplete side-by-side database was preserved for diagnosis: {output}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print("Verification failed; the live database was untouched.", file=sys.stderr)
        return 2
    print(f"Verified side-by-side archive: {output} ({messages} messages, {chats} chats)")
    print("No live database or source export was replaced or deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
