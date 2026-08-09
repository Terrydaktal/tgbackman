"""Find and optionally collapse identical media files using Btrfs reflinks.

The default operation is read-only.  ``--apply`` replaces duplicate files with
copy-on-write reflinks, preserving every per-chat pathname while allowing
Btrfs to share physical extents.  A normal copy is never used as a fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path


def _digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def deduplicate(root: Path, *, apply: bool) -> tuple[int, int, int]:
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"dedupe root is not a directory: {root}")
    if apply:
        filesystem = subprocess.run(
            ["stat", "-f", "-c", "%T", str(root)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
        if filesystem != "btrfs":
            raise RuntimeError(
                f"--apply requires a Btrfs filesystem (detected {filesystem or 'unknown'}); report-only scan is still safe"
            )
    groups: dict[tuple[int, str], list[Path]] = {}
    scanned = 0
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file() or path.name.endswith(".sha256") or ".part-" in path.name:
            continue
        scanned += 1
        groups.setdefault(_digest(path), []).append(path)
    duplicates = 0
    applied = 0
    for (_size, _digest_value), paths in groups.items():
        if len(paths) < 2:
            continue
        canonical = paths[0]
        for duplicate in paths[1:]:
            duplicates += 1
            print(f"duplicate: {duplicate} <- {canonical}")
            if not apply:
                continue
            fd, temporary = tempfile.mkstemp(prefix=f".{duplicate.name}.reflink-", dir=duplicate.parent)
            os.close(fd)
            try:
                os.unlink(temporary)
                subprocess.run(
                    ["cp", "--reflink=always", "--preserve=mode,timestamps", str(canonical), temporary],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                os.replace(temporary, duplicate)
                applied += 1
            except Exception:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
                raise
    return scanned, duplicates, applied


def main() -> int:
    parser = argparse.ArgumentParser(description="Find identical media and optionally replace duplicates with Btrfs reflinks")
    parser.add_argument("root", type=Path, help="Media/archive root to scan")
    parser.add_argument("--apply", action="store_true", help="Replace duplicates with reflinks; default is a report-only scan")
    args = parser.parse_args()
    scanned, duplicates, applied = deduplicate(args.root, apply=args.apply)
    print(f"Scanned {scanned:,} files; duplicate candidates {duplicates:,}; reflinks created {applied:,}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
