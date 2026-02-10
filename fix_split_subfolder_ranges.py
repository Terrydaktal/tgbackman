#!/usr/bin/env python3
"""
fix_split_subfolder_ranges.py

Rename per-chat subfolders in a split multi-chat HTML export output.

This fixes cases where the split step produced a folder like:
  ChatName/unknown__unknown/
even though the message HTML contains day separators (or timestamps) that allow
deriving a real range.

Safety:
- Never deletes content.
- Default is dry-run; pass --apply to actually rename folders.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Optional, Tuple


VALID_RANGE_DIR_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z__\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$"
)


def _iter_message_htmls(dirpath: str) -> List[str]:
    try:
        names = os.listdir(dirpath)
    except Exception:
        return []
    out: List[str] = []
    for n in names:
        if n.startswith("messages") and n.endswith(".html"):
            out.append(os.path.join(dirpath, n))
    return sorted(out)


def _compute_range_dir(backman_mod, subdir: str) -> Optional[str]:
    msg_files = _iter_message_htmls(subdir)
    if not msg_files:
        return None
    # _scan_html_message_files returns: (msg_count, first_dt, last_dt, first_src, last_src, basis)
    _, first_dt, last_dt, _, _, _ = backman_mod._scan_html_message_files(msg_files)  # type: ignore[attr-defined]
    return f"{backman_mod._fmt_dt_for_dir(first_dt)}__{backman_mod._fmt_dt_for_dir(last_dt)}"  # type: ignore[attr-defined]


def _should_consider(subdir_name: str, *, all_dirs: bool) -> bool:
    if all_dirs:
        return True
    if "unknown" in subdir_name:
        return True
    if not VALID_RANGE_DIR_RE.match(subdir_name):
        return True
    return False


def _rename_dir(src: str, dst: str, *, apply: bool) -> None:
    if not apply:
        print(f"DRY-RUN rename: {src} -> {dst}")
        return
    os.rename(src, dst)
    print(f"renamed: {src} -> {dst}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Split output root (contains per-chat folders)")
    ap.add_argument("--apply", action="store_true", help="Apply renames (default: dry-run)")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Consider all subfolders (default: only unknown/non-standard names).",
    )
    args = ap.parse_args()

    # Import local backman.py (same directory as this script).
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import backman  # type: ignore

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    chats: List[str]
    try:
        chats = sorted(os.listdir(root))
    except Exception as e:
        print(f"Failed to list root: {e}", file=sys.stderr)
        return 2

    renamed = 0
    planned = 0
    skipped = 0
    errors = 0

    for chat in chats:
        chat_dir = os.path.join(root, chat)
        if not os.path.isdir(chat_dir):
            continue
        try:
            subdirs = sorted(os.listdir(chat_dir))
        except Exception:
            continue
        for sub in subdirs:
            subdir = os.path.join(chat_dir, sub)
            if not os.path.isdir(subdir):
                continue
            if not _should_consider(sub, all_dirs=args.all):
                continue
            # Only touch dirs that look like an export root (contain messages*.html)
            if not _iter_message_htmls(subdir):
                continue
            try:
                new_name = _compute_range_dir(backman, subdir)
            except Exception as e:
                errors += 1
                print(f"error: {subdir}: {e}", file=sys.stderr)
                continue
            if not new_name or "unknown__unknown" == new_name:
                skipped += 1
                continue
            if new_name == sub:
                skipped += 1
                continue
            dst = os.path.join(chat_dir, new_name)
            if os.path.exists(dst):
                skipped += 1
                print(f"skip (dest exists): {subdir} -> {dst}", file=sys.stderr)
                continue
            try:
                if not args.apply:
                    planned += 1
                _rename_dir(subdir, dst, apply=args.apply)
                if args.apply:
                    renamed += 1
            except Exception as e:
                errors += 1
                print(f"error: rename failed: {subdir} -> {dst}: {e}", file=sys.stderr)

    if args.apply:
        print(f"done: renamed={renamed} skipped={skipped} errors={errors}")
    else:
        print(f"done: planned={planned} skipped={skipped} errors={errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
