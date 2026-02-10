#!/usr/bin/env python3
"""
backfill_split_export_meta.py

Backfill `.backman_export_meta.json` into split output folders so backman can label them as:
  html_single_chat_export_converted

We only touch directories that look like split outputs:
  <root>/<chat_name>/<START__END>/
where START__END matches backman's timestamp-dir format and contains messages*.html.

Safety:
- Never deletes content.
- Won't overwrite existing `.backman_export_meta.json`.
- Default is dry-run; pass --apply to write files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple


RANGE_DIR_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z__\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$"
)

META_NAME = ".backman_export_meta.json"


def _iter_message_htmls(dirpath: str) -> List[str]:
    try:
        names = os.listdir(dirpath)
    except Exception:
        return []
    out: List[str] = []
    for n in names:
        if n.startswith("messages") and n.endswith(".html"):
            out.append(os.path.join(dirpath, n))
    return out


def _looks_like_single_chat_export_root(dirpath: str) -> bool:
    # Split output roots always include messages*.html and shared assets folders.
    if not _iter_message_htmls(dirpath):
        return False
    try:
        dset = set(os.listdir(dirpath))
    except Exception:
        return False
    return ("css" in dset) and ("js" in dset)


def _write_json_atomic(path: str, obj: Dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Split output root (contains per-chat folders)")
    ap.add_argument("--apply", action="store_true", help="Write meta files (default: dry-run)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    created = 0
    planned = 0
    skipped = 0
    errors = 0

    try:
        chat_dirs = sorted(os.listdir(root))
    except Exception as e:
        print(f"Failed to list root: {e}", file=sys.stderr)
        return 2

    for chat in chat_dirs:
        chat_dir = os.path.join(root, chat)
        if not os.path.isdir(chat_dir):
            continue
        try:
            subdirs = sorted(os.listdir(chat_dir))
        except Exception:
            continue
        for sub in subdirs:
            if not RANGE_DIR_RE.match(sub):
                continue
            export_root = os.path.join(chat_dir, sub)
            if not os.path.isdir(export_root):
                continue
            if not _looks_like_single_chat_export_root(export_root):
                continue
            meta_path = os.path.join(export_root, META_NAME)
            if os.path.exists(meta_path):
                skipped += 1
                continue

            meta = {
                "tool": "backman",
                "kind": "html_single_chat_export_converted",
                "chat_name": chat,
                "converted_from": {
                    "kind": "html_multi_chat_export",
                    "export_root": None,
                    "chat_id": None,
                },
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "note": "Backfilled marker; original multi-export root unknown.",
            }

            try:
                if not args.apply:
                    planned += 1
                    print(f"DRY-RUN create: {meta_path}")
                    continue
                _write_json_atomic(meta_path, meta)
                created += 1
                print(f"created: {meta_path}")
            except Exception as e:
                errors += 1
                print(f"error: {meta_path}: {e}", file=sys.stderr)

    if args.apply:
        print(f"done: created={created} skipped={skipped} errors={errors}")
    else:
        print(f"done: planned={planned} skipped={skipped} errors={errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

