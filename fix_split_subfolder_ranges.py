#!/usr/bin/env python3
"""
fix_split_subfolder_ranges.py

Rename per-chat subfolders in a split multi-chat HTML/JSON/SQLite export output.

This fixes cases where the folder names are unknown/non-standard (e.g. ChatName/unknown__unknown/)
even though the backup data contains messages that allow deriving a real range.

Safety:
- Never deletes content.
- Default is dry-run; pass --apply to actually rename folders.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

VALID_RANGE_DIR_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z__\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$"
)

# Regexes for timestamp parsing (imported from rename_backup.py)
HTML_MSG_TS_RE = re.compile(
    r"(\d{2})\.(\d{2})\.(\d{4}) (\d{2}):(\d{2}):(\d{2})(?:\s+UTC([+-]\d{2}):(\d{2}))?"
)
HTML_TITLE_RE = re.compile(r'title="(\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}(?:\s+UTC[+-]\d{2}:\d{2})?)"')
JSON_DATE_RE = re.compile(r'"date":\s*"([^"]+)"')
JSON_DATE_UNIX_RE = re.compile(r'"date_unixtime":\s*(\d+)')

def parse_html_timestamp(ts_str: str) -> Optional[datetime]:
    try:
        m = HTML_MSG_TS_RE.match(ts_str.strip())
        if m:
            dd, mm, yyyy, hh, mi, ss, tz_h_s, tz_m_s = m.groups()
            tz_h = int(tz_h_s) if tz_h_s else 0
            tz_m = int(tz_m_s) if tz_m_s else 0
            offset = timezone(timedelta(hours=tz_h, minutes=(tz_m if tz_h >= 0 else -tz_m)))
            dt = datetime(int(yyyy), int(mm), int(dd), int(hh), int(mi), int(ss), tzinfo=offset)
            return dt.astimezone(timezone.utc)
    except Exception:
        pass
    return None

def format_utc_iso(dt: datetime) -> str:
    """Format datetime to YYYY-MM-DDTHH-MM-SSZ filesystem safe format."""
    return dt.strftime("%Y-%m-%dT%H-%M-%SZ")

def scan_html_files(dir_path: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    html_files = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            if f.lower().startswith("messages") and f.lower().endswith(".html"):
                html_files.append(os.path.join(root, f))
                
    if not html_files:
        return None, None
        
    min_dt = None
    max_dt = None
    
    for path in sorted(html_files):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                
            matches = HTML_TITLE_RE.findall(content)
            for ts_raw in matches:
                dt = parse_html_timestamp(ts_raw)
                if dt:
                    if min_dt is None or dt < min_dt:
                        min_dt = dt
                    if max_dt is None or dt > max_dt:
                        max_dt = dt
        except Exception:
            pass
            
    return min_dt, max_dt

def scan_json_files(dir_path: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    json_files = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            if f.lower() in ("result.json", "results.json", "export_results.json"):
                json_files.append(os.path.join(root, f))
                
    if not json_files:
        return None, None
        
    min_dt = None
    max_dt = None
    
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                while True:
                    chunk = f.read(16 * 1024 * 1024) # 16MB chunks
                    if not chunk:
                        break
                        
                    unix_matches = JSON_DATE_UNIX_RE.findall(chunk)
                    for ts_str in unix_matches:
                        try:
                            dt = datetime.fromtimestamp(int(ts_str), tz=timezone.utc)
                            if min_dt is None or dt < min_dt:
                                min_dt = dt
                            if max_dt is None or dt > max_dt:
                                max_dt = dt
                        except Exception:
                            pass
                            
                    iso_matches = JSON_DATE_RE.findall(chunk)
                    for iso_str in iso_matches:
                        try:
                            dt = datetime.fromisoformat(iso_str).replace(tzinfo=timezone.utc)
                            if min_dt is None or dt < min_dt:
                                min_dt = dt
                            if max_dt is None or dt > max_dt:
                                max_dt = dt
                        except Exception:
                            pass
        except Exception:
            pass
            
    return min_dt, max_dt

def scan_sqlite_file(dir_path: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    sqlite_files = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            if f.lower() == "database.sqlite":
                sqlite_files.append(os.path.join(root, f))
                
    if not sqlite_files:
        return None, None
        
    path = sqlite_files[0]
    try:
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        cursor.execute("SELECT MIN(time), MAX(time) FROM messages WHERE time IS NOT NULL AND time > 0;")
        min_epoch, max_epoch = cursor.fetchone()
        conn.close()
        
        if min_epoch and max_epoch:
            min_dt = datetime.fromtimestamp(int(min_epoch), tz=timezone.utc)
            max_dt = datetime.fromtimestamp(int(max_epoch), tz=timezone.utc)
            return min_dt, max_dt
    except Exception:
        pass
        
    return None, None

def detect_dates(dir_path: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    # 1. SQLite
    sq_min, sq_max = scan_sqlite_file(dir_path)
    if sq_min and sq_max:
        return sq_min, sq_max
        
    # 2. JSON
    js_min, js_max = scan_json_files(dir_path)
    if js_min and js_max:
        return js_min, js_max
        
    # 3. HTML
    ht_min, ht_max = scan_html_files(dir_path)
    if ht_min and ht_max:
        return ht_min, ht_max
        
    return None, None

def is_backup_root(dir_path: str) -> bool:
    try:
        names = [n.lower() for n in os.listdir(dir_path)]
    except Exception:
        return False
    if any(n.startswith("messages") and n.endswith(".html") for n in names):
        return True
    if any(n in ("result.json", "results.json", "export_results.json") for n in names):
        return True
    if "database.sqlite" in names:
        return True
    return False

def _compute_range_dir(subdir: str) -> Optional[str]:
    min_dt, max_dt = detect_dates(subdir)
    if min_dt and max_dt:
        return f"{format_utc_iso(min_dt)}__{format_utc_iso(max_dt)}"
    return None

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
            # Only touch dirs that look like an export root
            if not is_backup_root(subdir):
                continue
            try:
                new_name = _compute_range_dir(subdir)
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
