#!/usr/bin/env python3
"""
rename_backup.py

CLI utility to inspect a Telegram backup directory, find its earliest and
latest message timestamps, and rename the directory to the UTC date-span format:
YYYY-MM-DDTHH-MM-SSZ__YYYY-MM-DDTHH-MM-SSZ
"""

from __future__ import annotations

import os
import sys
import re
import sqlite3
from datetime import datetime, timezone, timedelta

HELP_TEXT = """
NAME
    rename_backup.py - Rename a Telegram backup directory based on its first and last message dates

SYNOPSIS
    python3 rename_backup.py [OPTIONS] <backup_directory>

DESCRIPTION
    Scans a Telegram backup directory (HTML, JSON, or SQLite format) recursively to find the
    dates of the earliest and latest messages. It then renames the backup folder to the format:
    YYYY-MM-DDTHH-MM-SSZ__YYYY-MM-DDTHH-MM-SSZ, corresponding to the UTC message spans.

OPTIONS
    -h, --help
        Show this help text and exit.
    -d, --dry-run
        Find dates and show what the folder would be renamed to, but do not actually rename it.
    -v, --verbose
        Print detailed information during the scanning and inspection process.

OPERATION
    The script auto-detects the backup format inside the target folder:
    1. HTML: Scans messages*.html files for message title timestamps.
    2. JSON: Scans result.json or results.json for message timestamps.
    3. SQLite: Queries database.sqlite for message epochs.
    
    It converts the minimum and maximum parsed timestamps to UTC, formats them using the
    ISO-8601 YYYY-MM-DDTHH-MM-SSZ scheme (replacing colons with dashes for filesystem safety),
    and renames the folder.

EXAMPLES
    python3 rename_backup.py "/media/lewis/1b/Telegram Backup/Ayman/June"
        Rename the folder "June" to its message time span (e.g. "2015-06-05T15-07-27Z__2018-12-14T18-53-43Z").

    python3 rename_backup.py -d "/media/lewis/1b/Telegram Backup/Ayman/June"
        Perform a dry run to inspect the timestamps without executing the rename.

FILES
    find_backup_overlaps.py
        Companion cataloging script in the same directory.

PATHS
    /media/lewis/1b/Telegram Backup/
        Standard storage path for your Telegram backups.

SECURITY NOTES
    This script operates strictly locally on your filesystem. No external network connections
    are established. Always perform a dry run first to verify the detected dates.

EXIT STATUS
    0   Successful completion.
    1   Failure (folder not found, unrecognized format, or no messages found).

AUTHORS
    Antigravity (Google DeepMind Team) & Terrydaktal
"""

# Regexes for timestamp parsing
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

def scan_html_files(dir_path: str, log_fn) -> tuple[Optional[datetime], Optional[datetime]]:
    log_fn("Scanning HTML backup files recursively...")
    html_files = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            if f.lower().startswith("messages") and f.lower().endswith(".html"):
                html_files.append(os.path.join(root, f))
                
    if not html_files:
        return None, None
        
    log_fn(f"Found {len(html_files)} HTML messages files to scan.")
    min_dt = None
    max_dt = None
    
    for path in sorted(html_files):
        log_fn(f"  Reading {os.path.basename(path)}...")
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
        except Exception as e:
            log_fn(f"  Error reading {path}: {e}")
            
    return min_dt, max_dt

def scan_json_files(dir_path: str, log_fn) -> tuple[Optional[datetime], Optional[datetime]]:
    log_fn("Scanning JSON export files...")
    json_files = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            if f.lower() in ("result.json", "results.json", "export_results.json"):
                json_files.append(os.path.join(root, f))
                
    if not json_files:
        return None, None
        
    log_fn(f"Found {len(json_files)} JSON export file(s) to scan.")
    min_dt = None
    max_dt = None
    
    for path in json_files:
        log_fn(f"  Scanning {os.path.basename(path)} footprint using fast regex...")
        try:
            # We read in chunks to prevent loading huge JSON files fully into memory
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                while True:
                    chunk = f.read(16 * 1024 * 1024) # 16MB chunks
                    if not chunk:
                        break
                        
                    # Check for unix timestamp numbers
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
                            
                    # Check for ISO string representations
                    iso_matches = JSON_DATE_RE.findall(chunk)
                    for iso_str in iso_matches:
                        try:
                            # E.g. "2015-06-05T15:07:27"
                            dt = datetime.fromisoformat(iso_str).replace(tzinfo=timezone.utc)
                            if min_dt is None or dt < min_dt:
                                min_dt = dt
                            if max_dt is None or dt > max_dt:
                                max_dt = dt
                        except Exception:
                            pass
        except Exception as e:
            log_fn(f"  Error scanning JSON {path}: {e}")
            
    return min_dt, max_dt

def scan_sqlite_file(dir_path: str, log_fn) -> tuple[Optional[datetime], Optional[datetime]]:
    log_fn("Scanning database.sqlite file...")
    sqlite_files = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            if f.lower() == "database.sqlite":
                sqlite_files.append(os.path.join(root, f))
                
    if not sqlite_files:
        return None, None
        
    path = sqlite_files[0]
    log_fn(f"Found SQLite backup at {path}.")
    
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
    except Exception as e:
        log_fn(f"  Error reading SQLite database {path}: {e}")
        
    return None, None

def main():
    # Handle help
    if "-h" in sys.argv or "--help" in sys.argv:
        print(HELP_TEXT)
        sys.exit(0)
        
    dry_run = "-d" in sys.argv or "--dry-run" in sys.argv
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    
    # Filter positional arguments
    args = [a for a in sys.argv[1:] if a not in ("-d", "--dry-run", "-v", "--verbose")]
    
    if len(args) != 1:
        print("Error: Missing target backup directory path.\n", file=sys.stderr)
        print("Usage: python3 rename_backup.py [OPTIONS] <backup_directory>", file=sys.stderr)
        print("Run with --help for detailed options.", file=sys.stderr)
        sys.exit(1)
        
    target_path = os.path.abspath(args[0])
    
    if not os.path.exists(target_path):
        print(f"Error: Target directory does not exist: {target_path}", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.isdir(target_path):
        print(f"Error: Target path is not a directory: {target_path}", file=sys.stderr)
        sys.exit(1)
        
    def log(msg):
        if verbose or not dry_run:
            print(msg)
            
    print(f"Analyzing backup folder: {target_path}")
    
    # Initialize timestamp accumulators
    min_dt, max_dt = None, None
    
    # 1. Check for SQLite database.sqlite
    sqlite_min, sqlite_max = scan_sqlite_file(target_path, log)
    if sqlite_min and sqlite_max:
        min_dt, max_dt = sqlite_min, sqlite_max
        
    # 2. Check for JSON result.json
    if not min_dt:
        json_min, json_max = scan_json_files(target_path, log)
        if json_min and json_max:
            min_dt, max_dt = json_min, json_max
            
    # 3. Check for HTML messages*.html
    if not min_dt:
        html_min, html_max = scan_html_files(target_path, log)
        if html_min and html_max:
            min_dt, max_dt = html_min, html_max
            
    if not min_dt or not max_dt:
        print("Error: Could not find any valid message timestamps in HTML, JSON, or SQLite formats.", file=sys.stderr)
        sys.exit(1)
        
    # Format new directory name
    min_iso = format_utc_iso(min_dt)
    max_iso = format_utc_iso(max_dt)
    new_name = f"{min_iso}__{max_iso}"
    
    parent_dir = os.path.dirname(target_path)
    old_name = os.path.basename(target_path)
    new_path = os.path.join(parent_dir, new_name)
    
    print("\n-------------------------------------------")
    print(f"Earliest Message Date : {min_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Latest Message Date   : {max_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Proposed Directory Name: {new_name}")
    print("-------------------------------------------")
    
    if old_name == new_name:
        print("The directory is already named correctly!")
        sys.exit(0)
        
    if dry_run:
        print(f"[DRY-RUN] Would rename: '{old_name}' -> '{new_name}'")
        print(f"[DRY-RUN] Target Path:  '{new_path}'")
        sys.exit(0)
        
    # Verify that the destination does not already exist
    if os.path.exists(new_path):
        print(f"Error: A directory with the name '{new_name}' already exists in parent folder!", file=sys.stderr)
        sys.exit(1)
        
    try:
        os.rename(target_path, new_path)
        print(f"Successfully renamed folder:\n  '{old_name}' -> '{new_name}'")
    except Exception as e:
        print(f"Error executing rename: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
