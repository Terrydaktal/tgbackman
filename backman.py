#!/usr/bin/env python3
"""
backman.py

Recursively scan a Telegram export folder for JSON exports and report:
- whether each is multi-chat ("Export Telegram data") vs single-chat ("Export chat history")
- nested exports (multiple export roots under a parent)
- message date range (min/max)
- message count
- chat count

Usage:
  python3 backman.py /path/to/backup_folder
  python3 backman.py /path/to/backup_folder --json
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

RESULT_NAMES = ("result.json", "results.json")

EXPORT_DATE_RE = re.compile(
    r"(?:DataExport|ChatExport)[_-](\d{2})[._-](\d{2})[._-](\d{4})"
)

HTML_MULTI_ROOT_NAMES = ("export_results.html",)
HTML_SINGLE_ROOT_NAME = "messages.html"

# Message timestamp for a chat message. Important: forwarded messages include their own
# embedded "date details" spans, which we intentionally do NOT use for chat date ranges.
HTML_MSG_TS_RE = re.compile(
    r'<div class="pull_right date details" title="'
    r'(\d{2})\.(\d{2})\.(\d{4}) (\d{2}):(\d{2}):(\d{2})(?: UTC([+-]\d{2}):(\d{2}))?'
)
HTML_DIV_MESSAGE_MARKER = '<div class="message'
# Day separators are service blocks like: <div class="message service" id="message-94">
HTML_DAY_SEPARATOR_MARKER = '<div class="message service" id="message-'
HTML_DAY_SEP_RE = re.compile(r"^(\d{1,2}) ([A-Za-z]+) (\d{4})$")
HTML_CHAT_NAME_RE = re.compile(r'<div class="text bold">\s*(.*?)\s*</div>', re.DOTALL)

_MONTHS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

def _parse_iso(dt_s: str) -> Optional[datetime]:
    # Telegram commonly: "2018-10-09T19:32:23"
    # Sometimes: "...Z" or "...+00:00"
    s = dt_s.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        d = datetime.fromisoformat(s)
        if d.tzinfo is None:
            # Telegram exports often omit tz; treat as naive UTC-ish for range comparisons.
            d = d.replace(tzinfo=timezone.utc)
        return d
    except Exception:
        return None

def _parse_unixtime(v: Any) -> Optional[datetime]:
    try:
        # sometimes string, sometimes number
        ts = int(v)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        return None

@dataclass
class ExportReport:
    export_root: str                 # directory containing result.json
    result_json: str                 # full path to result.json (or HTML entrypoint)
    fmt: str                         # "json" | "html"
    kind: str                        # "multi_chat_export" | "single_chat_export" | "html_multi_chat_export" | "html_single_chat_export" | "unknown"
    top_level_keys: List[str]
    inferred_export_date: Optional[str]  # YYYY-MM-DD if folder name matches common pattern
    chats_backed_up: Optional[int]
    messages_backed_up: Optional[int]
    first_message_utc: Optional[str]     # ISO
    last_message_utc: Optional[str]      # ISO
    first_message_source: Optional[str]  # file path (HTML) or JSON file
    last_message_source: Optional[str]   # file path (HTML) or JSON file
    date_range_basis: Optional[str]      # "message_timestamps" | "day_separators" | None

def _format_bytes(n: int) -> str:
    # IEC-ish, compact.
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    v = float(n)
    for u in units:
        if v < 1024 or u == units[-1]:
            if u == "B":
                return f"{int(v)}B"
            if v >= 100:
                return f"{v:0.0f}{u}"
            if v >= 10:
                return f"{v:0.1f}{u}"
            return f"{v:0.2f}{u}"
        v /= 1024.0
    return f"{n}B"

def _parse_human_bytes(tok: str) -> Optional[int]:
    """
    Parses tokens like: 1234, 49.4MiB, 1.09GiB, 726MB, 177G, 4B.
    Returns bytes or None.
    """
    s = tok.strip()
    if not s:
        return None
    if re.fullmatch(r"\d+", s):
        try:
            return int(s)
        except Exception:
            return None

    m = re.fullmatch(r"(\d+(?:\.\d+)?)([A-Za-z]+)", s)
    if not m:
        return None
    num_s, unit = m.group(1), m.group(2)
    try:
        num = float(num_s)
    except Exception:
        return None

    unit = unit.strip()
    # Normalize common dust/du-ish units.
    unit_map_1024 = {
        "B": 1,
        "K": 1024,
        "KB": 1000,
        "KiB": 1024,
        "M": 1024**2,
        "MB": 1000**2,
        "MiB": 1024**2,
        "G": 1024**3,
        "GB": 1000**3,
        "GiB": 1024**3,
        "T": 1024**4,
        "TB": 1000**4,
        "TiB": 1024**4,
        "P": 1024**5,
        "PB": 1000**5,
        "PiB": 1024**5,
    }
    mult = unit_map_1024.get(unit)
    if mult is None:
        # Some tools use lowercase suffixes.
        mult = unit_map_1024.get(unit.capitalize())
    if mult is None:
        return None
    try:
        return int(num * mult)
    except Exception:
        return None

def _parse_dust_total(stdout: str, path: str) -> Optional[int]:
    """
    Try to extract the total size for `path` from dust's output.
    Prefer lines that contain the (absolute) path.
    """
    p = os.path.abspath(path)
    lines = [ln.strip("\n") for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        return None

    # Prefer exact path matches in-line (dust typically prints the input path).
    candidates = [ln for ln in lines if p in ln]
    # Fallback: last line is often the total for -d 0.
    if not candidates:
        candidates = [lines[-1]]

    for ln in candidates:
        # Strip simple tree glyphs / bars to make tokenization easier.
        cleaned = ln.replace("│", " ").replace("█", " ").replace("▉", " ").replace("▊", " ").replace("▌", " ")
        parts = [x for x in cleaned.split() if x]
        # Find the first token that parses as a size.
        for tok in parts:
            b = _parse_human_bytes(tok)
            if b is not None:
                return b
    return None

def _dir_size_bytes(path: str) -> Tuple[Optional[int], str]:
    """
    Returns (bytes, tool_used). Prefers `dust` (fast) when available.

    Note: On some systems `dust` is installed as a snap, which may not work in
    sandboxed environments; we fallback to `du -sb` in that case.
    """
    p = os.path.abspath(path)

    dust = shutil.which("dust")
    if dust:
        # `-b` forces raw bytes, making parsing stable; depth 0 = only total.
        try:
            for args in ([dust, "-b", "-d", "0", "--no-color", p], [dust, "-b", "-d", "0", p]):
                proc = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if proc.returncode != 0:
                    continue
                b = _parse_dust_total(proc.stdout, p)
                if b is not None:
                    # Sanity: if dust returns tiny sizes for a directory that clearly has entries,
                    # it's likely parsing/behavior mismatch; fall back to du for correctness.
                    try:
                        has_entries = any(True for _ in os.scandir(p))
                    except Exception:
                        has_entries = False
                    # Some dust builds (notably snap-confined) can report only the directory
                    # entry size (4KiB) when they can't traverse contents. Treat that as
                    # suspicious when the directory has entries.
                    if has_entries and b <= 16 * 1024:
                        break
                    return b, "dust"
        except Exception:
            pass

    # Fallback: GNU coreutils du.
    try:
        proc = subprocess.run(
            ["du", "-sb", p],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            m = re.search(r"^\s*(\d+)\s+", proc.stdout)
            if m:
                return int(m.group(1)), "du"
    except Exception:
        pass

    return None, "unknown"

def _is_ancestor_dir(parent: str, child: str) -> bool:
    parent = os.path.abspath(parent)
    child = os.path.abspath(child)
    if parent == child:
        return True
    try:
        rel = os.path.relpath(child, parent)
    except Exception:
        return False
    return rel != os.pardir and not rel.startswith(os.pardir + os.sep)

def _render_export_tree(root: str, reports: List[ExportReport]) -> str:
    root = os.path.abspath(root)

    # Nodes are: input root + each export root.
    nodes: List[str] = [root]
    for r in reports:
        p = os.path.abspath(r.export_root)
        if p not in nodes:
            nodes.append(p)

    # Map export_root -> report for labels (root may or may not be in reports).
    report_by_root: Dict[str, ExportReport] = {os.path.abspath(r.export_root): r for r in reports}

    # Build parent pointers: closest containing node.
    parents: Dict[str, Optional[str]] = {n: None for n in nodes}
    for n in nodes:
        if n == root:
            continue
        best: Optional[str] = None
        for cand in nodes:
            if cand == n:
                continue
            if _is_ancestor_dir(cand, n):
                if best is None or len(cand) > len(best):
                    best = cand
        parents[n] = best or root

    children: Dict[str, List[str]] = {n: [] for n in nodes}
    for n, p in parents.items():
        if p and n != root:
            children[p].append(n)
    for k in children:
        children[k].sort()

    # Compute sizes (total sizes; parents include children).
    size_cache: Dict[str, Tuple[Optional[int], str]] = {}
    tools: Set[str] = set()
    for n in nodes:
        size_cache[n] = _dir_size_bytes(n)
        tools.add(size_cache[n][1])

    def label(n: str) -> str:
        rel = os.path.relpath(n, root)
        name = "." if rel == "." else rel
        r = report_by_root.get(n)
        if r:
            return f"{name} [{r.kind}]"
        return name

    def size_s(n: str) -> str:
        b, _tool = size_cache[n]
        return "?" if b is None else _format_bytes(b)

    lines: List[str] = []
    lines.append("Export tree (sizes are total directory sizes; parents include children):")
    lines.append(f"{size_s(root):>8}  {label(root)}")

    def walk(parent: str, prefix: str):
        kids = children.get(parent, [])
        for i, k in enumerate(kids):
            last = i == (len(kids) - 1)
            branch = "└── " if last else "├── "
            lines.append(f"{prefix}{branch}{size_s(k):>8}  {label(k)}")
            walk(k, prefix + ("    " if last else "│   "))

    walk(root, "")

    tool_note = ", ".join(sorted(t for t in tools if t != "unknown"))
    if tool_note:
        lines.append(f"(size tool: {tool_note})")
    return "\n".join(lines)

def _rewrite_chat_html_for_standalone(chat_root: str, chat_id: Optional[str] = None) -> None:
    """
    Make a chat folder self-contained by fixing references to shared assets that used to
    live two levels up in a multi-export (../../css, ../../js, ../../images, ...).
    """
    for fn in os.listdir(chat_root):
        if not (fn.startswith("messages") and fn.endswith(".html")):
            continue
        p = os.path.join(chat_root, fn)
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                s = f.read()
        except Exception:
            continue

        # Assets: multi-export chats use ../../css and ../../js.
        s2 = s
        s2 = s2.replace('href="../../css/', 'href="css/')
        s2 = s2.replace('href="../css/', 'href="css/')
        s2 = s2.replace('src="../../js/', 'src="js/')
        s2 = s2.replace('src="../js/', 'src="js/')
        s2 = s2.replace('src="../../images/', 'src="images/')
        s2 = s2.replace('href="../../images/', 'href="images/')
        s2 = s2.replace('src="../../profile_pictures/', 'src="profile_pictures/')
        s2 = s2.replace('href="../../profile_pictures/', 'href="profile_pictures/')

        # Back-links to the multi-export chat list. Point to the first page in this folder.
        s2 = s2.replace('href="../../lists/chats.html"', 'href="messages.html"')
        s2 = s2.replace('href="../../lists/chats.html#allow_back"', 'href="messages.html"')
        s2 = s2.replace('href="../lists/chats.html"', 'href="messages.html"')
        s2 = s2.replace('href="../lists/chats.html#allow_back"', 'href="messages.html"')

        # Media references: multi-export chats often reference their own media via
        # ../../chats/chat_XXX/<media> even when the HTML lives in chats/chat_XXX/.
        # Make those paths local so the chat folder is portable.
        if chat_id:
            s2 = s2.replace(f'../../chats/{chat_id}/', '')
            s2 = s2.replace(f'../chats/{chat_id}/', '')

        if s2 != s:
            with open(p, "w", encoding="utf-8") as f:
                f.write(s2)

def _sanitize_dirname(name: str, max_len: int = 120) -> str:
    # Keep readable names, but avoid path separators and control chars.
    s = name.strip()
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[\x00-\x1f\x7f]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        s = "unknown_chat"
    # Avoid trailing dots/spaces (Windows-hostile).
    s = s.rstrip(" .")
    if len(s) > max_len:
        s = s[:max_len].rstrip(" .")
    return s or "unknown_chat"

def _extract_chat_name_from_messages_html(messages_html_path: str) -> Optional[str]:
    try:
        with open(messages_html_path, "r", encoding="utf-8", errors="ignore") as f:
            # Read a bounded chunk; header is near the top.
            chunk = f.read(64 * 1024)
    except Exception:
        return None

    m = HTML_CHAT_NAME_RE.search(chunk)
    if not m:
        return None
    raw = m.group(1)
    # Remove nested tags if any (rare, but keep the plain text).
    raw = re.sub(r"<[^>]+>", "", raw)
    raw = html.unescape(raw)
    raw = raw.strip()
    return raw or None

def _fmt_dt_for_dir(d: Optional[datetime]) -> str:
    if not d:
        return "unknown"
    # Use UTC and filesystem-safe format (no colons).
    du = d.astimezone(timezone.utc)
    return du.strftime("%Y-%m-%dT%H-%M-%SZ")

def _scan_first_last_message_ts(paths: Iterable[str]) -> Tuple[Optional[datetime], Optional[datetime]]:
    first: Optional[datetime] = None
    last: Optional[datetime] = None
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    for m in HTML_MSG_TS_RE.finditer(line):
                        dd = int(m.group(1))
                        mm = int(m.group(2))
                        yyyy = int(m.group(3))
                        hh = int(m.group(4))
                        mi = int(m.group(5))
                        ss = int(m.group(6))
                        tz_h_s = m.group(7)
                        tz_m_s = m.group(8)
                        tz_h = int(tz_h_s) if tz_h_s else 0
                        tz_m = int(tz_m_s) if tz_m_s else 0
                        try:
                            offset = timezone(timedelta(hours=tz_h, minutes=(tz_m if tz_h >= 0 else -tz_m)))
                        except Exception:
                            offset = timezone.utc
                        d2 = datetime(yyyy, mm, dd, hh, mi, ss, tzinfo=offset).astimezone(timezone.utc)
                        if first is None or d2 < first:
                            first = d2
                        if last is None or d2 > last:
                            last = d2
        except Exception:
            continue
    return first, last

def _copy_tree(src: str, dst: str) -> None:
    # copytree is fast enough and preserves metadata via copy2
    shutil.copytree(src, dst, dirs_exist_ok=True, copy_function=shutil.copy2)

def split_multi_html_export_to_single_chat_exports(
    export_root: str,
    out_root: Optional[str] = None,
    only_chats: Optional[Set[str]] = None,
    dry_run: bool = False,
) -> str:
    """
    Turn a multi-chat HTML export into per-chat self-contained HTML exports.
    Output is created under `out_root` (or default under parent of export_root).
    """
    export_root = os.path.abspath(export_root)

    if out_root is None:
        parent = os.path.dirname(export_root.rstrip(os.sep))
        base = os.path.basename(export_root.rstrip(os.sep))
        out_root = os.path.join(parent, f"{base}_single_chats")
    out_root = os.path.abspath(out_root)

    if not os.path.isfile(os.path.join(export_root, "export_results.html")):
        raise ValueError(f"Not a multi-chat HTML export root (missing export_results.html): {export_root}")
    chats_dir = os.path.join(export_root, "chats")
    if not os.path.isdir(chats_dir):
        raise ValueError(f"Not a multi-chat HTML export root (missing chats/): {export_root}")

    if os.path.exists(out_root):
        raise FileExistsError(f"Output directory already exists: {out_root}")

    # Shared assets needed for HTML rendering.
    shared_dirs = ["css", "js", "images", "profile_pictures"]

    # First pass: determine stable, unique destination names per chat.
    chat_names = sorted(
        name for name in os.listdir(chats_dir)
        if name.startswith("chat_") and os.path.isdir(os.path.join(chats_dir, name))
    )
    if only_chats:
        want = set(only_chats)
        chat_names = [n for n in chat_names if n in want]

    dst_name_by_chat: Dict[str, str] = {}
    used: Set[str] = set()
    for chat_id in chat_names:
        src_messages = os.path.join(chats_dir, chat_id, "messages.html")
        title = _extract_chat_name_from_messages_html(src_messages) or chat_id
        base = _sanitize_dirname(title)
        dst = base
        if dst in used:
            dst = f"{base} ({chat_id})"
        used.add(dst)
        dst_name_by_chat[chat_id] = dst

    if dry_run:
        return out_root

    os.makedirs(out_root, exist_ok=False)

    # Copy each chat folder into its own export root and add shared assets.
    for i, chat_id in enumerate(chat_names, start=1):
        src_chat = os.path.join(chats_dir, chat_id)
        dst_chat_parent = os.path.join(out_root, dst_name_by_chat[chat_id])

        # Compute date range from the source chat (first/last message in this backup).
        src_msg_files = sorted(
            os.path.join(src_chat, fn)
            for fn in os.listdir(src_chat)
            if fn.startswith("messages") and fn.endswith(".html")
        )
        first_dt, last_dt = _scan_first_last_message_ts(src_msg_files)
        range_dir = f"{_fmt_dt_for_dir(first_dt)}__{_fmt_dt_for_dir(last_dt)}"
        dst_chat = os.path.join(dst_chat_parent, range_dir)

        os.makedirs(dst_chat, exist_ok=True)

        # Copy chat-local content (messages + media folders).
        for name in os.listdir(src_chat):
            sp = os.path.join(src_chat, name)
            dp = os.path.join(dst_chat, name)
            if os.path.isdir(sp):
                _copy_tree(sp, dp)
            else:
                shutil.copy2(sp, dp)

        # Copy shared assets into the chat root.
        for d in shared_dirs:
            sp = os.path.join(export_root, d)
            if os.path.isdir(sp):
                _copy_tree(sp, os.path.join(dst_chat, d))

        _rewrite_chat_html_for_standalone(dst_chat, chat_id=chat_id)

        # Progress every so often for large exports.
        if i == 1 or i % 25 == 0 or i == len(chat_names):
            print(f"[split] {i}/{len(chat_names)}: {chat_id} -> {dst_name_by_chat[chat_id]}/{range_dir}", file=sys.stderr)

    return out_root

def find_result_jsons(root: str) -> List[str]:
    hits: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn in RESULT_NAMES:
                hits.append(os.path.join(dirpath, fn))
    return sorted(set(hits))

def infer_export_date_from_path(p: str) -> Optional[str]:
    # Looks for DataExport_dd_mm_yyyy or ChatExport_dd_mm_yyyy in any path segment
    m = EXPORT_DATE_RE.search(p)
    if not m:
        return None
    dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
    try:
        d = datetime(int(yyyy), int(mm), int(dd), tzinfo=timezone.utc)
        return d.date().isoformat()
    except Exception:
        return None

def try_import_ijson():
    try:
        import ijson  # type: ignore
        return ijson
    except Exception:
        return None

def sniff_top_level_keys_ijson(ijson_mod, path: str, max_events: int = 2000) -> Set[str]:
    keys: Set[str] = set()
    with open(path, "rb") as f:
        for i, (prefix, event, value) in enumerate(ijson_mod.parse(f)):
            if prefix == "" and event == "map_key":
                keys.add(str(value))
                # Early exit once we have enough to classify
                if "chats" in keys or "messages" in keys:
                    if len(keys) >= 3:
                        break
            if i >= max_events:
                break
    return keys

def inspect_streaming_ijson(path: str) -> ExportReport:
    ijson_mod = try_import_ijson()
    if not ijson_mod:
        return inspect_via_json_load(path)

    top_keys = sniff_top_level_keys_ijson(ijson_mod, path)
    kind = "unknown"
    # Classification heuristic:
    if "chats" in top_keys:
        kind = "multi_chat_export"
    elif "messages" in top_keys:
        kind = "single_chat_export"

    chat_count: Optional[int] = 0 if kind == "multi_chat_export" else (1 if kind == "single_chat_export" else None)
    msg_count: int = 0
    first_dt: Optional[datetime] = None
    last_dt: Optional[datetime] = None

    # One full streaming pass: count chats, count messages, track min/max time.
    with open(path, "rb") as f:
        for prefix, event, value in ijson_mod.parse(f):
            # Count chats (only meaningful for multi-chat exports)
            if kind == "multi_chat_export":
                # Chat objects: chats.list.item.id (distinct from message IDs which are chats.list.item.messages.item.id)
                if prefix == "chats.list.item.id" and event in ("number", "string"):
                    chat_count = (chat_count or 0) + 1

            # Count messages by message id field (avoid reply_to_message_id etc.)
            if prefix.endswith(".messages.item.id") or prefix == "messages.item.id":
                if event in ("number", "string"):
                    msg_count += 1

            # Prefer unixtime if present
            if prefix.endswith(".messages.item.date_unixtime") or prefix == "messages.item.date_unixtime":
                if event in ("number", "string"):
                    d = _parse_unixtime(value)
                    if d:
                        if first_dt is None or d < first_dt:
                            first_dt = d
                        if last_dt is None or d > last_dt:
                            last_dt = d

            # Fallback to ISO date strings
            if prefix.endswith(".messages.item.date") or prefix == "messages.item.date":
                if event == "string":
                    d = _parse_iso(str(value))
                    if d:
                        if first_dt is None or d < first_dt:
                            first_dt = d
                        if last_dt is None or d > last_dt:
                            last_dt = d

    export_root = os.path.dirname(path)
    return ExportReport(
        export_root=export_root,
        result_json=path,
        fmt="json",
        kind=kind,
        top_level_keys=sorted(top_keys),
        inferred_export_date=infer_export_date_from_path(path),
        chats_backed_up=chat_count if kind == "multi_chat_export" else (1 if kind == "single_chat_export" else None),
        messages_backed_up=msg_count,
        first_message_utc=first_dt.isoformat() if first_dt else None,
        last_message_utc=last_dt.isoformat() if last_dt else None,
        first_message_source=path if first_dt else None,
        last_message_source=path if last_dt else None,
        date_range_basis="message_timestamps" if (first_dt or last_dt) else None,
    )

def inspect_via_json_load(path: str) -> ExportReport:
    # Fallback for smaller exports (loads entire file).
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    top_keys = sorted(list(data.keys())) if isinstance(data, dict) else []
    kind = "unknown"
    chats = None
    messages = None

    if isinstance(data, dict) and "chats" in data and isinstance(data["chats"], dict) and "list" in data["chats"]:
        kind = "multi_chat_export"
        chats = data["chats"]["list"]
    elif isinstance(data, dict) and "messages" in data and isinstance(data["messages"], list):
        kind = "single_chat_export"
        messages = data["messages"]

    chat_count: Optional[int] = None
    msg_count: Optional[int] = None
    first_dt: Optional[datetime] = None
    last_dt: Optional[datetime] = None

    def upd(d: Optional[datetime]):
        nonlocal first_dt, last_dt
        if not d:
            return
        if first_dt is None or d < first_dt:
            first_dt = d
        if last_dt is None or d > last_dt:
            last_dt = d

    if kind == "multi_chat_export" and isinstance(chats, list):
        chat_count = len(chats)
        msg_count = 0
        for c in chats:
            if not isinstance(c, dict):
                continue
            for m in c.get("messages", []) or []:
                if not isinstance(m, dict):
                    continue
                msg_count += 1
                if "date_unixtime" in m:
                    upd(_parse_unixtime(m["date_unixtime"]))
                if "date" in m:
                    upd(_parse_iso(str(m["date"])))
    elif kind == "single_chat_export" and isinstance(messages, list):
        chat_count = 1
        msg_count = 0
        for m in messages:
            if not isinstance(m, dict):
                continue
            msg_count += 1
            if "date_unixtime" in m:
                upd(_parse_unixtime(m["date_unixtime"]))
            if "date" in m:
                upd(_parse_iso(str(m["date"])))

    export_root = os.path.dirname(path)
    return ExportReport(
        export_root=export_root,
        result_json=path,
        fmt="json",
        kind=kind,
        top_level_keys=top_keys,
        inferred_export_date=infer_export_date_from_path(path),
        chats_backed_up=chat_count,
        messages_backed_up=msg_count,
        first_message_utc=first_dt.isoformat() if first_dt else None,
        last_message_utc=last_dt.isoformat() if last_dt else None,
        first_message_source=path if first_dt else None,
        last_message_source=path if last_dt else None,
        date_range_basis="message_timestamps" if (first_dt or last_dt) else None,
    )

def _upd_range(first_dt: Optional[datetime], last_dt: Optional[datetime], d: Optional[datetime]) -> Tuple[Optional[datetime], Optional[datetime]]:
    if not d:
        return first_dt, last_dt
    if first_dt is None or d < first_dt:
        first_dt = d
    if last_dt is None or d > last_dt:
        last_dt = d
    return first_dt, last_dt

def _parse_html_day_sep(line: str) -> Optional[datetime]:
    m = HTML_DAY_SEP_RE.match(line.strip())
    if not m:
        return None
    try:
        dd = int(m.group(1))
        mon_s = m.group(2)
        yyyy = int(m.group(3))
        mm = _MONTHS.get(mon_s)
        if not mm:
            return None
        return datetime(yyyy, mm, dd, 0, 0, 0, tzinfo=timezone.utc)
    except Exception:
        return None

def _scan_html_message_files(
    paths: Iterable[str],
) -> Tuple[
    int,
    Optional[datetime],
    Optional[datetime],
    Optional[str],
    Optional[str],
    Optional[str],
]:
    msg_count = 0
    first_msg_dt: Optional[datetime] = None
    last_msg_dt: Optional[datetime] = None
    first_msg_src: Optional[str] = None
    last_msg_src: Optional[str] = None

    first_day_dt: Optional[datetime] = None
    last_day_dt: Optional[datetime] = None

    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                expect_day_sep = False
                for line in f:
                    # Telegram's own per-chat count in lists/chats.html matches the number of
                    # message blocks, excluding day separators. Day separators are always
                    # "message service" with an id like message-<n>, while real messages may
                    # have ids with or without '-' (e.g. negative ids in some exports).
                    msg_count += line.count(HTML_DIV_MESSAGE_MARKER) - line.count(HTML_DAY_SEPARATOR_MARKER)

                    if expect_day_sep:
                        expect_day_sep = False
                        d = _parse_html_day_sep(line)
                        first_day_dt, last_day_dt = _upd_range(first_day_dt, last_day_dt, d)
                    if 'class="body details"' in line:
                        # Telegram's day separator is typically the next line after this div.
                        expect_day_sep = True

                    for m in HTML_MSG_TS_RE.finditer(line):
                        dd = int(m.group(1))
                        mm = int(m.group(2))
                        yyyy = int(m.group(3))
                        hh = int(m.group(4))
                        mi = int(m.group(5))
                        ss = int(m.group(6))
                        tz_h_s = m.group(7)
                        tz_m_s = m.group(8)
                        tz_h = int(tz_h_s) if tz_h_s else 0
                        tz_m = int(tz_m_s) if tz_m_s else 0
                        try:
                            offset = timezone(timedelta(hours=tz_h, minutes=(tz_m if tz_h >= 0 else -tz_m)))
                        except Exception:
                            offset = timezone.utc
                        d2 = datetime(yyyy, mm, dd, hh, mi, ss, tzinfo=offset).astimezone(timezone.utc)
                        if first_msg_dt is None or d2 < first_msg_dt:
                            first_msg_dt = d2
                            first_msg_src = p
                        if last_msg_dt is None or d2 > last_msg_dt:
                            last_msg_dt = d2
                            last_msg_src = p
        except Exception:
            continue

    # Prefer real message timestamps for "first message" semantics. Day separators are
    # coarse and can skew to midnight even if the first message is later that day.
    if first_msg_dt or last_msg_dt:
        return msg_count, first_msg_dt, last_msg_dt, first_msg_src, last_msg_src, "message_timestamps"
    return msg_count, first_day_dt, last_day_dt, None, None, ("day_separators" if (first_day_dt or last_day_dt) else None)

def find_html_export_roots(root: str) -> List[str]:
    roots: Set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        fset = set(filenames)
        dset = set(dirnames)

        if any(n in fset for n in HTML_MULTI_ROOT_NAMES):
            roots.add(dirpath)
            continue

        # Single-chat HTML exports typically have messages.html plus css/ + js/ folders.
        if HTML_SINGLE_ROOT_NAME in fset and "css" in dset and "js" in dset:
            roots.add(dirpath)
            continue
    return sorted(roots)

def _iter_html_message_files(export_root: str, kind: str) -> Iterable[str]:
    if kind == "html_multi_chat_export":
        chats_dir = os.path.join(export_root, "chats")
        if not os.path.isdir(chats_dir):
            return []
        # Important: only include the multi-export chat folders (chat_001, chat_002, ...).
        # Some users may also have nested single-chat exports under chats/ (e.g. chats/alex/)
        # which should not be counted as part of the multi-export.
        out: List[str] = []
        for name in sorted(os.listdir(chats_dir)):
            if not name.startswith("chat_"):
                continue
            chat_dir = os.path.join(chats_dir, name)
            if not os.path.isdir(chat_dir):
                continue
            try:
                for fn in sorted(os.listdir(chat_dir)):
                    if fn.startswith("messages") and fn.endswith(".html"):
                        out.append(os.path.join(chat_dir, fn))
            except Exception:
                continue
        return out

    # html_single_chat_export: message pages live in export_root itself.
    out2: List[str] = []
    for fn in os.listdir(export_root):
        if fn.startswith("messages") and fn.endswith(".html"):
            out2.append(os.path.join(export_root, fn))
    return sorted(out2)

def inspect_html_export(export_root: str) -> ExportReport:
    export_root = os.path.abspath(export_root)
    entrypoint = None
    for n in HTML_MULTI_ROOT_NAMES:
        p = os.path.join(export_root, n)
        if os.path.exists(p):
            entrypoint = p
            break
    kind = "html_multi_chat_export" if entrypoint else "html_single_chat_export"
    if entrypoint is None:
        entrypoint = os.path.join(export_root, HTML_SINGLE_ROOT_NAME)

    chat_count: Optional[int]
    if kind == "html_multi_chat_export":
        chats_dir = os.path.join(export_root, "chats")
        if os.path.isdir(chats_dir):
            chat_count = sum(
                1
                for name in os.listdir(chats_dir)
                if name.startswith("chat_") and os.path.isdir(os.path.join(chats_dir, name))
            )
        else:
            chat_count = None
    else:
        chat_count = 1

    msg_files = list(_iter_html_message_files(export_root, kind))
    msg_count, first_dt, last_dt, first_src, last_src, basis = _scan_html_message_files(msg_files)

    return ExportReport(
        export_root=export_root,
        result_json=entrypoint,
        fmt="html",
        kind=kind,
        top_level_keys=[],
        inferred_export_date=infer_export_date_from_path(export_root),
        chats_backed_up=chat_count,
        messages_backed_up=msg_count,
        first_message_utc=first_dt.isoformat() if first_dt else None,
        last_message_utc=last_dt.isoformat() if last_dt else None,
        first_message_source=first_src,
        last_message_source=last_src,
        date_range_basis=basis,
    )

def summarize_html_export(export_root: str) -> ExportReport:
    """
    Fast classification for HTML exports without scanning message bodies.
    """
    export_root = os.path.abspath(export_root)
    entrypoint = None
    for n in HTML_MULTI_ROOT_NAMES:
        p = os.path.join(export_root, n)
        if os.path.exists(p):
            entrypoint = p
            break
    kind = "html_multi_chat_export" if entrypoint else "html_single_chat_export"
    if entrypoint is None:
        entrypoint = os.path.join(export_root, HTML_SINGLE_ROOT_NAME)

    chat_count: Optional[int]
    if kind == "html_multi_chat_export":
        chats_dir = os.path.join(export_root, "chats")
        if os.path.isdir(chats_dir):
            chat_count = sum(
                1
                for name in os.listdir(chats_dir)
                if name.startswith("chat_") and os.path.isdir(os.path.join(chats_dir, name))
            )
        else:
            chat_count = None
    else:
        chat_count = 1

    return ExportReport(
        export_root=export_root,
        result_json=entrypoint,
        fmt="html",
        kind=kind,
        top_level_keys=[],
        inferred_export_date=infer_export_date_from_path(export_root),
        chats_backed_up=chat_count,
        messages_backed_up=None,
        first_message_utc=None,
        last_message_utc=None,
        first_message_source=None,
        last_message_source=None,
        date_range_basis=None,
    )

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Telegram export folder (will be scanned recursively)")
    ap.add_argument("--json", action="store_true", help="Output JSON (machine-readable)")
    ap.add_argument(
        "--no-inspect",
        action="store_true",
        help="Skip scanning message bodies for counts/date ranges (much faster).",
    )
    ap.add_argument(
        "--split-multi-html",
        action="store_true",
        help="Convert a multi-chat HTML export into per-chat self-contained exports (copies data).",
    )
    ap.add_argument(
        "--split-out",
        default=None,
        help="Output directory for --split-multi-html (default: <parent>/<basename>_single_chats).",
    )
    ap.add_argument(
        "--split-chat",
        action="append",
        default=[],
        help="Limit --split-multi-html to specific chat folders (repeatable), e.g. --split-chat chat_001",
    )
    ap.add_argument(
        "--split-dry-run",
        action="store_true",
        help="With --split-multi-html: only print planned output directory and exit.",
    )
    args = ap.parse_args()

    root = os.path.abspath(args.path)
    if not os.path.exists(root):
        print(f"Path does not exist: {root}", file=sys.stderr)
        return 2

    if args.split_multi_html:
        only = set(args.split_chat) if args.split_chat else None
        try:
            out = split_multi_html_export_to_single_chat_exports(
                root,
                out_root=args.split_out,
                only_chats=only,
                dry_run=args.split_dry_run,
            )
        except Exception as e:
            print(f"Split failed: {e}", file=sys.stderr)
            return 3
        if args.split_dry_run:
            print(out)
            return 0
        print(f"Wrote per-chat exports to: {out}")
        return 0

    reports: List[ExportReport] = []
    json_hits = find_result_jsons(root)
    if json_hits:
        if args.no_inspect:
            for p in json_hits:
                export_root = os.path.dirname(p)
                reports.append(
                    ExportReport(
                        export_root=export_root,
                        result_json=p,
                        fmt="json",
                        kind="unknown",
                        top_level_keys=[],
                        inferred_export_date=infer_export_date_from_path(p),
                        chats_backed_up=None,
                        messages_backed_up=None,
                        first_message_utc=None,
                        last_message_utc=None,
                        first_message_source=None,
                        last_message_source=None,
                        date_range_basis=None,
                    )
                )
        else:
            reports.extend(inspect_streaming_ijson(p) for p in json_hits)

    html_roots = find_html_export_roots(root)
    if html_roots:
        if args.no_inspect:
            reports.extend(summarize_html_export(r) for r in html_roots)
        else:
            reports.extend(inspect_html_export(r) for r in html_roots)

    if not reports:
        print("No Telegram exports found under the given path.", file=sys.stderr)
        print("Expected either:", file=sys.stderr)
        print("- JSON exports: result.json/results.json", file=sys.stderr)
        print("- HTML exports: export_results.html (multi-chat) or messages.html + css/ + js/ (single-chat)", file=sys.stderr)
        return 1

    # Nested export detection: multiple export roots under the input root
    # (This is just a count + listing; you can extend grouping if you want.)
    if args.json:
        print(json.dumps([asdict(r) for r in reports], indent=2, ensure_ascii=False))
    else:
        print(f"Found {len(reports)} Telegram export(s) under: {root}\n")
        for r in reports:
            print(f"- Export root: {r.export_root}")
            if r.fmt == "json":
                print(f"  result.json: {r.result_json}")
            else:
                print(f"  html entrypoint: {r.result_json}")
            print(f"  format: {r.fmt}")
            print(f"  kind: {r.kind}")
            if r.inferred_export_date:
                print(f"  inferred export folder date: {r.inferred_export_date}")
            if r.chats_backed_up is not None:
                print(f"  chats backed up: {r.chats_backed_up}")
            if r.messages_backed_up is not None:
                print(f"  messages backed up: {r.messages_backed_up}")
            print(f"  date range (UTC): {r.first_message_utc}  →  {r.last_message_utc}")
            if r.date_range_basis:
                print(f"  date range basis: {r.date_range_basis}")
            if r.first_message_source:
                print(f"  first message source: {r.first_message_source}")
            if r.last_message_source:
                print(f"  last message source: {r.last_message_source}")
            print()

        if len(reports) > 1:
            print("Nested/multiple exports detected (more than one export root under the given path).")

        print()
        print(_render_export_tree(root, reports))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
