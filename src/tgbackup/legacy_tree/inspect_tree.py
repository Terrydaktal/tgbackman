#!/usr/bin/env python3
"""
backup_inspect.py

Recursively scan a folder for Telegram backups (JSON, HTML, unofficial SQLite)
and print summary metadata, including message counts, chat summaries, date ranges,
and space footprint on disk.

Usage:
  python3 backup_inspect.py /path/to/backup_folder [--json] [--no-inspect] [--no-sizes] [--dedupe-unofficial-sqlite]
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

RESULT_NAMES = ("result.json", "results.json")
BACKMAN_EXPORT_META = ".backman_export_meta.json"

class _Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"

def _use_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return sys.stdout.isatty() and os.environ.get("TERM", "") not in ("", "dumb")
    except Exception:
        return False

def _c(s: str, code: str) -> str:
    if not _use_color():
        return s
    return f"{code}{s}{_Ansi.RESET}"

@dataclass
class ChatSummary:
    name: str
    messages_backed_up: Optional[int]
    first_message_utc: Optional[str]
    last_message_utc: Optional[str]

@dataclass
class ExportReport:
    export_root: str
    result_json: str
    fmt: str
    kind: str
    top_level_keys: List[str]
    inferred_export_date: Optional[str]
    chats_backed_up: Optional[int]
    messages_backed_up: Optional[int]
    first_message_utc: Optional[str]
    last_message_utc: Optional[str]
    first_message_source: Optional[str]
    last_message_source: Optional[str]
    date_range_basis: Optional[str]
    chat_summaries: Optional[List[ChatSummary]] = None

EXPORT_DATE_RE = re.compile(
    r"(?:DataExport|ChatExport)[_-](\d{2})[._-](\d{2})[._-](\d{4})"
)

HTML_MULTI_ROOT_NAMES = ("export_results.html",)
HTML_SINGLE_ROOT_NAME = "messages.html"

HTML_MSG_TS_RE = re.compile(
    r'<div class="pull_right date details" title="'
    r'(\d{2})\.(\d{2})\.(\d{4}) (\d{2}):(\d{2}):(\d{2})(?: UTC([+-]\d{2}):(\d{2}))?'
)
HTML_DIV_MESSAGE_MARKER = '<div class="message'
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
    s = dt_s.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def _parse_unixtime(v: Any) -> Optional[datetime]:
    try:
        val = int(v)
        return datetime.fromtimestamp(val, tz=timezone.utc)
    except Exception:
        return None

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
                    msg_count += line.count(HTML_DIV_MESSAGE_MARKER) - line.count(HTML_DAY_SEPARATOR_MARKER)

                    if expect_day_sep:
                        expect_day_sep = False
                        d = _parse_html_day_sep(line)
                        first_day_dt, last_day_dt = _upd_range(first_day_dt, last_day_dt, d)
                    if 'class="body details"' in line:
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

    if first_msg_dt or last_msg_dt:
        return msg_count, first_msg_dt, last_msg_dt, first_msg_src, last_msg_src, "message_timestamps"
    return msg_count, first_day_dt, last_day_dt, None, None, ("day_separators" if (first_day_dt or last_day_dt) else None)

def _sanitize_dirname(name: str, max_len: int = 120) -> str:
    s = name.strip()
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[\x00-\x1f\x7f]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        s = "unknown_chat"
    s = s.rstrip(" .")
    if len(s) > max_len:
        s = s[:max_len].rstrip(" .")
    return s or "unknown_chat"

def _extract_chat_name_from_messages_html(messages_html_path: str) -> Optional[str]:
    try:
        with open(messages_html_path, "r", encoding="utf-8", errors="ignore") as f:
            chunk = f.read(64 * 1024)
    except Exception:
        return None

    m = HTML_CHAT_NAME_RE.search(chunk)
    if not m:
        return None
    raw = m.group(1)
    raw = re.sub(r"<[^>]+>", "", raw)
    raw = html.unescape(raw)
    raw = raw.strip()
    return raw or None

def _fmt_dt_for_dir(d: Optional[datetime]) -> str:
    if not d:
        return "unknown"
    du = d.astimezone(timezone.utc)
    return du.strftime("%Y-%m-%dT%H-%M-%SZ")

def _maybe_mark_converted_single_html(export_root: str, kind: str) -> str:
    if kind != "html_single_chat_export":
        return kind
    if os.path.isfile(os.path.join(export_root, BACKMAN_EXPORT_META)):
        return "html_single_chat_export_converted"
    return kind

def find_result_jsons(root: str) -> List[str]:
    hits: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn in RESULT_NAMES:
                hits.append(os.path.join(dirpath, fn))
    return sorted(set(hits))

def _is_unofficial_telegram_sqlite_db(db_path: str) -> bool:
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except Exception:
        return False
    try:
        cur = con.cursor()
        cur.execute("select name from sqlite_master where type='table'")
        tables = {r[0] for r in cur.fetchall()}
        return {"messages", "chats", "users"}.issubset(tables)
    except Exception:
        return False
    finally:
        try:
            con.close()
        except Exception:
            pass

def find_unofficial_telegram_sqlite_dbs(root: str) -> List[str]:
    hits: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn == "database.sqlite":
                p = os.path.join(dirpath, fn)
                if _is_unofficial_telegram_sqlite_db(p):
                    hits.append(p)
    return sorted(set(hits))

def find_html_export_roots(root: str) -> List[str]:
    roots: Set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        fset = set(filenames)
        dset = set(dirnames)

        if any(n in fset for n in HTML_MULTI_ROOT_NAMES):
            roots.add(dirpath)
            continue

        if HTML_SINGLE_ROOT_NAME in fset and "css" in dset and "js" in dset:
            roots.add(dirpath)
            continue
    return sorted(roots)

def _iter_html_message_files(export_root: str, kind: str) -> Iterable[str]:
    if kind == "html_multi_chat_export":
        chats_dir = os.path.join(export_root, "chats")
        if not os.path.isdir(chats_dir):
            return []
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

    out2: List[str] = []
    for fn in os.listdir(export_root):
        if fn.startswith("messages") and fn.endswith(".html"):
            out2.append(os.path.join(export_root, fn))
    return sorted(out2)

def infer_export_date_from_path(p: str) -> Optional[str]:
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
        import ijson
        return ijson
    except Exception:
        return None

def sniff_top_level_keys_ijson(ijson_mod, path: str, max_events: int = 2000) -> Set[str]:
    keys: Set[str] = set()
    with open(path, "rb") as f:
        for i, (prefix, event, value) in enumerate(ijson_mod.parse(f)):
            if prefix == "" and event == "map_key":
                keys.add(str(value))
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
    if "chats" in top_keys:
        kind = "multi_chat_export"
    elif "messages" in top_keys:
        kind = "single_chat_export"

    chat_count: Optional[int] = 0 if kind == "multi_chat_export" else (1 if kind == "single_chat_export" else None)
    msg_count: int = 0
    first_dt: Optional[datetime] = None
    last_dt: Optional[datetime] = None
    chat_summaries: List[ChatSummary] = []

    cur_chat_id: Optional[str] = None
    cur_chat_name: Optional[str] = None
    cur_chat_msgs: int = 0
    cur_first: Optional[datetime] = None
    cur_last: Optional[datetime] = None

    def _flush_chat():
        nonlocal cur_chat_id, cur_chat_name, cur_chat_msgs, cur_first, cur_last
        if cur_chat_id is None:
            return
        nm = cur_chat_name or f"chat_{cur_chat_id}"
        chat_summaries.append(
            ChatSummary(
                name=nm,
                messages_backed_up=cur_chat_msgs,
                first_message_utc=cur_first.isoformat() if cur_first else None,
                last_message_utc=cur_last.isoformat() if cur_last else None,
            )
        )

    with open(path, "rb") as f:
        for prefix, event, value in ijson_mod.parse(f):
            if kind == "multi_chat_export":
                if prefix == "chats.list.item.id" and event in ("number", "string"):
                    if cur_chat_id is not None:
                        _flush_chat()
                    cur_chat_id = str(value)
                    cur_chat_name = None
                    cur_chat_msgs = 0
                    cur_first = None
                    cur_last = None
                    chat_count = (chat_count or 0) + 1

                if prefix == "chats.list.item.name" and event == "string":
                    cur_chat_name = str(value)

            if prefix.endswith(".messages.item.id") or prefix == "messages.item.id":
                if event in ("number", "string"):
                    msg_count += 1
                    if kind == "multi_chat_export" and prefix.startswith("chats.list.item.messages.item."):
                        cur_chat_msgs += 1

            if prefix.endswith(".messages.item.date_unixtime") or prefix == "messages.item.date_unixtime":
                if event in ("number", "string"):
                    d = _parse_unixtime(value)
                    if d:
                        if first_dt is None or d < first_dt:
                            first_dt = d
                        if last_dt is None or d > last_dt:
                            last_dt = d
                        if kind == "multi_chat_export" and prefix.startswith("chats.list.item.messages.item."):
                            cur_first, cur_last = _upd_range(cur_first, cur_last, d)

            if prefix.endswith(".messages.item.date") or prefix == "messages.item.date":
                if event == "string":
                    d = _parse_iso(str(value))
                    if d:
                        if first_dt is None or d < first_dt:
                            first_dt = d
                        if last_dt is None or d > last_dt:
                            last_dt = d
                        if kind == "multi_chat_export" and prefix.startswith("chats.list.item.messages.item."):
                            cur_first, cur_last = _upd_range(cur_first, cur_last, d)

    if kind == "multi_chat_export" and cur_chat_id is not None:
        _flush_chat()

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
        chat_summaries=chat_summaries if chat_summaries else None,
    )

def inspect_via_json_load(path: str) -> ExportReport:
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

    chat_summaries = []
    if kind == "multi_chat_export" and isinstance(chats, list):
        chat_count = len(chats)
        msg_count = 0
        for c in chats:
            if not isinstance(c, dict):
                continue
            cname = (
                (str(c.get("name")).strip() if c.get("name") is not None else None)
                or (str(c.get("title")).strip() if c.get("title") is not None else None)
                or (f"chat_{c.get('id')}" if c.get("id") is not None else None)
                or "unknown_chat"
            )
            c_msg = 0
            c_first: Optional[datetime] = None
            c_last: Optional[datetime] = None
            for m in c.get("messages", []) or []:
                if not isinstance(m, dict):
                    continue
                msg_count += 1
                c_msg += 1
                if "date_unixtime" in m:
                    d = _parse_unixtime(m["date_unixtime"])
                    upd(d)
                    c_first, c_last = _upd_range(c_first, c_last, d)
                if "date" in m:
                    d = _parse_iso(str(m["date"]))
                    upd(d)
                    c_first, c_last = _upd_range(c_first, c_last, d)
            chat_summaries.append(
                ChatSummary(
                    name=cname,
                    messages_backed_up=c_msg,
                    first_message_utc=c_first.isoformat() if c_first else None,
                    last_message_utc=c_last.isoformat() if c_last else None,
                )
            )
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
        chat_summaries=chat_summaries if (kind == "multi_chat_export" and chat_summaries) else None,
    )

def inspect_unofficial_telegram_sqlite(db_path: str, *, no_inspect: bool) -> ExportReport:
    export_root = os.path.abspath(os.path.dirname(db_path))
    src = os.path.abspath(db_path)

    if no_inspect:
        return ExportReport(
            export_root=export_root,
            result_json=src,
            fmt="sqlite",
            kind="sqlite_unofficial_backup",
            top_level_keys=[],
            inferred_export_date=infer_export_date_from_path(export_root),
            chats_backed_up=None,
            messages_backed_up=None,
            first_message_utc=None,
            last_message_utc=None,
            first_message_source=None,
            last_message_source=None,
            date_range_basis=None,
        )

    def _table_cols(con, table: str) -> Set[str]:
        cur2 = con.cursor()
        cur2.execute(f"pragma table_info({table})")
        return {str(r[1]) for r in cur2.fetchall()}

    def _pick_name_cols(cols: Set[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if "name" in cols:
            return "name", None, None
        first = "first_name" if "first_name" in cols else None
        last = "last_name" if "last_name" in cols else None
        if first or last:
            return None, first, last
        for c in ("title", "username", "phone", "phone_number"):
            if c in cols:
                return c, None, None
        return None, None, None

    con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    chat_summaries: List[ChatSummary] = []
    try:
        cur = con.cursor()
        cur.execute("select count(*) from messages")
        msgs = int(cur.fetchone()[0])
        cur.execute("select min(time), max(time) from messages")
        mn, mx = cur.fetchone()

        cur.execute(
            """
            select source_type, source_id, count(*) as msgs, min(time) as mn, max(time) as mx
            from messages
            where source_type is not null and source_id is not null
            group by source_type, source_id
            order by msgs desc
            """
        )
        rows = cur.fetchall()

        u_name = u_first = u_last = None
        c_name = c_first = c_last = None
        try:
            user_cols = _table_cols(con, "users")
            chat_cols = _table_cols(con, "chats")
            u_name, u_first, u_last = _pick_name_cols(user_cols)
            c_name, c_first, c_last = _pick_name_cols(chat_cols)
        except Exception:
            pass

        u_stmt = None
        if u_name:
            u_stmt = f"select {u_name} from users where id = ?"
        elif u_first or u_last:
            cols = ", ".join(c for c in (u_first, u_last) if c)
            u_stmt = f"select {cols} from users where id = ?"

        c_stmt = None
        if c_name:
            c_stmt = f"select {c_name} from chats where id = ?"
        elif c_first or c_last:
            cols = ", ".join(c for c in (c_first, c_last) if c)
            c_stmt = f"select {cols} from chats where id = ?"

        def _lookup(stmt: Optional[str], source_id: int) -> Optional[str]:
            if not stmt:
                return None
            try:
                cur3 = con.cursor()
                cur3.execute(stmt, (source_id,))
                r = cur3.fetchone()
            except Exception:
                return None
            if not r:
                return None
            if len(r) == 1:
                v = r[0]
                return (str(v).strip() if v is not None else None) or None
            parts = [str(x).strip() for x in r if x is not None and str(x).strip()]
            return (" ".join(parts).strip() or None) if parts else None

        for source_type, source_id, mcount, mn2, mx2 in rows:
            try:
                sid = int(source_id)
            except Exception:
                continue
            st = str(source_type)
            nm = None
            if st == "dialog":
                nm = _lookup(u_stmt, sid)
            elif st == "group":
                nm = _lookup(c_stmt, sid)
            name = nm or f"{st}:{sid}"
            first_dt2 = _parse_unixtime(mn2) if mn2 is not None else None
            last_dt2 = _parse_unixtime(mx2) if mx2 is not None else None
            chat_summaries.append(
                ChatSummary(
                    name=name,
                    messages_backed_up=int(mcount),
                    first_message_utc=first_dt2.isoformat() if first_dt2 else None,
                    last_message_utc=last_dt2.isoformat() if last_dt2 else None,
                )
            )
    finally:
        con.close()

    first_dt = _parse_unixtime(mn) if mn is not None else None
    last_dt = _parse_unixtime(mx) if mx is not None else None

    return ExportReport(
        export_root=export_root,
        result_json=src,
        fmt="sqlite",
        kind="sqlite_unofficial_backup",
        top_level_keys=[],
        inferred_export_date=infer_export_date_from_path(export_root),
        chats_backed_up=len(chat_summaries) if chat_summaries else None,
        messages_backed_up=msgs,
        first_message_utc=first_dt.isoformat() if first_dt else None,
        last_message_utc=last_dt.isoformat() if last_dt else None,
        first_message_source=src,
        last_message_source=src,
        date_range_basis="message_timestamps",
        chat_summaries=chat_summaries if chat_summaries else None,
    )

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
    kind = _maybe_mark_converted_single_html(export_root, kind)

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

    chat_summaries: List[ChatSummary] = []
    if kind == "html_multi_chat_export":
        msg_count = 0
        first_dt: Optional[datetime] = None
        last_dt: Optional[datetime] = None
        first_src: Optional[str] = None
        last_src: Optional[str] = None
        basis: Optional[str] = None

        chats_dir = os.path.join(export_root, "chats")
        if os.path.isdir(chats_dir):
            for name in sorted(os.listdir(chats_dir)):
                if not name.startswith("chat_"):
                    continue
                chat_dir = os.path.join(chats_dir, name)
                if not os.path.isdir(chat_dir):
                    continue
                try:
                    msg_files = [
                        os.path.join(chat_dir, fn)
                        for fn in sorted(os.listdir(chat_dir))
                        if fn.startswith("messages") and fn.endswith(".html")
                    ]
                except Exception:
                    continue
                c_count, c_first, c_last, c_first_src, c_last_src, c_basis = _scan_html_message_files(msg_files)
                title = _extract_chat_name_from_messages_html(os.path.join(chat_dir, "messages.html")) or name
                chat_summaries.append(
                    ChatSummary(
                        name=title,
                        messages_backed_up=c_count,
                        first_message_utc=c_first.isoformat() if c_first else None,
                        last_message_utc=c_last.isoformat() if c_last else None,
                    )
                )
                msg_count += c_count
                if c_first and (first_dt is None or c_first < first_dt):
                    first_dt = c_first
                    first_src = c_first_src
                if c_last and (last_dt is None or c_last > last_dt):
                    last_dt = c_last
                    last_src = c_last_src
                if c_basis == "message_timestamps":
                    basis = "message_timestamps"
                elif basis is None:
                    basis = c_basis
        else:
            first_dt = last_dt = None
            first_src = last_src = None
            basis = None
    else:
        msg_files2 = list(_iter_html_message_files(export_root, kind))
        msg_count, first_dt, last_dt, first_src, last_src, basis = _scan_html_message_files(msg_files2)

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
        chat_summaries=chat_summaries if (kind == "html_multi_chat_export" and chat_summaries) else None,
    )

def summarize_html_export(export_root: str) -> ExportReport:
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
    kind = _maybe_mark_converted_single_html(export_root, kind)

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

def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        n /= 1024.0
        if n < 1024.0:
            return f"{n:.2f} {unit}"
    return f"{n:.2f} PiB"

def _parse_human_bytes(tok: str) -> Optional[int]:
    m = re.match(r"^\s*([\d.]+)\s*([a-zA-Z]*)\s*$", tok)
    if not m:
        return None
    try:
        num = float(m.group(1))
    except Exception:
        return None
    unit = m.group(2).strip()
    if not unit:
        try:
            return int(num)
        except Exception:
            return None

    unit_map_1024 = {
        "B": 1,
        "K": 1024, "Kb": 1024, "Kib": 1024, "KiB": 1024,
        "M": 1024**2, "Mb": 1024**2, "Mib": 1024**2, "MiB": 1024**2,
        "G": 1024**3, "Gb": 1024**3, "Gib": 1024**3, "GiB": 1024**3,
        "T": 1024**4, "Tb": 1024**4, "Tib": 1024**4, "TiB": 1024**4,
        "P": 1024**5, "Pb": 1024**5, "Pib": 1024**5, "PiB": 1024**5,
    }
    mult = unit_map_1024.get(unit)
    if mult is None:
        mult = unit_map_1024.get(unit.capitalize())
    if mult is None:
        return None
    try:
        return int(num * mult)
    except Exception:
        return None

def _parse_dust_total(stdout: str, path: str) -> Optional[int]:
    p = os.path.abspath(path)
    lines = [ANSI_ESCAPE_RE.sub("", ln.strip("\n")) for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        return None

    candidates = [ln for ln in lines if p in ln]
    if not candidates:
        candidates = [lines[-1]]

    for ln in candidates:
        cleaned = ln.replace("│", " ").replace("█", " ").replace("▉", " ").replace("▊", " ").replace("▌", " ")
        parts = [x for x in cleaned.split() if x]
        for tok in parts:
            b = _parse_human_bytes(tok)
            if b is not None:
                return b
    return None

def _dir_size_bytes(path: str) -> Tuple[Optional[int], str]:
    p = os.path.abspath(path)
    dust = shutil.which("dust")
    if dust:
        try:
            for args in (
                [dust, "-b", "--depth", "0", "--no-colors", p],
                [dust, "-b", "-d", "0", "--no-colors", p],
                [dust, "-b", "--depth", "0", "--no-color", p],
                [dust, "-b", "-d", "0", "--no-color", p],
                [dust, "-b", "--depth", "0", p],
                [dust, "-b", "-d", "0", p],
                [dust, "--depth", "0", "--no-colors", p],
                [dust, "-d", "0", "--no-colors", p],
                [dust, "--depth", "0", p],
                [dust, "-d", "0", p],
            ):
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
                    try:
                        has_entries = any(True for _ in os.scandir(p))
                    except Exception:
                        has_entries = False
                    if has_entries and b <= 16 * 1024:
                        break
                    return b, "dust"
        except Exception:
            pass

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

def _bulk_dir_sizes_via_du(root: str) -> Tuple[Dict[str, int], str]:
    root = os.path.abspath(root)
    try:
        proc = subprocess.run(
            ["du", "-b", root],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return {}, "unknown"
        out: Dict[str, int] = {}
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            b_s, p = parts
            try:
                b = int(b_s)
            except Exception:
                continue
            out[os.path.abspath(p)] = b
        return out, "du"
    except Exception:
        return {}, "unknown"

def _chunks(items: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]

def _dust_call(args: List[str]) -> Optional[str]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return None
        return proc.stdout
    except Exception:
        return None

def _strip_dust_columns(s: str) -> str:
    if "│" in s:
        s = s.split("│", 1)[0]
    s = re.sub(r"\s+\d+%\s*$", "", s)
    return s.strip().rstrip("/")

def _parse_dust_tree(stdout: str, root: str) -> Dict[str, int]:
    root = os.path.abspath(root)
    out: Dict[str, int] = {}
    stack: List[str] = []
    for raw in stdout.splitlines():
        line = ANSI_ESCAPE_RE.sub("", raw.rstrip("\n"))
        if not line.strip():
            continue

        cleaned = (
            line.replace("│", " ")
            .replace("█", " ")
            .replace("▉", " ")
            .replace("▊", " ")
            .replace("▌", " ")
            .replace("▓", " ")
            .replace("▒", " ")
            .replace("░", " ")
        )
        b: Optional[int] = None
        toks = [x for x in cleaned.split() if x]
        size_i = -1
        for i, tok in enumerate(toks):
            b_try = _parse_human_bytes(tok)
            if b_try is not None:
                b = b_try
                size_i = i
                break
        if b is None:
            continue

        if root in line:
            idx = line.find(root)
            path_label = line[idx:].strip().rstrip("/")
            p = os.path.abspath(path_label)
            if _is_ancestor_dir(root, p):
                out[p] = b
            continue

        marker_match = None
        for pat in (
            r"├──\s+",
            r"└──\s+",
            r"├─┬\s+",
            r"└─┬\s+",
            r"\|--\s+",
            r"\+--\s+",
            r"`--\s+",
        ):
            m = re.search(pat, line)
            if m:
                marker_match = m
                break

        if not marker_match:
            rel_tok: Optional[str] = None
            if size_i >= 0:
                for tok in toks[size_i + 1 :]:
                    if tok in (".", "./."):
                        rel_tok = "."
                        break
                    if tok.startswith("./") or tok.startswith("../") or ("/" in tok) or (os.sep in tok):
                        rel_tok = tok
                        break
            if rel_tok is not None:
                idx = line.find(rel_tok)
                if idx != -1:
                    path_label = _strip_dust_columns(line[idx:])
                    if path_label in (".", "./."):
                        p = root
                    else:
                        if path_label.startswith("./"):
                            path_label = path_label[2:]
                        path_label = path_label.lstrip(os.sep)
                        p = os.path.abspath(os.path.join(root, path_label))
                    if _is_ancestor_dir(root, p):
                        out[p] = b
                        rel2 = os.path.relpath(p, root)
                        stack = [] if rel2 == "." else rel2.split(os.sep)
                        continue

            if size_i >= 0 and size_i < len(toks):
                size_tok = toks[size_i]
                j = line.find(size_tok)
                if j != -1:
                    cand = _strip_dust_columns(line[j + len(size_tok) :])
                    cand = cand.lstrip(" -\t")
                    if cand:
                        if cand == os.path.basename(root) and not out:
                            out[root] = b
                            stack = []
                            continue
                        p = os.path.abspath(os.path.join(root, cand))
                        if _is_ancestor_dir(root, p):
                            out[p] = b
                            rel2 = os.path.relpath(p, root)
                            stack = [] if rel2 == "." else rel2.split(os.sep)
                            continue

            if not out:
                out[root] = b
            else:
                if root in line:
                    out[root] = b
            continue

        idx = marker_match.start()
        name = _strip_dust_columns(line[marker_match.end() :])
        if not name:
            continue

        depth_prefix = 0
        j = idx
        while j > 0:
            if j >= 4 and line[j - 4 : j] in ("│   ", "    "):
                depth_prefix += 1
                j -= 4
                continue
            if j >= 3 and line[j - 3 : j] in ("|  ", "│  "):
                depth_prefix += 1
                j -= 3
                continue
            break
        depth = depth_prefix + 1

        p: Optional[str] = None
        if not out:
            if name == os.path.basename(root):
                out[root] = b
                stack = []
                continue
        if name == ".":
            p = root
            stack = []
        elif os.path.isabs(name):
            p = os.path.abspath(name)
            if _is_ancestor_dir(root, p):
                rel = os.path.relpath(p, root)
                stack = [] if rel == "." else rel.split(os.sep)
        elif (os.sep in name) or ("/" in name):
            rel = name
            if rel.startswith("./"):
                rel = rel[2:]
            rel = rel.lstrip(os.sep)
            p = os.path.abspath(os.path.join(root, rel))
            if _is_ancestor_dir(root, p):
                rel2 = os.path.relpath(p, root)
                stack = [] if rel2 == "." else rel2.split(os.sep)
        else:
            if depth <= 0:
                continue
            if len(stack) >= depth:
                stack = stack[: depth - 1]
            while len(stack) < depth - 1:
                stack.append("_")
            stack.append(name)
            rel = os.path.join(*stack) if stack else "."
            p = os.path.abspath(os.path.join(root, rel))

        if p and _is_ancestor_dir(root, p):
            out[p] = b

    return out

def _bulk_dir_sizes_via_dust_tree(root: str, max_depth: int, max_lines: Optional[int] = None) -> Tuple[Dict[str, int], str]:
    dust = shutil.which("dust")
    if not dust:
        return {}, "unknown"

    root = os.path.abspath(root)
    max_depth = max(0, int(max_depth))
    n_lines = None
    if max_lines is not None:
        try:
            n_lines = max(1, int(max_lines))
        except Exception:
            n_lines = None

    arg_sets: List[List[str]] = []
    base0 = [dust, "--depth", str(max_depth), "--no-colors", root]
    base1 = [dust, "--depth", str(max_depth), root]
    base0s = [dust, "-d", str(max_depth), "--no-colors", root]
    base1s = [dust, "-d", str(max_depth), root]
    baseb0 = [dust, "-b", "--depth", str(max_depth), "--no-colors", root]
    baseb1 = [dust, "-b", "--depth", str(max_depth), root]
    baseb0s = [dust, "-b", "-d", str(max_depth), "--no-colors", root]
    baseb1s = [dust, "-b", "-d", str(max_depth), root]

    if n_lines is not None:
        arg_sets.append([dust, "--depth", str(max_depth), "--number-of-lines", str(n_lines), "--no-colors", root])
        arg_sets.append([dust, "--depth", str(max_depth), "--number-of-lines", str(n_lines), root])
        arg_sets.append([dust, "-b", "--depth", str(max_depth), "--number-of-lines", str(n_lines), "--no-colors", root])
        arg_sets.append([dust, "-b", "--depth", str(max_depth), "--number-of-lines", str(n_lines), root])
        arg_sets.append([dust, "-d", str(max_depth), "-n", str(n_lines), "--no-colors", root])
        arg_sets.append([dust, "-d", str(max_depth), "-n", str(n_lines), root])
        arg_sets.append([dust, "-b", "-d", str(max_depth), "-n", str(n_lines), "--no-colors", root])
        arg_sets.append([dust, "-b", "-d", str(max_depth), "-n", str(n_lines), root])

    arg_sets.append(base0)
    arg_sets.append(base1)
    arg_sets.append(baseb0)
    arg_sets.append([dust, "--depth", str(max_depth), "--no-color", root])
    arg_sets.append([dust, "-d", str(max_depth), "--no-color", root])
    arg_sets.append([dust, "-b", "--depth", str(max_depth), "--no-color", root])
    arg_sets.append([dust, "-b", "-d", str(max_depth), "--no-color", root])
    arg_sets.append(base0s)
    arg_sets.append(base1s)
    arg_sets.append(baseb0s)
    arg_sets.append(baseb1)
    arg_sets.append(baseb1s)

    for args in arg_sets:
        stdout = _dust_call(args)
        if stdout is None:
            continue
        m = _parse_dust_tree(stdout, root)
        if m:
            return m, "dust"
    return {}, "unknown"

def _bulk_export_sizes_once(scan_root: str, export_roots: List[str]) -> Tuple[Dict[str, Optional[int]], str]:
    scan_root = os.path.abspath(scan_root)
    abs_roots = [os.path.abspath(p) for p in export_roots]

    out: Dict[str, Optional[int]] = {p: None for p in abs_roots}
    dust = shutil.which("dust")
    if not dust:
        du_map, _tool = _bulk_dir_sizes_via_du(scan_root)
        if du_map:
            for p in abs_roots:
                out[p] = du_map.get(p)
            return out, "du"
        return out, "unknown"

    max_depth = 0
    for p in abs_roots:
        if not _is_ancestor_dir(scan_root, p):
            continue
        rel = os.path.relpath(p, scan_root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        max_depth = max(max_depth, depth)

    try:
        top_dirs = sum(1 for e in os.scandir(scan_root) if e.is_dir(follow_symlinks=False))
    except Exception:
        top_dirs = max(200, len(abs_roots))
    max_lines = max(200, 1 + top_dirs + len(abs_roots) + 100)

    dust_map, tool1 = _bulk_dir_sizes_via_dust_tree(scan_root, max_depth=max_depth, max_lines=max_lines)
    if dust_map:
        for p in abs_roots:
            out[p] = dust_map.get(p)

    def _chain_is_only_child(parent: str, child: str) -> bool:
        try:
            child_base = os.path.basename(child)
            if os.path.abspath(os.path.dirname(child)) != os.path.abspath(parent):
                return False
            subdirs: List[str] = []
            for ent in os.scandir(parent):
                name = ent.name
                if name.startswith("."):
                    continue
                if ent.is_dir(follow_symlinks=False):
                    subdirs.append(name)
                    if len(subdirs) > 1:
                        return False
                else:
                    return False
            return len(subdirs) == 1 and subdirs[0] == child_base
        except Exception:
            return False

    def _infer_from_ancestor(p: str) -> Optional[int]:
        cur = os.path.abspath(p)
        while True:
            if cur in dust_map:
                break
            if cur == scan_root:
                return None
            nxt = os.path.abspath(os.path.dirname(cur))
            if nxt == cur:
                return None
            cur = nxt

        ancestor = cur
        size = dust_map.get(ancestor)
        if size is None:
            return None

        if ancestor == p:
            return size

        rel = os.path.relpath(p, ancestor)
        parts = [] if rel == "." else rel.split(os.sep)
        cur_parent = ancestor
        cur_path = ancestor
        for part in parts:
            cur_path = os.path.join(cur_parent, part)
            if not _chain_is_only_child(cur_parent, cur_path):
                return None
            cur_parent = cur_path
        return size

    if dust_map:
        for p in abs_roots:
            if out[p] is None:
                out[p] = _infer_from_ancestor(p)

    missing = [p for p in abs_roots if out[p] is None]
    if missing:
        du_map, tool2 = _bulk_dir_sizes_via_du(scan_root)
        if du_map:
            for p in missing:
                out[p] = du_map.get(p)
            return out, ("dust+du" if dust_map else "du")
        return out, (tool1 if dust_map else tool2)

    return out, ("dust" if dust_map else tool1)

def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect and summarize Telegram backups recursively")
    ap.add_argument("path", help="Telegram export folder (will be scanned recursively)")
    ap.add_argument("--json", action="store_true", help="Output JSON format (machine-readable)")
    ap.add_argument(
        "--no-inspect",
        action="store_true",
        help="Skip scanning message bodies for counts/date ranges (much faster)",
    )
    ap.add_argument(
        "--no-sizes",
        action="store_true",
        help="Do not compute on-disk sizes for discovered exports",
    )
    ap.add_argument(
        "--dedupe-unofficial-sqlite",
        action="store_true",
        help=(
            "For unofficial SQLite backups: suppress .sqlite files outside .telegram_backup/ "
            "when a canonical DB exists under the same folder"
        ),
    )
    args = ap.parse_args()

    root = os.path.abspath(args.path)
    if not os.path.exists(root):
        print(f"Path does not exist: {root}", file=sys.stderr)
        return 2

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

    sqlite_hits = find_unofficial_telegram_sqlite_dbs(root)
    if sqlite_hits:
        if args.dedupe_unofficial_sqlite:
            canon = [
                p
                for p in sqlite_hits
                if f"{os.sep}.telegram_backup{os.sep}" in os.path.abspath(p)
            ]
            if canon:
                filtered: List[str] = []
                for p in sqlite_hits:
                    ap_p = os.path.abspath(p)
                    if ap_p in canon:
                        filtered.append(p)
                        continue
                    d = os.path.dirname(ap_p)
                    if any(os.path.commonpath([d, os.path.abspath(q)]) == d for q in canon):
                        continue
                    filtered.append(p)
                sqlite_hits = filtered
        reports.extend(inspect_unofficial_telegram_sqlite(p, no_inspect=args.no_inspect) for p in sqlite_hits)

    if not reports:
        print("No Telegram exports found under the given path.", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([asdict(r) for r in reports], indent=2, ensure_ascii=False))
        return 0

    size_tool_used: Set[str] = set()
    size_by_export_root: Dict[str, Optional[int]] = {}
    if not args.no_sizes:
        export_roots = [r.export_root for r in reports]
        size_by_export_root, tool = _bulk_export_sizes_once(root, export_roots)
        if tool:
            size_tool_used.add(tool)

    def _is_collection_mode() -> bool:
        if len(reports) < 5:
            return False
        if not all(r.kind in ("html_single_chat_export", "html_single_chat_export_converted", "single_chat_export") for r in reports):
            return False
        try:
            for r in reports:
                rel = os.path.relpath(r.export_root, root)
                if rel == ".":
                    return False
                if rel.count(os.sep) < 1:
                    return False
        except Exception:
            return False
        return True

    collection_mode = _is_collection_mode()

    def _fmt_int(v: Optional[int]) -> str:
        return "?" if v is None else str(v)

    def _fmt_size(p: str) -> str:
        if args.no_sizes:
            return "?"
        b = size_by_export_root.get(os.path.abspath(p))
        return "?" if b is None else _format_bytes(b)

    def _fmt_iso_utc(s: Optional[str]) -> str:
        if not s:
            return "?"
        d = _parse_iso(s)
        if not d:
            return "?"
        du = d.astimezone(timezone.utc)
        return du.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _line_for_report(r: ExportReport) -> str:
        return _line_for_report_base(r, root=root, base=root, indent="")

    def _line_for_report_base(r: ExportReport, *, root: str, base: str, indent: str) -> str:
        rel = os.path.relpath(r.export_root, base)
        parts = [] if rel == "." else rel.split(os.sep)
        if collection_mode and len(parts) >= 2:
            chat = parts[0]
            sub = " ".join(parts[1:])
        else:
            chat = parts[0] if parts else os.path.basename(r.export_root)
            sub = " ".join(parts[1:]) if len(parts) > 1 else ""

        left = indent + _c(chat, _Ansi.CYAN)
        if sub:
            left += " " + _c(sub, _Ansi.DIM)

        fmt = _c(r.fmt, _Ansi.MAGENTA)
        chats_s = _c(_fmt_int(r.chats_backed_up), _Ansi.YELLOW)
        msgs_s = _c(_fmt_int(r.messages_backed_up), _Ansi.YELLOW)
        size_s = _c(_fmt_size(r.export_root), _Ansi.GREEN)
        kind = _c(r.kind, _Ansi.BLUE)

        range_s = ""
        if r.kind in ("html_single_chat_export", "html_single_chat_export_converted", "single_chat_export"):
            range_s = (
                " "
                + _c("range:", _Ansi.DIM)
                + _c(_fmt_iso_utc(r.first_message_utc), _Ansi.DIM)
                + _c(" → ", _Ansi.DIM)
                + _c(_fmt_iso_utc(r.last_message_utc), _Ansi.DIM)
            )

        return (
            f"{left} {fmt} chats:{chats_s} messages:{msgs_s} "
            f"size on disk: {size_s} {kind}{range_s}"
        )

    def _print_chat_summaries(r: ExportReport, *, indent: str = "") -> None:
        if not r.chat_summaries:
            return
        def _k(cs: ChatSummary) -> Tuple[int, str]:
            m = cs.messages_backed_up
            return (-(m if m is not None else -1), cs.name)

        for cs in sorted(r.chat_summaries, key=_k):
            fd = _parse_iso(cs.first_message_utc) if cs.first_message_utc else None
            ld = _parse_iso(cs.last_message_utc) if cs.last_message_utc else None
            rng = f"{_fmt_dt_for_dir(fd)}__{_fmt_dt_for_dir(ld)}"
            msgs = "?" if cs.messages_backed_up is None else str(cs.messages_backed_up)
            line = (
                indent
                + "  "
                + _c(cs.name, _Ansi.CYAN)
                + " "
                + _c(rng, _Ansi.DIM)
                + " messages:"
                + _c(msgs, _Ansi.YELLOW)
            )
            print(line)

    if collection_mode:
        print(_c(f"Export root: {root}", _Ansi.BOLD))
        print()
        groups: Dict[str, List[ExportReport]] = {}
        for r in reports:
            rel = os.path.relpath(r.export_root, root)
            parts = [] if rel == "." else rel.split(os.sep)
            key = parts[0] if parts else os.path.basename(r.export_root)
            groups.setdefault(key, []).append(r)

        def _msg_sort_val(v: Optional[int]) -> int:
            return v if v is not None else -1

        def _group_sort_key(item: Tuple[str, List[ExportReport]]) -> tuple:
            name, rs = item
            m = max((_msg_sort_val(r.messages_backed_up) for r in rs), default=-1)
            return (-m, name)

        def _sub_label(r: ExportReport) -> str:
            rel = os.path.relpath(r.export_root, root)
            parts = [] if rel == "." else rel.split(os.sep)
            if len(parts) <= 1:
                return os.path.basename(r.export_root)
            return " ".join(parts[1:])

        for _chat_name, rs in sorted(groups.items(), key=_group_sort_key):
            def _within_key(r: ExportReport) -> tuple:
                m = _msg_sort_val(r.messages_backed_up)
                return (-m, _sub_label(r))

            rs_sorted = sorted(rs, key=_within_key)
            if not rs_sorted:
                continue

            print(_line_for_report(rs_sorted[0]))
            _print_chat_summaries(rs_sorted[0])

            for r2 in rs_sorted[1:]:
                fmt = _c(r2.fmt, _Ansi.MAGENTA)
                chats_s = _c(_fmt_int(r2.chats_backed_up), _Ansi.YELLOW)
                msgs_s = _c(_fmt_int(r2.messages_backed_up), _Ansi.YELLOW)
                size_s = _c(_fmt_size(r2.export_root), _Ansi.GREEN)
                kind = _c(r2.kind, _Ansi.BLUE)

                range_s = ""
                if r2.kind in ("html_single_chat_export", "html_single_chat_export_converted", "single_chat_export"):
                    range_s = (
                        " "
                        + _c("range:", _Ansi.DIM)
                        + _c(_fmt_iso_utc(r2.first_message_utc), _Ansi.DIM)
                        + _c(" → ", _Ansi.DIM)
                        + _c(_fmt_iso_utc(r2.last_message_utc), _Ansi.DIM)
                    )

                left = "      " + _c(_sub_label(r2), _Ansi.DIM)
                print(
                    f"{left} {fmt} chats:{chats_s} messages:{msgs_s} "
                    f"size on disk: {size_s} {kind}{range_s}"
                )
                _print_chat_summaries(r2)
    else:
        def _is_ancestor(a: str, b: str) -> bool:
            try:
                ap = os.path.abspath(a)
                bp = os.path.abspath(b)
                if ap == bp:
                    return False
                return os.path.commonpath([ap, bp]) == ap
            except Exception:
                return False

        report_by_root: Dict[str, ExportReport] = {os.path.abspath(r.export_root): r for r in reports}
        roots = sorted(report_by_root.keys(), key=lambda p: (p.count(os.sep), p))

        parent_of: Dict[str, Optional[str]] = {p: None for p in roots}
        for p in roots:
            best: Optional[str] = None
            for cand in roots:
                if _is_ancestor(cand, p):
                    if best is None or len(cand) > len(best):
                        best = cand
            parent_of[p] = best

        children: Dict[str, List[str]] = {p: [] for p in roots}
        for p, par in parent_of.items():
            if par:
                children[par].append(p)

        def _msg_sort_val(v: Optional[int]) -> int:
            return v if v is not None else -1

        def _node_sort_key(p: str, *, base: str) -> tuple:
            r = report_by_root[p]
            rel = os.path.relpath(p, base)
            return (-_msg_sort_val(r.messages_backed_up), rel)

        def _print_node(p: str, *, depth: int, base: str) -> None:
            r = report_by_root[p]
            indent = "  " * depth
            print(_line_for_report_base(r, root=root, base=base, indent=indent))
            _print_chat_summaries(r, indent=indent)

            kids = children.get(p) or []
            for ch in sorted(kids, key=lambda x: _node_sort_key(x, base=p)):
                _print_node(ch, depth=depth + 1, base=p)

        print(_c(f"Export root: {root}", _Ansi.BOLD))
        print()
        top = [p for p in roots if parent_of[p] is None]
        for p in sorted(top, key=lambda x: _node_sort_key(x, base=root)):
            _print_node(p, depth=0, base=root)

    if not args.no_sizes and size_tool_used:
        tool_note = ", ".join(sorted(t for t in size_tool_used if t != "unknown"))
        if tool_note:
            print()
            print(_c(f"(size tool: {tool_note})", _Ansi.DIM))

    return 0

if __name__ == "__main__":
    sys.exit(main())
