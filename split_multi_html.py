#!/usr/bin/env python3
"""
split_multi_html.py

Convert a multi-chat Telegram HTML export into per-chat self-contained exports.
Outputs single-chat HTML exports that can be moved or indexed independently.

Usage:
  python3 split_multi_html.py /path/to/raw_multi_export [--out /path/to/output_dir] [--chat chat_001] [--dry-run]
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import sys
import urllib.parse
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

BACKMAN_EXPORT_META = ".backman_export_meta.json"

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

HTML_ATTR_URL_RE = re.compile(
    r'(?P<prefix>\b(?:src|href)\s*=\s*)(?P<q>["\'])(?P<url>[^"\']+)(?P=q)',
    re.IGNORECASE,
)
HTML_CSS_URL_RE = re.compile(
    r'url\(\s*(?P<q>["\']?)(?P<url>[^"\')]+)(?P=q)\s*\)',
    re.IGNORECASE,
)

LINK_CHAT_LOCAL_MEDIA_DIRS = {
    "photos",
    "files",
    "video_files",
    "voice_messages",
    "audio_files",
    "documents",
    "sticker_files",
    "stickers",
    "animations",
    "round_video_messages",
    "profile_pictures",
}

_SKIP_SHARED_MEDIA_TOPLEVEL = {
    "css",
    "js",
    "images",
    "profile_pictures",
    "lists",
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

def _should_skip_url(url: str) -> bool:
    u = url.strip()
    if not u:
        return True
    if u.startswith("#"):
        return True
    lu = u.lower()
    if lu.startswith(("http://", "https://", "mailto:", "javascript:", "data:", "tg:", "tel:")):
        return True
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", u):
        return True
    return False

def _resolve_local_url_to_export_file(
    url: str,
    src_html_dir: str,
    export_root: str,
) -> Optional[Tuple[str, str]]:
    if _should_skip_url(url):
        return None

    u0 = html.unescape(url).strip()
    u0 = u0.split("?", 1)[0].strip()
    if not u0:
        return None

    candidates = [u0]
    u_no_frag = u0.split("#", 1)[0].strip()
    if u_no_frag and u_no_frag != u0:
        candidates.append(u_no_frag)

    seen: Set[str] = set()
    seen_abs: Set[str] = set()

    def _consider_candidate(rel_or_abs: str) -> Optional[Tuple[str, str]]:
        abs_p = (
            os.path.abspath(os.path.normpath(rel_or_abs))
            if os.path.isabs(rel_or_abs)
            else os.path.abspath(os.path.normpath(os.path.join(src_html_dir, rel_or_abs)))
        )
        if abs_p in seen_abs:
            return None
        seen_abs.add(abs_p)
        if not _is_ancestor_dir(export_root, abs_p):
            return None
        if not os.path.isfile(abs_p):
            return None
        rel = os.path.relpath(abs_p, export_root)
        top = rel.split(os.sep, 1)[0]
        if top in _SKIP_SHARED_MEDIA_TOPLEVEL:
            return None
        return abs_p, rel

    for c in candidates:
        variants = [c]
        try:
            v1 = urllib.parse.unquote(c)
            variants.append(v1)
            v2 = urllib.parse.unquote(v1)
            variants.append(v2)
        except Exception:
            pass

        for v in variants:
            if not v or v in seen:
                continue
            seen.add(v)
            hit = _consider_candidate(v)
            if hit:
                return hit

            if "/" not in v and "\\" not in v and not v.startswith("."):
                for d in sorted(LINK_CHAT_LOCAL_MEDIA_DIRS):
                    hit = _consider_candidate(os.path.join(d, v))
                    if hit:
                        return hit

    return None

def _write_json(path: str, obj: Any) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except Exception:
        return

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

def _copy_tree(src: str, dst: str) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True, copy_function=shutil.copy2)

def _rewrite_chat_html_for_standalone(chat_root: str, chat_id: Optional[str] = None) -> None:
    for fn in os.listdir(chat_root):
        if not (fn.startswith("messages") and fn.endswith(".html")):
            continue
        p = os.path.join(chat_root, fn)
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                s = f.read()
        except Exception:
            continue

        s2 = s
        s2 = s2.replace('href="../../css/', 'href="css/')
        s2 = s2.replace('href="../css/', 'href="css/')
        s2 = s2.replace('src="../../js/', 'src="js/')
        s2 = s2.replace('src="../js/', 'src="js/')
        s2 = s2.replace('src="../../images/', 'src="images/')
        s2 = s2.replace('href="../../images/', 'href="images/')
        s2 = s2.replace('src="../../profile_pictures/', 'src="profile_pictures/')
        s2 = s2.replace('href="../../profile_pictures/', 'href="profile_pictures/')

        s2 = s2.replace('href="../../lists/chats.html"', 'href="messages.html"')
        s2 = s2.replace('href="../../lists/chats.html#allow_back"', 'href="messages.html"')
        s2 = s2.replace('href="../lists/chats.html"', 'href="messages.html"')
        s2 = s2.replace('href="../lists/chats.html#allow_back"', 'href="messages.html"')

        if chat_id:
            s2 = s2.replace(f'../../chats/{chat_id}/', '')
            s2 = s2.replace(f'../chats/{chat_id}/', '')

        if s2 != s:
            with open(p, "w", encoding="utf-8") as f:
                f.write(s2)

def _copy_and_localize_shared_media_for_chat(
    export_root: str,
    src_chat: str,
    dst_chat: str,
    chat_id: str,
) -> None:
    export_root = os.path.abspath(export_root)
    src_chat = os.path.abspath(src_chat)
    dst_chat = os.path.abspath(dst_chat)

    src_msg_files = sorted(
        os.path.join(src_chat, fn)
        for fn in os.listdir(src_chat)
        if fn.startswith("messages") and fn.endswith(".html")
    )
    if not src_msg_files:
        return

    abs_to_media_relposix: Dict[str, str] = {}

    for src_html in src_msg_files:
        src_html_dir = os.path.dirname(src_html)
        try:
            with open(src_html, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "href=" not in line and "src=" not in line and "url(" not in line:
                        continue
                    for m in HTML_ATTR_URL_RE.finditer(line):
                        url = m.group("url")
                        resolved = _resolve_local_url_to_export_file(url, src_html_dir, export_root)
                        if not resolved:
                            continue
                        abs_p, rel = resolved
                        rel_posix = rel.replace(os.sep, "/")
                        if not _is_ancestor_dir(src_chat, abs_p):
                            abs_to_media_relposix[abs_p] = rel_posix

                    for m in HTML_CSS_URL_RE.finditer(line):
                        url = m.group("url")
                        resolved = _resolve_local_url_to_export_file(url, src_html_dir, export_root)
                        if not resolved:
                            continue
                        abs_p, rel = resolved
                        rel_posix = rel.replace(os.sep, "/")
                        if not _is_ancestor_dir(src_chat, abs_p):
                            abs_to_media_relposix[abs_p] = rel_posix
        except Exception:
            continue

    def _quote_url_path(rel_posix: str) -> str:
        return urllib.parse.quote(rel_posix, safe="/")

    media_root = os.path.join(dst_chat, "media")
    for abs_p, rel_posix in abs_to_media_relposix.items():
        dp = os.path.join(media_root, rel_posix.replace("/", os.sep))
        os.makedirs(os.path.dirname(dp), exist_ok=True)
        try:
            shutil.copy2(abs_p, dp)
        except Exception:
            continue

    for src_html in src_msg_files:
        src_html_dir = os.path.dirname(src_html)
        dst_html = os.path.join(dst_chat, os.path.basename(src_html))
        if not os.path.isfile(dst_html):
            continue

        tmp = dst_html + ".tmp"

        def _rewrite_url(url: str) -> Optional[str]:
            resolved = _resolve_local_url_to_export_file(url, src_html_dir, export_root)
            if not resolved:
                return None
            abs_p, _rel = resolved
            if _is_ancestor_dir(src_chat, abs_p):
                rel_local = os.path.relpath(abs_p, src_chat).replace(os.sep, "/")
                return _quote_url_path(rel_local)

            rel_posix = abs_to_media_relposix.get(abs_p)
            if not rel_posix:
                return None
            return f"media/{_quote_url_path(rel_posix)}"

        def _sub_attr(m: re.Match) -> str:
            prefix = m.group("prefix")
            q = m.group("q")
            url = m.group("url")
            nu = _rewrite_url(url)
            if not nu:
                return m.group(0)
            return f"{prefix}{q}{nu}{q}"

        def _sub_css(m: re.Match) -> str:
            q = m.group("q")
            url = m.group("url")
            nu = _rewrite_url(url)
            if not nu:
                return m.group(0)
            return f"url({q}{nu}{q})"

        try:
            with open(dst_html, "r", encoding="utf-8", errors="ignore") as rf, open(
                tmp, "w", encoding="utf-8"
            ) as wf:
                for line in rf:
                    if "href=" not in line and "src=" not in line and "url(" not in line:
                        wf.write(line)
                        continue
                    line2 = HTML_ATTR_URL_RE.sub(_sub_attr, line)
                    line2 = HTML_CSS_URL_RE.sub(_sub_css, line2)
                    wf.write(line2)
            os.replace(tmp, dst_html)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

def split_multi_html_export_to_single_chat_exports(
    export_root: str,
    out_root: Optional[str] = None,
    only_chats: Optional[Set[str]] = None,
    dry_run: bool = False,
) -> str:
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

    shared_dirs = ["css", "js", "images", "profile_pictures"]

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

    for i, chat_id in enumerate(chat_names, start=1):
        src_chat = os.path.join(chats_dir, chat_id)
        dst_chat_parent = os.path.join(out_root, dst_name_by_chat[chat_id])

        src_msg_files = sorted(
            os.path.join(src_chat, fn)
            for fn in os.listdir(src_chat)
            if fn.startswith("messages") and fn.endswith(".html")
        )
        _, first_dt, last_dt, _, _, _ = _scan_html_message_files(src_msg_files)
        range_dir = f"{_fmt_dt_for_dir(first_dt)}__{_fmt_dt_for_dir(last_dt)}"
        dst_chat = os.path.join(dst_chat_parent, range_dir)

        os.makedirs(dst_chat, exist_ok=True)

        for name in os.listdir(src_chat):
            sp = os.path.join(src_chat, name)
            dp = os.path.join(dst_chat, name)
            if os.path.isdir(sp):
                _copy_tree(sp, dp)
            else:
                shutil.copy2(sp, dp)

        for d in shared_dirs:
            sp = os.path.join(export_root, d)
            if os.path.isdir(sp):
                _copy_tree(sp, os.path.join(dst_chat, d))

        _rewrite_chat_html_for_standalone(dst_chat, chat_id=chat_id)
        _copy_and_localize_shared_media_for_chat(export_root, src_chat, dst_chat, chat_id=chat_id)
        _write_json(
            os.path.join(dst_chat, BACKMAN_EXPORT_META),
            {
                "tool": "backman",
                "kind": "html_single_chat_export_converted",
                "converted_from": {
                    "kind": "html_multi_chat_export",
                    "export_root": export_root,
                    "chat_id": chat_id,
                },
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

        if i == 1 or i % 25 == 0 or i == len(chat_names):
            print(f"[split] {i}/{len(chat_names)}: {chat_id} -> {dst_name_by_chat[chat_id]}/{range_dir}", file=sys.stderr)

    return out_root

def main() -> int:
    ap = argparse.ArgumentParser(description="Split multi-chat HTML export into single chat folders")
    ap.add_argument("path", help="Telegram raw multi-chat export directory")
    ap.add_argument(
        "--out",
        default=None,
        help="Output directory (default: <parent>/<basename>_single_chats)",
    )
    ap.add_argument(
        "--chat",
        action="append",
        default=[],
        help="Limit splitting to specific chat folders (repeatable, e.g. --chat chat_001)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned output directory and exit",
    )
    args = ap.parse_args()

    root = os.path.abspath(args.path)
    if not os.path.exists(root):
        print(f"Path does not exist: {root}", file=sys.stderr)
        return 2

    only = set(args.chat) if args.chat else None
    try:
        out = split_multi_html_export_to_single_chat_exports(
            root,
            out_root=args.out,
            only_chats=only,
            dry_run=args.dry_run,
        )
    except Exception as e:
        print(f"Split failed: {e}", file=sys.stderr)
        return 3

    if args.dry_run:
        print(out)
        return 0
    print(f"Wrote per-chat exports to: {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
