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
import sqlite3
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

RESULT_NAMES = ("result.json", "results.json")
UNOFFICIAL_SQLITE_NAMES = ("database.sqlite",)

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

def _write_json(path: str, obj: Any) -> None:
    # Best-effort: metadata should never break the split.
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except Exception:
        return

def _maybe_mark_converted_single_html(export_root: str, kind: str) -> str:
    """
    If this looks like a per-chat export produced by backman split, label it as converted.
    """
    if kind != "html_single_chat_export":
        return kind
    if os.path.isfile(os.path.join(export_root, BACKMAN_EXPORT_META)):
        return "html_single_chat_export_converted"
    return kind

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

# Local URL extraction for HTML (messages.html).
HTML_ATTR_URL_RE = re.compile(
    r'(?P<prefix>\b(?:src|href)\s*=\s*)(?P<q>["\'])(?P<url>[^"\']+)(?P=q)',
    re.IGNORECASE,
)
HTML_CSS_URL_RE = re.compile(
    r'url\(\s*(?P<q>["\']?)(?P<url>[^"\')]+)(?P=q)\s*\)',
    re.IGNORECASE,
)

# For link-checking (broader than split rewrite): includes poster= and srcset=
LINK_ATTR_URL_RE = re.compile(
    r'(?P<prefix>\b(?:src|href|poster)\s*=\s*)(?P<q>["\'])(?P<url>[^"\']+)(?P=q)',
    re.IGNORECASE,
)
LINK_ATTR_SRCSET_RE = re.compile(
    r'(?P<prefix>\bsrcset\s*=\s*)(?P<q>["\'])(?P<val>[^"\']+)(?P=q)',
    re.IGNORECASE,
)
LINK_MULTI_EXPORT_REF_RE = re.compile(
    r'\b(?:href|src)\s*=\s*(?P<q>["\'])(?P<url>(?:\.\./)+chats/chat_\d+/[^"\']+)(?P=q)',
    re.IGNORECASE,
)

# Used by link-checking to identify media-ish paths (keeps signal high when scanning huge exports)
LINK_MEDIA_DIRS = {
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
    "images",
    # Used by backman split to localize shared media
    "media",
}
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

# File-like suffixes that should not be treated as web host TLDs for bare refs
# such as "src.zip" or "document.pdf".
_LIKELY_FILE_EXTS = {
    "7z", "aac", "ai", "apk", "avi", "bat", "bin", "bmp", "csv", "db", "doc",
    "docx", "dmg", "epub", "exe", "flac", "gif", "gz", "heic", "html", "htm",
    "iso", "jpeg", "jpg", "js", "json", "m4a", "mkv", "mov", "mp3", "mp4",
    "ogg", "pdf", "png", "ppt", "pptx", "rar", "rtf", "sqlite", "svg", "tar",
    "tgz", "tiff", "txt", "wav", "webm", "webp", "xls", "xlsx", "xml", "yaml",
    "yml", "zip",
}

_SKIP_SHARED_MEDIA_TOPLEVEL = {
    # These are copied separately as shared assets for standalone rendering.
    "css",
    "js",
    "images",
    "profile_pictures",
    # Not media; avoid copying navigation pages.
    "lists",
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

def _iter_all_html_files(root: str) -> Iterable[str]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".html"):
                yield os.path.join(dirpath, fn)

def _iter_chat_export_roots(root: str) -> Iterable[str]:
    """
    Yield chat export roots: directories that contain at least one messages*.html.
    (Works for both a single chat export root and a collection of per-chat exports.)
    """
    for dirpath, _dirnames, filenames in os.walk(root):
        if any(fn.startswith("messages") and fn.endswith(".html") for fn in filenames):
            yield dirpath

def _count_files_recursive(p: str) -> int:
    n = 0
    stack = [p]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif e.is_file(follow_symlinks=False):
                            n += 1
                    except Exception:
                        continue
        except Exception:
            continue
    return n

def _normalize_url_to_fs_path(url: str) -> str:
    u = html.unescape(url).strip()
    u = u.split("#", 1)[0].split("?", 1)[0].strip()
    return u

def _strip_leading_dot_segments(p: str) -> str:
    # Collapse leading "./" and "../" for classification; does not resolve on disk.
    s = p.replace("\\", "/")
    out: List[str] = []
    for seg in s.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            continue
        out.append(seg)
    return "/".join(out)

def _resolve_local_ref_for_check(
    url: str,
    src_html_path: str,
    root: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (resolved_abs_path, note).
    note is non-None for special cases (outside root, absolute, etc.).
    """
    if _should_skip_url(url):
        return None, None

    u0 = _normalize_url_to_fs_path(url)
    if not u0:
        return None, None

    # Telegram exports sometimes keep percent-encoding in on-disk filenames; try both.
    u_candidates = [u0]
    try:
        uq = urllib.parse.unquote(u0)
        if uq and uq != u0:
            u_candidates.append(uq)
    except Exception:
        pass

    src_dir = os.path.dirname(src_html_path)
    root_abs = os.path.abspath(root)
    chosen_abs: Optional[str] = None
    chosen_note: Optional[str] = None

    for u in u_candidates:
        if not u:
            continue
        if os.path.isabs(u):
            abs_p = os.path.abspath(os.path.normpath(u))
            note = "absolute"
        else:
            abs_p = os.path.abspath(os.path.normpath(os.path.join(src_dir, u)))
            note = None

        try:
            within = os.path.commonpath([root_abs, abs_p]) == root_abs
        except Exception:
            within = False
        if not within:
            note = (note + ",outside-root") if note else "outside-root"

        if os.path.exists(abs_p):
            return abs_p, note

        if chosen_abs is None:
            chosen_abs = abs_p
            chosen_note = note

    # Bare host-like refs (e.g. weather.com, abcmovies.ru/path) are external links
    # in Telegram HTML and should not be reported as missing local files. But keep
    # true local files if they actually exist (handled by the exists-return above).
    if _looks_like_external_host_ref(u0):
        return None, None
    # Telegram also linkifies plain text tokens like "src.zip" or "Language.MOV".
    # If such a bare filename token does not resolve to a real file locally, treat
    # it as non-local text and skip it to avoid noisy false positives.
    if _looks_like_bare_file_token_ref(u0):
        return None, None

    return chosen_abs, chosen_note


def _looks_like_external_host_ref(url: str) -> bool:
    u = _normalize_url_to_fs_path(url)
    if not u:
        return False
    u = u.strip()
    if not u or u.startswith((".", "/", "#")):
        return False
    if "\\" in u or " " in u:
        return False
    host = u.split("/", 1)[0].strip()
    if not host or host.count(".") < 1:
        return False
    # IPv4 literals (with optional CIDR suffix in original url) are external refs.
    if _looks_like_ipv4_literal(host):
        return True
    # Be permissive for host-like tokens seen in message text (including underscores).
    if not re.fullmatch(r"[A-Za-z0-9._-]+", host):
        return False
    tld = host.rsplit(".", 1)[-1].lower()
    if tld in _LIKELY_FILE_EXTS:
        return False
    if not re.fullmatch(r"[a-z]{2,24}", tld):
        return False
    return True


def _looks_like_bare_file_token_ref(url: str) -> bool:
    u = _normalize_url_to_fs_path(url)
    if not u:
        return False
    u = u.strip()
    if not u or u.startswith((".", "/", "#")):
        return False
    # Only a single path segment, e.g. "src.zip" / "Language.MOV".
    if "/" in u or "\\" in u:
        return False
    # Typical "filename.ext" pattern; extension in known file-like set.
    if "." not in u:
        return False
    ext = u.rsplit(".", 1)[-1].lower()
    if ext not in _LIKELY_FILE_EXTS:
        return False
    return True


def _looks_like_ipv4_literal(host: str) -> bool:
    parts = host.split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not re.fullmatch(r"\d{1,3}", p):
            return False
        try:
            v = int(p)
        except Exception:
            return False
        if v < 0 or v > 255:
            return False
    return True

def _is_media_url_for_check(url: str) -> bool:
    """
    Media in Telegram exports is almost always stored under specific directories. For a
    "media-only" scan, we restrict to those directories to avoid false positives from
    pasted code snippets and linkified domains.
    """
    if _should_skip_url(url):
        return False
    u = _normalize_url_to_fs_path(url)
    if not u:
        return False
    u2 = _strip_leading_dot_segments(u)
    parts = [p for p in u2.replace("\\", "/").lstrip("/").split("/") if p]
    return any(p in LINK_MEDIA_DIRS for p in parts)

def _classify_media_ref_for_check(url: str) -> Optional[str]:
    """
    For media-like refs, classify whether it points at:
    - shared media localized under media/ (split output)
    - chat-local media dirs like files/, photos/, etc.
    Returns "shared", "chat_local", or None.
    """
    if _should_skip_url(url):
        return None
    u = _normalize_url_to_fs_path(url)
    if not u:
        return None
    u2 = _strip_leading_dot_segments(u)
    parts = [p for p in u2.replace("\\", "/").lstrip("/").split("/") if p]
    if not parts:
        return None
    head = parts[0]
    if head == "media":
        return "shared"
    if head in LINK_CHAT_LOCAL_MEDIA_DIRS:
        return "chat_local"
    return None

def _target_chat_id_from_url_for_check(url: str) -> Optional[str]:
    if _should_skip_url(url):
        return None
    u = _normalize_url_to_fs_path(url)
    if not u:
        return None
    u2 = _strip_leading_dot_segments(u).replace("\\", "/").lstrip("/")
    m = re.match(r"^chats/chat_(\d+)(?:/|$)", u2)
    if not m:
        return None
    return m.group(1)

def _source_chat_id_from_html_path_for_check(html_path: str, root: str) -> Optional[str]:
    try:
        rel = os.path.relpath(html_path, root).replace("\\", "/")
    except Exception:
        return None
    m = re.search(r"(?:^|/)chats/chat_(\d+)(?:/|$)", rel)
    if not m:
        return None
    return m.group(1)

def _iter_srcset_urls(val: str) -> Iterable[str]:
    for part in val.split(","):
        part = part.strip()
        if not part:
            continue
        url = part.split()[0].strip()
        if url:
            yield url

@dataclass(frozen=True)
class LinkMissingRef:
    html_file: str
    line_no: int
    attr: str  # src/href/poster/srcset/url(...)
    url: str
    resolved_path: str
    note: Optional[str] = None

@dataclass(frozen=True)
class LinkSchemeRef:
    html_file: str
    line_no: int
    attr: str  # src/href/poster/srcset
    url: str

def _scan_html_file_for_bad_refs(
    html_path: str,
    root: str,
    *,
    scope: str,
    allow_outside_root: bool,
    check_split_leftovers: bool,
    is_multi_export_root: bool,
    max_keep: int,
    scheme_prefixes: Optional[Tuple[str, ...]] = None,
    max_keep_schemes: int = 200,
) -> Tuple[
    int,
    int,
    int,
    int,
    List[LinkMissingRef],
    int,
    List[LinkMissingRef],
    int,
    List[LinkMissingRef],
    int,
    List[LinkSchemeRef],
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
]:
    """
    Returns:
      total_refs, skipped_refs, ok_refs,
      missing_total, missing_kept,
      outside_total, outside_kept,
      split_leftover_total, split_leftover_kept,
      scheme_total, scheme_kept,
      media_shared_refs, media_chat_local_refs,
      media_shared_ok, media_chat_local_ok,
      cross_chat_total, cross_chat_ok,
      bucket_non_local, bucket_cross_chat, bucket_media_shared,
      bucket_media_chat_local, bucket_other_local
    """
    total = 0
    skipped = 0
    ok = 0
    missing_total = 0
    outside_total = 0
    split_leftover_total = 0
    missing_kept: List[LinkMissingRef] = []
    outside_kept: List[LinkMissingRef] = []
    split_leftover_kept: List[LinkMissingRef] = []
    scheme_total = 0
    scheme_kept: List[LinkSchemeRef] = []
    media_shared_refs = 0
    media_chat_local_refs = 0
    media_shared_ok = 0
    media_chat_local_ok = 0
    cross_chat_total = 0
    cross_chat_ok = 0
    bucket_non_local = 0
    bucket_cross_chat = 0
    bucket_media_shared = 0
    bucket_media_chat_local = 0
    bucket_other_local = 0
    in_style_block = False
    src_chat_id = _source_chat_id_from_html_path_for_check(html_path, root)

    scheme_prefixes_l = tuple((scheme_prefixes or ()))

    def _maybe_keep_scheme(ln: int, attr: str, url: str) -> None:
        nonlocal scheme_total
        if not scheme_prefixes_l:
            return
        u = url.strip()
        if not u:
            return
        lu = u.lower()
        if any(lu.startswith(p) for p in scheme_prefixes_l):
            scheme_total += 1
            if len(scheme_kept) < max_keep_schemes:
                scheme_kept.append(LinkSchemeRef(html_path, ln, attr, u))

    def _bucket_for(url: str, cls: Optional[str], is_cross_chat: bool) -> str:
        if _should_skip_url(url):
            return "non_local"
        if _looks_like_external_host_ref(url) or _looks_like_bare_file_token_ref(url):
            return "non_local"
        if is_cross_chat:
            return "cross_chat"
        if cls == "shared":
            return "media_shared"
        if cls == "chat_local":
            return "media_chat_local"
        return "other_local"

    def _inc_bucket(name: str) -> None:
        nonlocal bucket_non_local, bucket_cross_chat, bucket_media_shared, bucket_media_chat_local, bucket_other_local
        if name == "non_local":
            bucket_non_local += 1
        elif name == "cross_chat":
            bucket_cross_chat += 1
        elif name == "media_shared":
            bucket_media_shared += 1
        elif name == "media_chat_local":
            bucket_media_chat_local += 1
        else:
            bucket_other_local += 1

    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            for ln, line in enumerate(f, start=1):
                line_l = line.lower()
                # Track <style> blocks so we only treat url(...) in real CSS contexts.
                if "<style" in line_l:
                    in_style_block = True
                if "</style" in line_l:
                    end_style_after = True
                else:
                    end_style_after = False

                if (
                    "href" not in line_l
                    and "src" not in line_l
                    and "poster" not in line_l
                    and "srcset" not in line_l
                    and "url(" not in line_l
                ):
                    if end_style_after:
                        in_style_block = False
                    continue

                if (
                    check_split_leftovers
                    and not is_multi_export_root
                    and os.path.basename(html_path).startswith("messages")
                    and "chats/chat_" in line_l
                ):
                    for m in LINK_MULTI_EXPORT_REF_RE.finditer(line):
                        url = m.group("url")
                        if scope == "media" and not _is_media_url_for_check(url):
                            continue
                        split_leftover_total += 1
                        if len(split_leftover_kept) < max_keep:
                            split_leftover_kept.append(
                                LinkMissingRef(
                                    html_path,
                                    ln,
                                    "multi_export_ref",
                                    url,
                                    "(should be localized into this chat folder)",
                                    "multi-export-leftover",
                                )
                            )

                for m in LINK_ATTR_URL_RE.finditer(line):
                    url = m.group("url")
                    total += 1
                    attr = m.group("prefix").split("=", 1)[0].strip()
                    _maybe_keep_scheme(ln, attr, url)
                    cls = None
                    if _is_media_url_for_check(url):
                        cls = _classify_media_ref_for_check(url)
                        if cls == "shared":
                            media_shared_refs += 1
                        elif cls == "chat_local":
                            media_chat_local_refs += 1
                    tgt_chat_id = _target_chat_id_from_url_for_check(url)
                    is_cross_chat = bool(tgt_chat_id and src_chat_id and tgt_chat_id != src_chat_id)
                    if is_cross_chat:
                        cross_chat_total += 1
                    _inc_bucket(_bucket_for(url, cls, is_cross_chat))
                    if scope == "media" and not _is_media_url_for_check(url):
                        skipped += 1
                        continue
                    resolved, note = _resolve_local_ref_for_check(url, html_path, root)
                    if resolved is None:
                        skipped += 1
                        continue
                    is_outside = bool(note and "outside-root" in note)
                    exists = os.path.exists(resolved)
                    if is_outside and not allow_outside_root:
                        outside_total += 1
                        if len(outside_kept) < max_keep:
                            outside_kept.append(LinkMissingRef(html_path, ln, attr, url, resolved, note))
                        continue
                    if exists:
                        ok += 1
                        if cls == "shared":
                            media_shared_ok += 1
                        elif cls == "chat_local":
                            media_chat_local_ok += 1
                        if is_cross_chat:
                            cross_chat_ok += 1
                        continue
                    missing_total += 1
                    if len(missing_kept) < max_keep:
                        missing_kept.append(LinkMissingRef(html_path, ln, attr, url, resolved, note))

                for m in LINK_ATTR_SRCSET_RE.finditer(line):
                    val = m.group("val")
                    for url in _iter_srcset_urls(val):
                        total += 1
                        _maybe_keep_scheme(ln, "srcset", url)
                        cls = None
                        if _is_media_url_for_check(url):
                            cls = _classify_media_ref_for_check(url)
                            if cls == "shared":
                                media_shared_refs += 1
                            elif cls == "chat_local":
                                media_chat_local_refs += 1
                        tgt_chat_id = _target_chat_id_from_url_for_check(url)
                        is_cross_chat = bool(tgt_chat_id and src_chat_id and tgt_chat_id != src_chat_id)
                        if is_cross_chat:
                            cross_chat_total += 1
                        _inc_bucket(_bucket_for(url, cls, is_cross_chat))
                        if scope == "media" and not _is_media_url_for_check(url):
                            skipped += 1
                            continue
                        resolved, note = _resolve_local_ref_for_check(url, html_path, root)
                        if resolved is None:
                            skipped += 1
                            continue
                        is_outside = bool(note and "outside-root" in note)
                        exists = os.path.exists(resolved)
                        if is_outside and not allow_outside_root:
                            outside_total += 1
                            if len(outside_kept) < max_keep:
                                outside_kept.append(LinkMissingRef(html_path, ln, "srcset", url, resolved, note))
                            continue
                        if exists:
                            ok += 1
                            if cls == "shared":
                                media_shared_ok += 1
                            elif cls == "chat_local":
                                media_chat_local_ok += 1
                            if is_cross_chat:
                                cross_chat_ok += 1
                            continue
                        missing_total += 1
                        if len(missing_kept) < max_keep:
                            missing_kept.append(LinkMissingRef(html_path, ln, "srcset", url, resolved, note))

                for m in HTML_CSS_URL_RE.finditer(line):
                    url = m.group("url")
                    total += 1
                    cls = None
                    if _is_media_url_for_check(url):
                        cls = _classify_media_ref_for_check(url)
                        if cls == "shared":
                            media_shared_refs += 1
                        elif cls == "chat_local":
                            media_chat_local_refs += 1
                    tgt_chat_id = _target_chat_id_from_url_for_check(url)
                    is_cross_chat = bool(tgt_chat_id and src_chat_id and tgt_chat_id != src_chat_id)
                    if is_cross_chat:
                        cross_chat_total += 1
                    _inc_bucket(_bucket_for(url, cls, is_cross_chat))
                    # Only consider url(...) inside actual CSS contexts (style="..." or <style> blocks).
                    if not in_style_block and 'style="' not in line_l and "style='" not in line_l:
                        skipped += 1
                        continue
                    if scope == "media" and not _is_media_url_for_check(url):
                        skipped += 1
                        continue
                    resolved, note = _resolve_local_ref_for_check(url, html_path, root)
                    if resolved is None:
                        skipped += 1
                        continue
                    is_outside = bool(note and "outside-root" in note)
                    exists = os.path.exists(resolved)
                    if is_outside and not allow_outside_root:
                        outside_total += 1
                        if len(outside_kept) < max_keep:
                            outside_kept.append(LinkMissingRef(html_path, ln, "url(...)", url, resolved, note))
                        continue
                    if exists:
                        ok += 1
                        if cls == "shared":
                            media_shared_ok += 1
                        elif cls == "chat_local":
                            media_chat_local_ok += 1
                        if is_cross_chat:
                            cross_chat_ok += 1
                        continue
                    missing_total += 1
                    if len(missing_kept) < max_keep:
                        missing_kept.append(LinkMissingRef(html_path, ln, "url(...)", url, resolved, note))

                if end_style_after:
                    in_style_block = False
    except Exception:
        return 0, 0, 0, 0, [], 0, [], 0, [], 0, [], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    return (
        total,
        skipped,
        ok,
        missing_total,
        missing_kept,
        outside_total,
        outside_kept,
        split_leftover_total,
        split_leftover_kept,
        scheme_total,
        scheme_kept,
        media_shared_refs,
        media_chat_local_refs,
        media_shared_ok,
        media_chat_local_ok,
        cross_chat_total,
        cross_chat_ok,
        bucket_non_local,
        bucket_cross_chat,
        bucket_media_shared,
        bucket_media_chat_local,
        bucket_other_local,
    )

def _cmd_check_links(
    root: str,
    *,
    json_out: bool,
    max_missing: int,
    scope: str,
    allow_outside_root: bool,
    check_split_leftovers: bool,
    fail_fast: bool,
    list_schemes: str,
    max_schemes: int,
    count_media_files: bool,
    _disable_auto_grouping: bool = False,
    _exclude_roots: Optional[Tuple[str, ...]] = None,
) -> int:
    root = os.path.abspath(root)

    # When pointed at a parent folder that contains multiple backups,
    # report each detected backup separately.
    detected_html_roots = [os.path.abspath(p) for p in find_html_export_roots(root)]
    detected_json_roots = [os.path.abspath(os.path.dirname(p)) for p in find_result_jsons(root)]
    detected_sqlite_roots = [os.path.abspath(os.path.dirname(p)) for p in find_unofficial_telegram_sqlite_dbs(root)]
    detected_backup_roots = sorted(
        {
            p
            for p in (detected_html_roots + detected_json_roots + detected_sqlite_roots)
            if os.path.isdir(p)
        },
        key=lambda p: (p.count(os.sep), p),
    )
    if (
        not _disable_auto_grouping
        and not json_out
        and len(detected_backup_roots) > 1
    ):
        print(_c(f"Root: {root}", _Ansi.BOLD))
        print(f"Detected backups: {len(detected_backup_roots)}")
        overall_rc = 0
        for i, dr in enumerate(detected_backup_roots, start=1):
            print()
            print(_c(f"Backup {i}/{len(detected_backup_roots)}: {dr}", _Ansi.BOLD))
            subroots = tuple(
                p for p in detected_backup_roots
                if _is_ancestor_dir(dr, p)
            )
            rc = _cmd_check_links(
                dr,
                json_out=False,
                max_missing=max_missing,
                scope=scope,
                allow_outside_root=allow_outside_root,
                check_split_leftovers=check_split_leftovers,
                fail_fast=fail_fast,
                list_schemes=list_schemes,
                max_schemes=max_schemes,
                count_media_files=count_media_files,
                _disable_auto_grouping=True,
                _exclude_roots=subroots,
            )
            if rc != 0:
                overall_rc = 1
        return overall_rc

    html_files = sorted(_iter_all_html_files(root))
    if _exclude_roots:
        ex = [os.path.abspath(p) for p in _exclude_roots if os.path.abspath(p) != root]
        if ex:
            html_files = [
                p for p in html_files
                if not any(_is_ancestor_dir(e, p) for e in ex)
            ]
    is_multi_export_root = os.path.isfile(os.path.join(root, "export_results.html")) or os.path.isdir(os.path.join(root, "chats"))

    total_refs = 0
    skipped_refs = 0
    ok_refs = 0
    missing_total = 0
    outside_total = 0
    split_leftover_total = 0
    scheme_total = 0
    media_shared_refs_total = 0
    media_chat_local_refs_total = 0
    media_shared_ok_total = 0
    media_chat_local_ok_total = 0
    cross_chat_refs_total = 0
    cross_chat_refs_ok_total = 0
    bucket_non_local_total = 0
    bucket_cross_chat_total = 0
    bucket_media_shared_total = 0
    bucket_media_chat_local_total = 0
    bucket_other_local_total = 0
    missing_kept: List[LinkMissingRef] = []
    outside_kept: List[LinkMissingRef] = []
    split_leftover_kept: List[LinkMissingRef] = []
    scheme_kept: List[LinkSchemeRef] = []

    scheme_names = [s.strip().lower() for s in list_schemes.split(",") if s.strip()]
    scheme_prefixes: Optional[Tuple[str, ...]] = tuple(f"{s}:" for s in scheme_names) if scheme_names else None

    for p in html_files:
        (
            t,
            s,
            ok,
            m_total,
            m_kept,
            o_total,
            o_kept,
            sl_total,
            sl_kept,
            sc_total,
            sc_kept,
            ms_total,
            ml_total,
            ms_ok,
            ml_ok,
            xc_total,
            xc_ok,
            b_non_local,
            b_cross_chat,
            b_media_shared,
            b_media_chat_local,
            b_other_local,
        ) = _scan_html_file_for_bad_refs(
            p,
            root,
            scope=scope,
            allow_outside_root=allow_outside_root,
            check_split_leftovers=check_split_leftovers,
            is_multi_export_root=is_multi_export_root,
            max_keep=max_missing,
            scheme_prefixes=scheme_prefixes,
            max_keep_schemes=max_schemes,
        )
        total_refs += t
        skipped_refs += s
        ok_refs += ok
        missing_total += m_total
        outside_total += o_total
        split_leftover_total += sl_total
        scheme_total += sc_total
        media_shared_refs_total += ms_total
        media_chat_local_refs_total += ml_total
        media_shared_ok_total += ms_ok
        media_chat_local_ok_total += ml_ok
        cross_chat_refs_total += xc_total
        cross_chat_refs_ok_total += xc_ok
        bucket_non_local_total += b_non_local
        bucket_cross_chat_total += b_cross_chat
        bucket_media_shared_total += b_media_shared
        bucket_media_chat_local_total += b_media_chat_local
        bucket_other_local_total += b_other_local

        for m in m_kept:
            if len(missing_kept) >= max_missing:
                break
            missing_kept.append(m)
        for m in o_kept:
            if len(outside_kept) >= max_missing:
                break
            outside_kept.append(m)
        for m in sl_kept:
            if len(split_leftover_kept) >= max_missing:
                break
            split_leftover_kept.append(m)
        for m in sc_kept:
            if len(scheme_kept) >= max_schemes:
                break
            scheme_kept.append(m)

        if (m_total or o_total or sl_total) and fail_fast:
            break

    shared_media_files = 0
    chat_local_media_files = 0
    if count_media_files:
        seen_chat_roots: Set[str] = set()
        for cr in _iter_chat_export_roots(root):
            if cr in seen_chat_roots:
                continue
            seen_chat_roots.add(cr)
            shared_p = os.path.join(cr, "media")
            if os.path.isdir(shared_p):
                shared_media_files += _count_files_recursive(shared_p)
            for d in LINK_CHAT_LOCAL_MEDIA_DIRS:
                lp = os.path.join(cr, d)
                if os.path.isdir(lp):
                    chat_local_media_files += _count_files_recursive(lp)

    if json_out:
        out = {
            "root": root,
            "is_multi_export_root": is_multi_export_root,
            "html_files_scanned": len(html_files),
            "total_refs": total_refs,
            "skipped_refs": skipped_refs,
            "ok_refs": ok_refs,
            "media_shared_refs_total": media_shared_refs_total,
            "media_chat_local_refs_total": media_chat_local_refs_total,
            "media_shared_refs_ok": media_shared_ok_total,
            "media_chat_local_refs_ok": media_chat_local_ok_total,
            "cross_chat_refs_total": cross_chat_refs_total,
            "cross_chat_refs_ok": cross_chat_refs_ok_total,
            "bucket_non_local_total": bucket_non_local_total,
            "bucket_cross_chat_total": bucket_cross_chat_total,
            "bucket_media_shared_total": bucket_media_shared_total,
            "bucket_media_chat_local_total": bucket_media_chat_local_total,
            "bucket_other_local_total": bucket_other_local_total,
            "scheme_refs_total": scheme_total if scheme_prefixes else None,
            "scheme_refs": [
                {"html_file": m.html_file, "line_no": m.line_no, "attr": m.attr, "url": m.url}
                for m in scheme_kept
            ]
            if scheme_prefixes
            else [],
            "shared_media_files": shared_media_files if count_media_files else None,
            "chat_local_media_files": chat_local_media_files if count_media_files else None,
            "split_leftover_refs": [
                {"html_file": m.html_file, "line_no": m.line_no, "attr": m.attr, "url": m.url, "note": m.note}
                for m in split_leftover_kept
            ],
            "split_leftover_refs_total": split_leftover_total,
            "outside_root_refs": [
                {
                    "html_file": m.html_file,
                    "line_no": m.line_no,
                    "attr": m.attr,
                    "url": m.url,
                    "resolved_path": m.resolved_path,
                    "note": m.note,
                }
                for m in outside_kept
            ],
            "outside_root_refs_total": outside_total,
            "missing_refs": [
                {
                    "html_file": m.html_file,
                    "line_no": m.line_no,
                    "attr": m.attr,
                    "url": m.url,
                    "resolved_path": m.resolved_path,
                    "note": m.note,
                }
                for m in missing_kept
            ],
            "missing_refs_total": missing_total,
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 1 if (missing_total or outside_total or split_leftover_total) else 0

    print(_c(f"Root: {root}", _Ansi.BOLD))
    print(f"HTML files scanned: {len(html_files)}")
    bucket_sum = (
        bucket_non_local_total
        + bucket_cross_chat_total
        + bucket_media_shared_total
        + bucket_media_chat_local_total
        + bucket_other_local_total
    )
    print(
        "Ref classes (exclusive): "
        f"total={total_refs} "
        f"non_files={bucket_non_local_total} "
        f"cross_chat={bucket_cross_chat_total} "
        f"media_shared={bucket_media_shared_total} "
        f"chat_local={bucket_media_chat_local_total} "
        f"html_stuff={bucket_other_local_total}"
    )
    if bucket_sum != total_refs:
        print(_c(f"Note: class sum={bucket_sum} differs from total={total_refs}", _Ansi.YELLOW))
    if count_media_files:
        print(
            "Media files: "
            f"shared_in_media_dir={shared_media_files} "
            f"chat_local_in_dirs={chat_local_media_files}"
        )
    if scheme_prefixes:
        print(f"Scheme refs (in HTML attrs): total={scheme_total} ({', '.join(scheme_names)})")

    if not missing_total and not outside_total and not split_leftover_total:
        print()
        print(_c("All local references resolved.", _Ansi.GREEN))
        return 0

    print()
    if split_leftover_total and not is_multi_export_root:
        print(_c("Leftover multi-export links found in messages*.html (should be localized):", _Ansi.YELLOW))
        for i, m in enumerate(split_leftover_kept[:max_missing], start=1):
            rel_html = os.path.relpath(m.html_file, root)
            print(f"{i:>4}. {rel_html}:{m.line_no} {m.url}")
        if split_leftover_total > len(split_leftover_kept):
            print(_c(f"... and {split_leftover_total - len(split_leftover_kept)} more", _Ansi.DIM))
        print()

    if outside_total:
        print(_c("References that resolve outside the provided root:", _Ansi.YELLOW))
        for i, m in enumerate(outside_kept[:max_missing], start=1):
            rel_html = os.path.relpath(m.html_file, root)
            note = f" [{m.note}]" if m.note else ""
            print(f"{i:>4}. {rel_html}:{m.line_no} {m.attr}=\"{m.url}\" -> {m.resolved_path}{note}")
        if outside_total > len(outside_kept):
            print(_c(f"... and {outside_total - len(outside_kept)} more", _Ansi.DIM))
        print()

    if missing_total:
        print(_c("Missing local references:", _Ansi.RED))
        for i, m in enumerate(missing_kept[:max_missing], start=1):
            rel_html = os.path.relpath(m.html_file, root)
            note = f" [{m.note}]" if m.note else ""
            print(f"{i:>4}. {rel_html}:{m.line_no} {m.attr}=\"{m.url}\" -> {m.resolved_path}{note}")
        if missing_total > len(missing_kept):
            print(_c(f"... and {missing_total - len(missing_kept)} more", _Ansi.DIM))

    if scheme_prefixes and scheme_total:
        print()
        print(_c("Scheme refs found in HTML attributes:", _Ansi.CYAN))
        for i, m in enumerate(scheme_kept[:max_schemes], start=1):
            rel_html = os.path.relpath(m.html_file, root)
            print(f"{i:>4}. {rel_html}:{m.line_no} {m.attr}=\"{m.url}\"")
        if scheme_total > len(scheme_kept):
            print(_c(f"... and {scheme_total - len(scheme_kept)} more", _Ansi.DIM))

    return 1

@dataclass
class ChatSummary:
    name: str
    messages_backed_up: Optional[int]
    first_message_utc: Optional[str]
    last_message_utc: Optional[str]


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
    chat_summaries: Optional[List[ChatSummary]] = None

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
    lines = [ANSI_ESCAPE_RE.sub("", ln.strip("\n")) for ln in stdout.splitlines() if ln.strip()]
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
            for args in (
                # Newer dust uses --no-colors (plural); keep --no-color for compat.
                [dust, "-b", "--depth", "0", "--no-colors", p],
                [dust, "-b", "-d", "0", "--no-colors", p],
                [dust, "-b", "--depth", "0", "--no-color", p],
                [dust, "-b", "-d", "0", "--no-color", p],
                [dust, "-b", "--depth", "0", p],
                [dust, "-b", "-d", "0", p],
                # Fallback to human output if -b isn't supported.
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

def _bulk_dir_sizes_via_du(root: str) -> Tuple[Dict[str, int], str]:
    """
    Single-pass directory sizing.

    This is dramatically faster than calling `du -sb` repeatedly for many nested
    directories because it walks the filesystem once and reports all subdir totals.
    """
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
            # GNU du output: "<bytes>\t<path>"
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

def _parse_dust_tree(stdout: str, root: str) -> Dict[str, int]:
    """
    Parse `dust -b` output. We map whatever "path label" dust prints back to an
    absolute path under `root`.
    """
    root = os.path.abspath(root)
    out: Dict[str, int] = {}

    def _strip_dust_columns(s: str) -> str:
        # Many dust versions append a bar/percent table separated by the box-drawing
        # vertical bar. Chat names can contain ASCII '|' so we only split on '│'.
        if "│" in s:
            s = s.split("│", 1)[0]
        # Some builds may not use the box-drawing separator; still strip a trailing
        # percent column if present (e.g. "...   24%").
        s = re.sub(r"\s+\d+%\s*$", "", s)
        return s.strip().rstrip("/")

    # dust output format varies a bit between versions/configs. It may print:
    # - absolute paths (common)
    # - basenames with tree indentation (like `tree`)
    # - relative paths (common) like `chats/chat_001` or `./chats/chat_001`, either
    #   as a tree or as a flat list (no glyphs).
    #
    # We support all of the above:
    # - If the line contains `root`, we grab the substring starting at `root` and
    #   treat it as the path label. This handles spaces in directory names.
    # - Else, if it looks like tree output, we reconstruct from indentation.
    # - Else, try to recover a relative path by finding the first token that looks
    #   like a path and taking the rest of the line from there (preserves spaces).
    stack: List[str] = []
    for raw in stdout.splitlines():
        line = ANSI_ESCAPE_RE.sub("", raw.rstrip("\n"))
        if not line.strip():
            continue

        # Parse size: find the first token that parses as a size (bytes or human).
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

        # Fast path: dust prints full paths (including root) on each line.
        if root in line:
            idx = line.find(root)
            path_label = line[idx:].strip().rstrip("/")
            p = os.path.abspath(path_label)
            if _is_ancestor_dir(root, p):
                out[p] = b
            else:
                # If `root` occurs in other columns (rare), fall back to tree parsing.
                pass
            continue

        # Find tree branch marker if present.
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
            # Flat-list output: try to find a path-like token after the size.
            # Example forms:
            #   12345  chats/alex
            #   4.00KiB  ./chats/Alternative group
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
                        # Keep stack in sync for any subsequent tree-like lines.
                        rel2 = os.path.relpath(p, root)
                        stack = [] if rel2 == "." else rel2.split(os.sep)
                        continue

            # Another flat form is just: "<size> <name>" (no slashes, no tree glyphs),
            # which is common when depth=1 and the tool prints basenames.
            if size_i >= 0 and size_i < len(toks):
                size_tok = toks[size_i]
                j = line.find(size_tok)
                if j != -1:
                    cand = _strip_dust_columns(line[j + len(size_tok) :])
                    # Drop any leading punctuation that might remain after stripping.
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

            # Root line: dust commonly uses "." or prints the input path.
            # If unsure, treat the first parsed line as the root total.
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

        # Determine depth by counting indent blocks right before the marker.
        # Most builds use 4-char blocks ("│   " / "    "), but tolerate 3-char ASCII.
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

        # Map label -> absolute path under root.
        p: Optional[str] = None
        if not out:
            # First line may be shown as "└─┬ <basename>" with bars. Treat it as root.
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
            # Some dust versions print relative paths even at deeper levels.
            rel = name
            if rel.startswith("./"):
                rel = rel[2:]
            rel = rel.lstrip(os.sep)
            p = os.path.abspath(os.path.join(root, rel))
            if _is_ancestor_dir(root, p):
                rel2 = os.path.relpath(p, root)
                stack = [] if rel2 == "." else rel2.split(os.sep)
        else:
            # Basename-only output: maintain a stack of path components.
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
    """
    One dust invocation over `root`, returning directory totals for paths under it.
    This matches the speed profile of running `dust <root>` manually.
    """
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

    # Note: dust defaults to showing only a limited number of lines.
    # For export roots with hundreds of chat folders, we must raise the limit or
    # we'll miss many sizes and print '?'.
    arg_sets: List[List[str]] = []

    # Prefer the user's typical fast output (human units + tree glyphs), then fall
    # back to bytes output if available. Some dust builds don't support short flags
    # for depth/lines, so we try both long and short forms.
    base0 = [dust, "--depth", str(max_depth), "--no-colors", root]
    base1 = [dust, "--depth", str(max_depth), root]
    base0s = [dust, "-d", str(max_depth), "--no-colors", root]
    base1s = [dust, "-d", str(max_depth), root]

    baseb0 = [dust, "-b", "--depth", str(max_depth), "--no-colors", root]
    baseb1 = [dust, "-b", "--depth", str(max_depth), root]
    baseb0s = [dust, "-b", "-d", str(max_depth), "--no-colors", root]
    baseb1s = [dust, "-b", "-d", str(max_depth), root]
    if n_lines is not None:
        # Long-form flags first.
        arg_sets.append([dust, "--depth", str(max_depth), "--number-of-lines", str(n_lines), "--no-colors", root])
        arg_sets.append([dust, "--depth", str(max_depth), "--number-of-lines", str(n_lines), root])
        arg_sets.append([dust, "-b", "--depth", str(max_depth), "--number-of-lines", str(n_lines), "--no-colors", root])
        arg_sets.append([dust, "-b", "--depth", str(max_depth), "--number-of-lines", str(n_lines), root])

        # Short-form flags.
        arg_sets.append([dust, "-d", str(max_depth), "-n", str(n_lines), "--no-colors", root])
        arg_sets.append([dust, "-d", str(max_depth), "-n", str(n_lines), root])
        arg_sets.append([dust, "-b", "-d", str(max_depth), "-n", str(n_lines), "--no-colors", root])
        arg_sets.append([dust, "-b", "-d", str(max_depth), "-n", str(n_lines), root])

    arg_sets.append(base0)
    arg_sets.append(base1)
    arg_sets.append(baseb0)
    # Compat: some builds use --no-color (singular). Try it too.
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
    """
    Compute sizes for multiple export roots using a single dust invocation over scan_root.
    Returns: (abs_path -> bytes|None, tool_used).
    """
    scan_root = os.path.abspath(scan_root)
    abs_roots = [os.path.abspath(p) for p in export_roots]

    # We'll try dust first (fast), then fall back to a single-pass du for robustness.
    out: Dict[str, Optional[int]] = {p: None for p in abs_roots}
    dust = shutil.which("dust")
    if not dust:
        du_map, _tool = _bulk_dir_sizes_via_du(scan_root)
        if du_map:
            for p in abs_roots:
                out[p] = du_map.get(p)
            return out, "du"
        return out, "unknown"

    # Compute the minimum depth needed to include all export roots.
    max_depth = 0
    for p in abs_roots:
        if not _is_ancestor_dir(scan_root, p):
            continue
        rel = os.path.relpath(p, scan_root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        max_depth = max(max_depth, depth)

    # Ensure we don't truncate output. Some dust builds apply the line limit globally,
    # so for depth>1 we may need roughly (top-level dirs + export roots) lines.
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
        """
        True if `child` is the only non-hidden entry inside `parent` and it's a dir.
        We ignore hidden entries (dotfiles) since exporters sometimes drop them.
        """
        try:
            child_base = os.path.basename(child)
            # Only consider direct filesystem parent/child.
            if os.path.abspath(os.path.dirname(child)) != os.path.abspath(parent):
                return False
            subdirs: List[str] = []
            # Any non-hidden files mean parent size != child size.
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
        """
        If dust didn't print `p` but did print an ancestor, we can safely re-use the
        ancestor size only when the path from ancestor -> p is a single-child chain.
        This is common for the split layout: ChatName/START__END.
        """
        cur = os.path.abspath(p)
        # Walk up until we hit something dust reported.
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

        # Ensure ancestor contains only the subdir chain down to p.
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

def _dir_size_bytes_dust_only(path: str) -> Optional[int]:
    dust = shutil.which("dust")
    if not dust:
        return None
    p = os.path.abspath(path)
    for args in (
        [dust, "-b", "-d", "0", "--no-color", p],
        [dust, "-b", "-d", "0", p],
    ):
        stdout = _dust_call(args)
        if not stdout:
            continue
        b = _parse_dust_total(stdout, p)
        if b is not None:
            return b
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

def _render_export_tree(root: str, reports: List[ExportReport]) -> str:
    root = os.path.abspath(root)

    # Nodes are: input root + each export root.
    nodes: List[str] = [root]
    for r in reports:
        p = os.path.abspath(r.export_root)
        if p not in nodes:
            nodes.append(p)

    # Also include intermediate directories between root and each export root so the
    # printed tree matches the actual nesting on disk (e.g. ChatName/START__END).
    # This keeps node counts small (only ancestors of discovered export roots).
    extra: Set[str] = set()
    for n in list(nodes):
        if n == root:
            continue
        cur = os.path.abspath(os.path.dirname(n))
        while _is_ancestor_dir(root, cur) and cur != root:
            extra.add(cur)
            cur2 = os.path.abspath(os.path.dirname(cur))
            if cur2 == cur:
                break
            cur = cur2
    for p in sorted(extra):
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
    # Prefer dust for speed. Run dust once on the root at a depth large enough to
    # include all export roots, then map the printed labels back to absolute paths.
    size_cache: Dict[str, Tuple[Optional[int], str]] = {}
    tools: Set[str] = set()
    max_depth = 0
    for n in nodes:
        if n == root:
            continue
        rel = os.path.relpath(n, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > max_depth:
            max_depth = depth

    # Tell dust to print enough entries at the first level to cover all unique
    # top-level directories that contain exports (common case: hundreds of chats).
    first_level: Set[str] = set()
    for n in nodes:
        if n == root:
            continue
        rel = os.path.relpath(n, root)
        if rel == ".":
            continue
        first = rel.split(os.sep, 1)[0]
        if first:
            first_level.add(first)
    dust_lines = max(200, len(first_level) + 50)

    dust_map, dust_tool = _bulk_dir_sizes_via_dust_tree(root, max_depth=max_depth, max_lines=dust_lines)
    if dust_map:
        tools.add(dust_tool)

    # Important for performance: do NOT fall back to per-node dust calls here.
    # If parsing misses paths, we'd end up spawning dust hundreds of times, which
    # dominates runtime. Instead, show "?" for missing sizes.
    for n in nodes:
        b = dust_map.get(n)
        size_cache[n] = (b, "dust" if b is not None else "unknown")
        tools.add(size_cache[n][1])

    # Heuristic to eliminate '?' for the common case where a container directory
    # contains exactly one child directory (e.g. ChatName contains only START__END).
    # If dust reported the parent size but not the child size, treat the child size
    # as the parent's total. This avoids slow per-node sizing and matches reality
    # for the 1-child layout produced by --split-multi-html.
    def _infer_single_child_size(child: str) -> Optional[int]:
        p = parents.get(child)
        if not p or p == child:
            return None
        pb, _pt = size_cache.get(p, (None, "unknown"))
        if pb is None:
            return None
        try:
            # Only consider direct filesystem parent/child.
            if os.path.abspath(os.path.dirname(child)) != os.path.abspath(p):
                return None
            if not os.path.isdir(p) or not os.path.isdir(child):
                return None
            child_base = os.path.basename(child)
            subdirs: List[str] = []
            for ent in os.scandir(p):
                if not ent.is_dir(follow_symlinks=False):
                    continue
                name = ent.name
                if name.startswith("."):
                    continue
                subdirs.append(name)
                if len(subdirs) > 1:
                    break
            if len(subdirs) == 1 and subdirs[0] == child_base:
                return pb
        except Exception:
            return None
        return None

    for n in nodes:
        b, tool = size_cache.get(n, (None, "unknown"))
        if b is None:
            inferred = _infer_single_child_size(n)
            if inferred is not None:
                size_cache[n] = (inferred, "dust(parent)")
                tools.add("dust(parent)")

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

def _repair_local_html_links_in_place(root: str, *, apply_changes: bool) -> Tuple[int, int, int]:
    """
    Rewrite only broken local HTML links with unescaped '#' in filenames
    to safe URL-encoded relative paths (in-place).

    Returns: (html_files_scanned, html_files_changed, links_rewritten)
    """
    root = os.path.abspath(root)
    html_files = sorted(_iter_all_html_files(root))
    changed_files = 0
    rewritten = 0

    def _quote_url_path(rel_posix: str) -> str:
        return urllib.parse.quote(rel_posix, safe="/")

    for p in html_files:
        src_dir = os.path.dirname(p)
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                s = f.read()
        except Exception:
            continue

        changed = False

        def _rewrite_url(url: str) -> Optional[str]:
            nonlocal rewritten, changed
            if _should_skip_url(url):
                return None
            raw = html.unescape(url).strip()
            raw = raw.split("?", 1)[0].strip()
            # Only target unescaped hash links (the browser-truncation case).
            if not raw or "#" not in raw or "%23" in raw.lower():
                return None
            # Require that the literal path containing '#' exists as a local file.
            # If only the pre-# part exists (e.g. page.html#anchor), skip.
            if os.path.isabs(raw):
                abs_raw = os.path.abspath(os.path.normpath(raw))
            else:
                abs_raw = os.path.abspath(os.path.normpath(os.path.join(src_dir, raw)))
            if not _is_ancestor_dir(root, abs_raw) or not os.path.isfile(abs_raw):
                return None
            rel_local = os.path.relpath(abs_raw, src_dir).replace(os.sep, "/")
            nu = _quote_url_path(rel_local)
            if nu == url:
                return None
            rewritten += 1
            changed = True
            return nu

        def _sub_attr(m: re.Match) -> str:
            prefix = m.group("prefix")
            q = m.group("q")
            url = m.group("url")
            nu = _rewrite_url(url)
            if not nu:
                return m.group(0)
            return f"{prefix}{q}{nu}{q}"

        def _sub_srcset(m: re.Match) -> str:
            prefix = m.group("prefix")
            q = m.group("q")
            val = m.group("val")
            parts = []
            local_changed = False
            for part in val.split(","):
                raw = part.strip()
                if not raw:
                    continue
                bits = raw.split()
                if not bits:
                    continue
                url0 = bits[0]
                nu = _rewrite_url(url0)
                if nu:
                    bits[0] = nu
                    local_changed = True
                parts.append(" ".join(bits))
            if not local_changed:
                return m.group(0)
            return f"{prefix}{q}{', '.join(parts)}{q}"

        def _sub_css_url(m: re.Match) -> str:
            q = m.group("q")
            url = m.group("url")
            nu = _rewrite_url(url)
            if not nu:
                return m.group(0)
            return f"url({q}{nu}{q})"

        s2 = LINK_ATTR_URL_RE.sub(_sub_attr, s)
        s2 = LINK_ATTR_SRCSET_RE.sub(_sub_srcset, s2)
        s2 = HTML_CSS_URL_RE.sub(_sub_css_url, s2)

        if s2 != s:
            changed_files += 1
            if apply_changes:
                try:
                    with open(p, "w", encoding="utf-8") as f:
                        f.write(s2)
                except Exception:
                    continue

    return len(html_files), changed_files, rewritten

def _should_skip_url(url: str) -> bool:
    u = url.strip()
    if not u:
        return True
    if u.startswith("#"):
        return True
    lu = u.lower()
    if lu.startswith(("http://", "https://", "mailto:", "javascript:", "data:", "tg:", "tel:")):
        return True
    # Schemes like "file:" are not expected in Telegram exports; ignore any scheme-ish URLs.
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", u):
        return True
    return False

def _resolve_local_url_to_export_file(
    url: str,
    src_html_dir: str,
    export_root: str,
) -> Optional[Tuple[str, str]]:
    """
    Resolve a local URL (href/src/url(...)) to an existing file under export_root.
    Returns (abs_path, rel_path_from_export_root) or None.
    """
    if _should_skip_url(url):
        return None

    u0 = html.unescape(url).strip()
    # Strip query; local file paths ignore it.
    u0 = u0.split("?", 1)[0].strip()
    if not u0:
        return None

    # Telegram exports are inconsistent:
    # - Sometimes the on-disk filename is percent-encoded literally (e.g. "U2%20...mp3"),
    #   and the HTML uses the same (so we must NOT unquote).
    # - Sometimes the HTML is percent-encoded but the on-disk filename is decoded
    #   (so we MUST unquote).
    #
    # Also, Telegram sometimes emits links with an unescaped '#' in the filename
    # (broken in browsers). For splitting/copying, we still want to resolve the
    # underlying file so we can rewrite to a safe %23 URL later.
    candidates = [u0]
    u_no_frag = u0.split("#", 1)[0].strip()
    if u_no_frag and u_no_frag != u0:
        candidates.append(u_no_frag)

    seen: Set[str] = set()
    seen_abs: Set[str] = set()

    def _consider_candidate(rel_or_abs: str) -> Optional[Tuple[str, str]]:
        # rel_or_abs may be absolute or relative to src_html_dir.
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
        # Try raw first (handles literal %xx filenames), then decoded variants.
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

            # Fallback for ambiguous bare filenames often found in Telegram HTML:
            # try common chat-local media dirs (e.g. files/, photos/, ...).
            if "/" not in v and "\\" not in v and not v.startswith("."):
                for d in sorted(LINK_CHAT_LOCAL_MEDIA_DIRS):
                    hit = _consider_candidate(os.path.join(d, v))
                    if hit:
                        return hit

    return None

def _copy_and_localize_shared_media_for_chat(
    export_root: str,
    src_chat: str,
    dst_chat: str,
    chat_id: str,
) -> None:
    """
    Copy any "shared" media referenced by a chat's messages*.html (typically ../../photos/...,
    ../../files/..., etc.) into dst_chat/media/<relative-from-export-root>/ and rewrite the
    HTML to point to media/... so the chat export is portable.

    Safety: never deletes or modifies the source export.
    """
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

    # For files that live outside this chat folder (shared across the export), we copy them
    # into dst_chat/media/<rel-from-export-root>/ and rewrite links to media/<url-encoded rel>.
    abs_to_media_relposix: Dict[str, str] = {}

    # First pass: scan source HTML for referenced files under export_root (excluding chat-local content).
    for src_html in src_msg_files:
        src_html_dir = os.path.dirname(src_html)
        try:
            with open(src_html, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    # Fast path: skip regexes on most lines.
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
        # Encode reserved characters so the URL resolves to the exact on-disk path.
        # Important for literal '%' in filenames (e.g. "U2%20..." should become "U2%2520...").
        return urllib.parse.quote(rel_posix, safe="/")

    # Copy referenced media into dst_chat/media/<relpath>.
    media_root = os.path.join(dst_chat, "media")
    for abs_p, rel_posix in abs_to_media_relposix.items():
        dp = os.path.join(media_root, rel_posix.replace("/", os.sep))
        os.makedirs(os.path.dirname(dp), exist_ok=True)
        try:
            shutil.copy2(abs_p, dp)
        except Exception:
            # Best-effort: missing/unreadable files should not abort the entire split.
            continue

    # Second pass: rewrite the copied HTML files in dst_chat to point at media/... paths.
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
                # Chat-local file: rewrite to a safe, URL-encoded relative path within the chat.
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
        # Important: use the same logic as inspection (timestamps + day-separator fallback),
        # otherwise chats with only service/day-separator entries become unknown__unknown.
        _, first_dt, last_dt, _, _, _ = _scan_html_message_files(src_msg_files)
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

def _is_unofficial_telegram_sqlite_db(db_path: str) -> bool:
    """
    Heuristic detector for the unofficial Telegram backup DB used by pre-export tools.
    We only look for expected tables.
    """
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

def _resolve_existing_under_root(
    url: str,
    base_dir: str,
    root: str,
) -> Optional[str]:
    """
    Resolve a (local) URL relative to base_dir and return an existing absolute path
    under root. Tries raw and unquoted variants to handle Telegram exports that
    store literal %xx sequences in filenames.
    """
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
            abs_p = os.path.abspath(os.path.normpath(os.path.join(base_dir, v)))
            if not _is_ancestor_dir(root, abs_p):
                continue
            if os.path.isfile(abs_p):
                return abs_p
    return None

def _quote_url_path_posix(rel_posix: str) -> str:
    # Quote reserved characters so file:// resolution hits the exact on-disk name.
    # Most important: literal '%' and '#'.
    return urllib.parse.quote(rel_posix, safe="/")

def _resolve_multi_chat_ref_to_abs(url: str, multi_root: str) -> Optional[Tuple[str, str]]:
    """
    If url contains chats/chat_XXX/... return (abs_path, rel_posix_from_multi_root).
    """
    if _should_skip_url(url):
        return None

    u0 = html.unescape(url).strip()
    u0 = u0.split("?", 1)[0].strip()
    if not u0:
        return None

    marker = "chats/chat_"
    idx = u0.find(marker)
    if idx < 0:
        return None

    rel0 = u0[idx:]
    rels = [rel0]
    rel_no_frag = rel0.split("#", 1)[0].strip()
    if rel_no_frag and rel_no_frag != rel0:
        rels.append(rel_no_frag)

    seen: Set[str] = set()
    for rel in rels:
        variants = [rel]
        try:
            v1 = urllib.parse.unquote(rel)
            variants.append(v1)
            v2 = urllib.parse.unquote(v1)
            variants.append(v2)
        except Exception:
            pass

        for v in variants:
            if not v or v in seen:
                continue
            seen.add(v)

            abs_p = os.path.abspath(os.path.normpath(os.path.join(multi_root, v)))
            if not _is_ancestor_dir(multi_root, abs_p):
                continue
            if os.path.isfile(abs_p):
                rel_posix = os.path.relpath(abs_p, multi_root).replace(os.sep, "/")
                return abs_p, rel_posix
    return None

def find_unofficial_telegram_sqlite_dbs(root: str) -> List[str]:
    hits: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn in UNOFFICIAL_SQLITE_NAMES:
                p = os.path.join(dirpath, fn)
                if _is_unofficial_telegram_sqlite_db(p):
                    hits.append(p)
    return sorted(set(hits))

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
        # Returns (name_col, first_col, last_col)
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

        # Group by source_type/source_id: maps to users (dialogs) and chats (groups).
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

    # One full streaming pass: count chats, count messages, track min/max time.
    with open(path, "rb") as f:
        for prefix, event, value in ijson_mod.parse(f):
            # Count chats (only meaningful for multi-chat exports)
            if kind == "multi_chat_export":
                # Chat objects: chats.list.item.id (distinct from message IDs which are chats.list.item.messages.item.id)
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

            # Count messages by message id field (avoid reply_to_message_id etc.)
            if prefix.endswith(".messages.item.id") or prefix == "messages.item.id":
                if event in ("number", "string"):
                    msg_count += 1
                    if kind == "multi_chat_export" and prefix.startswith("chats.list.item.messages.item."):
                        cur_chat_msgs += 1

            # Prefer unixtime if present
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

            # Fallback to ISO date strings
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
        chat_summaries: List[ChatSummary] = []
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
        chat_summaries = []
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
        # One pass: compute per-chat summaries and aggregate totals/range.
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
        "--no-sizes",
        action="store_true",
        help="Do not compute on-disk sizes for discovered exports (faster on slow disks).",
    )
    ap.add_argument(
        "--dedupe-unofficial-sqlite",
        action="store_true",
        help=(
            "For unofficial SQLite backups: suppress database.sqlite files outside .telegram_backup/ "
            "when a .telegram_backup DB exists under the same folder (avoids duplicates)."
        ),
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
    ap.add_argument(
        "--check-links",
        action="store_true",
        help="Scan all .html files under the given path and verify that local refs resolve on disk.",
    )
    ap.add_argument(
        "--repair-html-links",
        action="store_true",
        help=(
            "Repair broken local href/src/poster/srcset/url(...) links with unescaped '#'"
            " in .html files by URL-encoding the filename."
        ),
    )
    ap.add_argument(
        "--repair-dry-run",
        action="store_true",
        help="With --repair-html-links: report how many files/links would change without writing files.",
    )
    ap.add_argument("--check-scope", choices=("all", "media"), default="all")
    ap.add_argument("--check-max-missing", type=int, default=200)
    ap.add_argument("--check-allow-outside-root", action="store_true")
    ap.add_argument("--check-no-split-leftovers-check", action="store_true")
    ap.add_argument("--check-fail-fast", action="store_true")
    ap.add_argument("--check-list-schemes", default="")
    ap.add_argument("--check-max-schemes", type=int, default=200)
    ap.add_argument("--check-count-media-files", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.path)
    if not os.path.exists(root):
        print(f"Path does not exist: {root}", file=sys.stderr)
        return 2

    action_flags = [
        ("--split-multi-html", args.split_multi_html),
        ("--check-links", args.check_links),
        ("--repair-html-links", args.repair_html_links),
    ]
    enabled = [name for name, on in action_flags if on]
    if len(enabled) > 1:
        print(f"Only one action flag may be used at a time; got: {', '.join(enabled)}", file=sys.stderr)
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

    if args.check_links:
        return _cmd_check_links(
            root,
            json_out=args.json,
            max_missing=args.check_max_missing,
            scope=args.check_scope,
            allow_outside_root=args.check_allow_outside_root,
            check_split_leftovers=not args.check_no_split_leftovers_check,
            fail_fast=args.check_fail_fast,
            list_schemes=args.check_list_schemes,
            max_schemes=args.check_max_schemes,
            count_media_files=args.check_count_media_files,
        )

    if args.repair_html_links:
        scanned, changed, rewritten = _repair_local_html_links_in_place(
            root,
            apply_changes=not args.repair_dry_run,
        )
        mode = "dry-run" if args.repair_dry_run else "applied"
        print(f"Repair mode: {mode}")
        print(f"HTML files scanned: {scanned}")
        print(f"HTML files changed: {changed}")
        print(f"Links rewritten: {rewritten}")
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

    sqlite_hits = find_unofficial_telegram_sqlite_dbs(root)
    if sqlite_hits:
        if args.dedupe_unofficial_sqlite:
            # Many unofficial backups include both:
            # - <root>/database.sqlite (often a copied/derived view)
            # - <root>/**/.telegram_backup/<account>/database.sqlite (canonical backing store)
            # When both exist under the same folder, prefer the .telegram_backup DB to avoid duplicates.
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
                    # Suppress this DB if there's any canonical DB nested under the same folder.
                    if any(os.path.commonpath([d, os.path.abspath(q)]) == d for q in canon):
                        continue
                    filtered.append(p)
                sqlite_hits = filtered
        reports.extend(inspect_unofficial_telegram_sqlite(p, no_inspect=args.no_inspect) for p in sqlite_hits)

    if not reports:
        print("No Telegram exports found under the given path.", file=sys.stderr)
        print("Expected either:", file=sys.stderr)
        print("- JSON exports: result.json/results.json", file=sys.stderr)
        print("- HTML exports: export_results.html (multi-chat) or messages.html + css/ + js/ (single-chat)", file=sys.stderr)
        print("- Unofficial SQLite backups: database.sqlite (with chats/messages/users tables)", file=sys.stderr)
        return 1

    # Nested export detection: multiple export roots under the input root
    # (This is just a count + listing; you can extend grouping if you want.)
    if args.json:
        print(json.dumps([asdict(r) for r in reports], indent=2, ensure_ascii=False))
    else:
        # Pre-compute sizes once (dust, with du fallback) for all export roots.
        size_tool_used: Set[str] = set()
        size_by_export_root: Dict[str, Optional[int]] = {}
        if not args.no_sizes:
            export_roots = [r.export_root for r in reports]
            size_by_export_root, tool = _bulk_export_sizes_once(root, export_roots)
            if tool:
                size_tool_used.add(tool)

        # Collection mode: lots of single-chat exports under a container directory
        # (common for the split output layout: ChatName/START__END).
        def _is_collection_mode() -> bool:
            if len(reports) < 5:
                return False
            if not all(r.kind in ("html_single_chat_export", "html_single_chat_export_converted") for r in reports):
                return False
            try:
                for r in reports:
                    rel = os.path.relpath(r.export_root, root)
                    if rel == ".":
                        return False
                    # Expect at least ChatName/START__END
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
            """
            Format a report line, but compute the "label" relative to `base` (not necessarily the
            scanned `root`). This is used for nested printing so children show up as "g" under
            their parent export root, instead of repeating the parent's name.
            """
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
            # Sorted by message count desc; unknowns last.
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

        # Grouping by "export root": if collection mode, show a single section for the
        # scanned root; otherwise, show one section per discovered export root.
        if collection_mode:
            print(_c(f"Export root: {root}", _Ansi.BOLD))
            print()
            # Group by the top-level folder under root (usually the chat name in split output).
            groups: Dict[str, List[ExportReport]] = {}
            for r in reports:
                rel = os.path.relpath(r.export_root, root)
                parts = [] if rel == "." else rel.split(os.sep)
                key = parts[0] if parts else os.path.basename(r.export_root)
                groups.setdefault(key, []).append(r)

            def _msg_sort_val(v: Optional[int]) -> int:
                # Unknowns last.
                return v if v is not None else -1

            def _group_sort_key(item: Tuple[str, List[ExportReport]]) -> tuple:
                # Order chat groups by the max message count of any subfolder (avoids double counting
                # when multiple exports exist for the same chat), then by name.
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
                # Within a chat, print the largest export first, then any additional subfolders indented
                # (so repeated chat names don't spam the output).
                def _within_key(r: ExportReport) -> tuple:
                    m = _msg_sort_val(r.messages_backed_up)
                    return (-m, _sub_label(r))

                rs_sorted = sorted(rs, key=_within_key)
                if not rs_sorted:
                    continue

                # Primary line (keeps the existing "ChatName SUB" format).
                print(_line_for_report(rs_sorted[0]))
                _print_chat_summaries(rs_sorted[0])

                # Additional exports for the same chat (indented; only show the subfolder label).
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
            # Nest results by filesystem hierarchy (export roots within other export roots).
            def _is_ancestor(a: str, b: str) -> bool:
                try:
                    ap = os.path.abspath(a)
                    bp = os.path.abspath(b)
                    if ap == bp:
                        return False
                    return os.path.commonpath([ap, bp]) == ap
                except Exception:
                    return False

            # Map export_root -> report and build a parent/children forest using the nearest ancestor export root.
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
                # Primary: messages desc (unknowns last). Secondary: relative path label for stability.
                rel = os.path.relpath(p, base)
                return (-_msg_sort_val(r.messages_backed_up), rel)

            def _print_node(p: str, *, depth: int, base: str) -> None:
                r = report_by_root[p]
                indent = "  " * depth
                print(_line_for_report_base(r, root=root, base=base, indent=indent))
                _print_chat_summaries(r, indent=indent)

                # Print nested exports (children) after the parent's own summaries.
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
    raise SystemExit(main())
