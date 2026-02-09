#!/usr/bin/env python3
"""
check_html_links.py

Scan all .html files under a backup folder and verify that local (on-disk) links resolve.

We check:
- href="..."
- src="..."
- poster="..."
- srcset="..." (best-effort; parses URLs in the list)
- CSS url(...) inside HTML (inline styles / <style> blocks)

We skip:
- External URLs (http/https/mailto/data/javascript/tg/tel, etc.)
- Anchors (#...)

Exit codes:
- 0: no missing local targets
- 1: missing local targets found
- 2: invalid usage / path not found
"""

from __future__ import annotations

import argparse
import html as _html
import json
import os
import re
import sys
import urllib.parse
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class _Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"


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


ATTR_URL_RE = re.compile(
    r'(?P<prefix>\b(?:src|href|poster)\s*=\s*)(?P<q>["\'])(?P<url>[^"\']+)(?P=q)',
    re.IGNORECASE,
)
ATTR_SRCSET_RE = re.compile(
    r'(?P<prefix>\bsrcset\s*=\s*)(?P<q>["\'])(?P<val>[^"\']+)(?P=q)',
    re.IGNORECASE,
)
CSS_URL_RE = re.compile(
    r'url\(\s*(?P<q>["\']?)(?P<url>[^"\')]+)(?P=q)\s*\)',
    re.IGNORECASE,
)
MULTI_EXPORT_REF_RE = re.compile(
    r'\b(?:href|src)\s*=\s*(?P<q>["\'])(?P<url>(?:\.\./)+chats/chat_\d+/[^"\']+)(?P=q)',
    re.IGNORECASE,
)


_SKIP_SCHEMES = (
    "http://",
    "https://",
    "mailto:",
    "javascript:",
    "data:",
    "tg:",
    "tel:",
)

MEDIA_DIRS = {
    # Common Telegram export media dirs (varies by export type / version).
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

# Media dirs that represent "chat content" when present directly under a chat export root.
# (We intentionally exclude css/js/images because those are rendering assets in Telegram HTML exports.)
CHAT_LOCAL_MEDIA_DIRS = {
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

MEDIA_EXTS = {
    # Images
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".avif",
    # Video
    ".mp4",
    ".m4v",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".3gp",
    # Audio
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wav",
    ".flac",
    # Docs / archives (often shared as files)
    ".pdf",
    ".txt",
    ".zip",
    ".rar",
    ".7z",
}


def _iter_html_files(root: str) -> Iterator[str]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".html"):
                yield os.path.join(dirpath, fn)

def _iter_chat_roots(root: str) -> Iterator[str]:
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


def _should_skip_url(url: str) -> bool:
    u = url.strip()
    if not u:
        return True
    if u.startswith("#"):
        return True
    lu = u.lower()
    if lu.startswith(_SKIP_SCHEMES):
        return True
    # Ignore any other scheme-ish value (e.g. file:, chrome:, etc.)
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", u):
        return True
    return False


def _normalize_url_to_fs_path(url: str) -> str:
    u = _html.unescape(url).strip()
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


def _resolve_local_ref(
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

    return chosen_abs, chosen_note

def _is_media_url(url: str) -> bool:
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
    # Normalize separators for splitting, ignoring leading ../ and ./ segments.
    u2 = _strip_leading_dot_segments(u)
    parts = [p for p in u2.replace("\\", "/").lstrip("/").split("/") if p]
    if any(p in MEDIA_DIRS for p in parts):
        return True
    return False


def _iter_srcset_urls(val: str) -> Iterator[str]:
    # srcset format: "url1 1x, url2 2x" or "url 640w, url 1280w"
    # We take the first token of each comma-separated entry.
    for part in val.split(","):
        part = part.strip()
        if not part:
            continue
        url = part.split()[0].strip()
        if url:
            yield url


@dataclass(frozen=True)
class MissingRef:
    html_file: str
    line_no: int
    attr: str  # src/href/poster/srcset/url(...)
    url: str
    resolved_path: str
    note: Optional[str] = None


@dataclass(frozen=True)
class SchemeRef:
    html_file: str
    line_no: int
    attr: str  # src/href/poster/srcset
    url: str


def scan_html_file_for_bad_refs(
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
) -> Tuple[int, int, int, int, List[MissingRef], int, List[MissingRef], int, List[MissingRef], int, List[SchemeRef]]:
    """
    Returns:
      total_refs, skipped_refs, ok_refs,
      missing_total, missing_kept,
      outside_total, outside_kept,
      split_leftover_total, split_leftover_kept,
      scheme_total, scheme_kept
    """
    total = 0
    skipped = 0
    ok = 0
    missing_total = 0
    outside_total = 0
    split_leftover_total = 0
    missing_kept: List[MissingRef] = []
    outside_kept: List[MissingRef] = []
    split_leftover_kept: List[MissingRef] = []
    scheme_total = 0
    scheme_kept: List[SchemeRef] = []
    in_style_block = False

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
                scheme_kept.append(SchemeRef(html_path, ln, attr, u))

    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            for ln, line in enumerate(f, start=1):
                line_l = line.lower()
                # Track <style> blocks so we only treat url(...) in real CSS contexts.
                if "<style" in line_l:
                    in_style_block = True
                if "</style" in line_l:
                    # Note: if both <style and </style are on the same line, we'll still
                    # process the line as CSS context.
                    end_style_after = True
                else:
                    end_style_after = False

                # Avoid heavy regex work on most lines.
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

                # If we're checking a split output (not a multi-export root), flag any remaining
                # multi-export style references in message pages.
                if (
                    check_split_leftovers
                    and not is_multi_export_root
                    and os.path.basename(html_path).startswith("messages")
                    and os.path.basename(html_path).lower().endswith(".html")
                    and "chats/chat_" in line_l
                ):
                    for m in MULTI_EXPORT_REF_RE.finditer(line):
                        url = m.group("url")
                        # Optional media-only filtering: these are always media-like, but keep consistent.
                        if scope == "media" and not _is_media_url(url):
                            continue
                        split_leftover_total += 1
                        if len(split_leftover_kept) < max_keep:
                            split_leftover_kept.append(
                                MissingRef(
                                    html_path,
                                    ln,
                                    "multi_export_ref",
                                    url,
                                    "(should be localized into this chat folder)",
                                    "multi-export-leftover",
                                )
                            )

                for m in ATTR_URL_RE.finditer(line):
                    url = m.group("url")
                    total += 1
                    _maybe_keep_scheme(ln, m.group(0).split("=", 1)[0].strip(), url)
                    if scope == "media" and not _is_media_url(url):
                        skipped += 1
                        continue
                    resolved, note = _resolve_local_ref(url, html_path, root)
                    if resolved is None:
                        skipped += 1
                        continue
                    is_outside = bool(note and "outside-root" in note)
                    exists = os.path.exists(resolved)
                    if is_outside and not allow_outside_root:
                        outside_total += 1
                        if len(outside_kept) < max_keep:
                            outside_kept.append(
                                MissingRef(
                                    html_path,
                                    ln,
                                    m.group(0).split("=", 1)[0].strip(),
                                    url,
                                    resolved,
                                    note,
                                )
                            )
                        continue
                    if exists:
                        ok += 1
                        continue
                    missing_total += 1
                    if len(missing_kept) < max_keep:
                        missing_kept.append(
                            MissingRef(
                                html_path,
                                ln,
                                m.group(0).split("=", 1)[0].strip(),
                                url,
                                resolved,
                                note,
                            )
                        )

                for m in ATTR_SRCSET_RE.finditer(line):
                    val = m.group("val")
                    for url in _iter_srcset_urls(val):
                        total += 1
                        _maybe_keep_scheme(ln, "srcset", url)
                        if scope == "media" and not _is_media_url(url):
                            skipped += 1
                            continue
                        resolved, note = _resolve_local_ref(url, html_path, root)
                        if resolved is None:
                            skipped += 1
                            continue
                        is_outside = bool(note and "outside-root" in note)
                        exists = os.path.exists(resolved)
                        if is_outside and not allow_outside_root:
                            outside_total += 1
                            if len(outside_kept) < max_keep:
                                outside_kept.append(MissingRef(html_path, ln, "srcset", url, resolved, note))
                            continue
                        if exists:
                            ok += 1
                            continue
                        missing_total += 1
                        if len(missing_kept) < max_keep:
                            missing_kept.append(MissingRef(html_path, ln, "srcset", url, resolved, note))

                for m in CSS_URL_RE.finditer(line):
                    url = m.group("url")
                    total += 1
                    # Only consider url(...) inside actual CSS contexts (style="..." or <style> blocks).
                    if not in_style_block and 'style="' not in line_l and "style='" not in line_l:
                        skipped += 1
                        continue
                    if scope == "media" and not _is_media_url(url):
                        skipped += 1
                        continue
                    resolved, note = _resolve_local_ref(url, html_path, root)
                    if resolved is None:
                        skipped += 1
                        continue
                    is_outside = bool(note and "outside-root" in note)
                    exists = os.path.exists(resolved)
                    if is_outside and not allow_outside_root:
                        outside_total += 1
                        if len(outside_kept) < max_keep:
                            outside_kept.append(MissingRef(html_path, ln, "url(...)", url, resolved, note))
                        continue
                    if exists:
                        ok += 1
                        continue
                    missing_total += 1
                    if len(missing_kept) < max_keep:
                        missing_kept.append(MissingRef(html_path, ln, "url(...)", url, resolved, note))

                if end_style_after:
                    in_style_block = False
    except Exception:
        # Treat unreadable files as having no refs; the caller can decide whether to care.
        return 0, 0, 0, 0, [], 0, [], 0, [], 0, []

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
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Root folder to scan (recursively)")
    ap.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    ap.add_argument("--max-missing", type=int, default=200, help="Limit missing refs printed (default: 200)")
    ap.add_argument(
        "--scope",
        choices=("all", "media"),
        default="all",
        help="Which refs to validate: all local refs, or only media-like refs (default: all).",
    )
    ap.add_argument(
        "--allow-outside-root",
        action="store_true",
        help="Treat refs that resolve outside the provided root as OK if they exist on disk (default: fail them).",
    )
    ap.add_argument(
        "--no-split-leftovers-check",
        action="store_true",
        help="Disable detection of leftover multi-export links (../../chats/chat_XXX/...) in messages*.html.",
    )
    ap.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first missing ref (useful for quick checks).",
    )
    ap.add_argument(
        "--list-schemes",
        default="",
        help='Comma-separated scheme names to list when they appear in HTML attributes (href/src/poster/srcset), e.g. "file,tg".',
    )
    ap.add_argument(
        "--max-schemes",
        type=int,
        default=200,
        help="Limit scheme refs printed (default: 200).",
    )
    ap.add_argument(
        "--count-media-files",
        action="store_true",
        help="Also count on-disk media files: shared under each chat's media/ vs chat-local media dirs.",
    )
    args = ap.parse_args(argv)

    root = os.path.abspath(args.path)
    if not os.path.exists(root):
        print(f"Path does not exist: {root}", file=sys.stderr)
        return 2

    html_files = list(_iter_html_files(root))
    html_files.sort()

    # Heuristic to detect scanning an original Telegram multi-chat export root.
    is_multi_export_root = os.path.isfile(os.path.join(root, "export_results.html")) or os.path.isdir(
        os.path.join(root, "chats")
    )

    total_refs = 0
    skipped_refs = 0
    ok_refs = 0
    missing_total = 0
    outside_total = 0
    split_leftover_total = 0
    scheme_total = 0
    missing_kept: List[MissingRef] = []
    outside_kept: List[MissingRef] = []
    split_leftover_kept: List[MissingRef] = []
    scheme_kept: List[SchemeRef] = []

    scheme_prefixes: Optional[Tuple[str, ...]]
    scheme_names = [s.strip().lower() for s in args.list_schemes.split(",") if s.strip()]
    if scheme_names:
        scheme_prefixes = tuple(f"{s}:" for s in scheme_names)
    else:
        scheme_prefixes = None

    for p in html_files:
        t, s, ok, m_total, m_kept, o_total, o_kept, sl_total, sl_kept, sc_total, sc_kept = scan_html_file_for_bad_refs(
            p,
            root,
            scope=args.scope,
            allow_outside_root=args.allow_outside_root,
            check_split_leftovers=(not args.no_split_leftovers_check),
            is_multi_export_root=is_multi_export_root,
            max_keep=args.max_missing,
            scheme_prefixes=scheme_prefixes,
            max_keep_schemes=args.max_schemes,
        )
        total_refs += t
        skipped_refs += s
        ok_refs += ok
        missing_total += m_total
        outside_total += o_total
        split_leftover_total += sl_total
        scheme_total += sc_total
        # Keep a bounded sample list across all files.
        for m in m_kept:
            if len(missing_kept) >= args.max_missing:
                break
            missing_kept.append(m)
        for m in o_kept:
            if len(outside_kept) >= args.max_missing:
                break
            outside_kept.append(m)
        for m in sl_kept:
            if len(split_leftover_kept) >= args.max_missing:
                break
            split_leftover_kept.append(m)
        for m in sc_kept:
            if len(scheme_kept) >= args.max_schemes:
                break
            scheme_kept.append(m)

        if (m_total or o_total or sl_total) and args.fail_fast:
            break

    shared_media_files = 0
    chat_local_media_files = 0
    if args.count_media_files:
        # Best-effort counts; this is independent of link resolution.
        seen_chat_roots: Set[str] = set()
        for cr in _iter_chat_roots(root):
            if cr in seen_chat_roots:
                continue
            seen_chat_roots.add(cr)

            shared_p = os.path.join(cr, "media")
            if os.path.isdir(shared_p):
                shared_media_files += _count_files_recursive(shared_p)

            for d in CHAT_LOCAL_MEDIA_DIRS:
                lp = os.path.join(cr, d)
                if os.path.isdir(lp):
                    chat_local_media_files += _count_files_recursive(lp)

    if args.json:
        out = {
            "root": root,
            "is_multi_export_root": is_multi_export_root,
            "html_files_scanned": len(html_files),
            "total_refs": total_refs,
            "skipped_refs": skipped_refs,
            "ok_refs": ok_refs,
            "scheme_refs_total": scheme_total if scheme_prefixes else None,
            "scheme_refs": [
                {
                    "html_file": m.html_file,
                    "line_no": m.line_no,
                    "attr": m.attr,
                    "url": m.url,
                }
                for m in scheme_kept
            ]
            if scheme_prefixes
            else [],
            "shared_media_files": shared_media_files if args.count_media_files else None,
            "chat_local_media_files": chat_local_media_files if args.count_media_files else None,
            "split_leftover_refs": [
                {
                    "html_file": m.html_file,
                    "line_no": m.line_no,
                    "attr": m.attr,
                    "url": m.url,
                    "note": m.note,
                }
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
    scope_note = f"scope={args.scope}"
    print(
        f"Refs: total={total_refs} ok={ok_refs} skipped={skipped_refs} "
        f"split_leftover={split_leftover_total} outside_root={outside_total} missing={missing_total} ({scope_note})"
    )
    if args.count_media_files:
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
        lim = args.max_missing
        for i, m in enumerate(split_leftover_kept[:lim], start=1):
            rel_html = os.path.relpath(m.html_file, root)
            print(f"{i:>4}. {rel_html}:{m.line_no} {m.url}")
        if split_leftover_total > len(split_leftover_kept):
            print(_c(f"... and {split_leftover_total - len(split_leftover_kept)} more", _Ansi.DIM))
        print()

    if outside_total:
        print(_c("References that resolve outside the provided root:", _Ansi.YELLOW))
        lim = args.max_missing
        for i, m in enumerate(outside_kept[:lim], start=1):
            rel_html = os.path.relpath(m.html_file, root)
            note = f" [{m.note}]" if m.note else ""
            print(
                f"{i:>4}. {rel_html}:{m.line_no} {m.attr}=\"{m.url}\" -> {m.resolved_path}{note}"
            )
        if outside_total > len(outside_kept):
            print(_c(f"... and {outside_total - len(outside_kept)} more", _Ansi.DIM))
        print()

    if missing_total:
        print(_c("Missing local references:", _Ansi.RED))
        lim = args.max_missing
        for i, m in enumerate(missing_kept[:lim], start=1):
            rel_html = os.path.relpath(m.html_file, root)
            note = f" [{m.note}]" if m.note else ""
            print(
                f"{i:>4}. {rel_html}:{m.line_no} {m.attr}=\"{m.url}\" -> {m.resolved_path}{note}"
            )
        if missing_total > len(missing_kept):
            print(_c(f"... and {missing_total - len(missing_kept)} more", _Ansi.DIM))

    if scheme_prefixes and scheme_total:
        print()
        print(_c("Scheme refs found in HTML attributes:", _Ansi.CYAN))
        lim = args.max_schemes
        for i, m in enumerate(scheme_kept[:lim], start=1):
            rel_html = os.path.relpath(m.html_file, root)
            print(f"{i:>4}. {rel_html}:{m.line_no} {m.attr}=\"{m.url}\"")
        if scheme_total > len(scheme_kept):
            print(_c(f"... and {scheme_total - len(scheme_kept)} more", _Ansi.DIM))

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
