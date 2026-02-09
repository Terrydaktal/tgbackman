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
    u = urllib.parse.unquote(u)
    return u


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

    u = _normalize_url_to_fs_path(url)
    if not u:
        return None, None

    src_dir = os.path.dirname(src_html_path)
    if os.path.isabs(u):
        # Treat absolute paths as suspicious; still check if they exist.
        abs_p = os.path.abspath(os.path.normpath(u))
        note = "absolute"
    else:
        abs_p = os.path.abspath(os.path.normpath(os.path.join(src_dir, u)))
        note = None

    root_abs = os.path.abspath(root)
    try:
        within = os.path.commonpath([root_abs, abs_p]) == root_abs
    except Exception:
        within = False
    if not within:
        # This can happen with ../../.. links that escape root; still validate existence.
        note = (note + ",outside-root") if note else "outside-root"

    return abs_p, note

def _is_media_url(url: str) -> bool:
    """
    Heuristic: treat as "media" if the URL points into common Telegram media dirs or has a
    media-like extension.
    """
    if _should_skip_url(url):
        return False
    u = _normalize_url_to_fs_path(url)
    if not u:
        return False
    # Normalize separators for splitting.
    seg0 = u.replace("\\", "/").lstrip("/").split("/", 1)[0]
    if seg0 in MEDIA_DIRS:
        return True
    ext = os.path.splitext(u)[1].lower()
    if ext in MEDIA_EXTS:
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


def scan_html_file_for_bad_refs(
    html_path: str,
    root: str,
    *,
    scope: str,
    allow_outside_root: bool,
    max_keep: int,
) -> Tuple[int, int, int, int, List[MissingRef], int, List[MissingRef]]:
    """
    Returns:
      total_refs, skipped_refs, ok_refs,
      missing_total, missing_kept,
      outside_total, outside_kept
    """
    total = 0
    skipped = 0
    ok = 0
    missing_total = 0
    outside_total = 0
    missing_kept: List[MissingRef] = []
    outside_kept: List[MissingRef] = []

    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            for ln, line in enumerate(f, start=1):
                # Avoid heavy regex work on most lines.
                if (
                    "href" not in line
                    and "src" not in line
                    and "poster" not in line
                    and "srcset" not in line
                    and "url(" not in line
                ):
                    continue

                for m in ATTR_URL_RE.finditer(line):
                    url = m.group("url")
                    total += 1
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
    except Exception:
        # Treat unreadable files as having no refs; the caller can decide whether to care.
        return 0, 0, 0, 0, [], 0, []

    return total, skipped, ok, missing_total, missing_kept, outside_total, outside_kept


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
        "--fail-fast",
        action="store_true",
        help="Stop after the first missing ref (useful for quick checks).",
    )
    args = ap.parse_args(argv)

    root = os.path.abspath(args.path)
    if not os.path.exists(root):
        print(f"Path does not exist: {root}", file=sys.stderr)
        return 2

    html_files = list(_iter_html_files(root))
    html_files.sort()

    total_refs = 0
    skipped_refs = 0
    ok_refs = 0
    missing_total = 0
    outside_total = 0
    missing_kept: List[MissingRef] = []
    outside_kept: List[MissingRef] = []

    for p in html_files:
        t, s, ok, m_total, m_kept, o_total, o_kept = scan_html_file_for_bad_refs(
            p,
            root,
            scope=args.scope,
            allow_outside_root=args.allow_outside_root,
            max_keep=args.max_missing,
        )
        total_refs += t
        skipped_refs += s
        ok_refs += ok
        missing_total += m_total
        outside_total += o_total
        # Keep a bounded sample list across all files.
        for m in m_kept:
            if len(missing_kept) >= args.max_missing:
                break
            missing_kept.append(m)
        for m in o_kept:
            if len(outside_kept) >= args.max_missing:
                break
            outside_kept.append(m)

        if (m_total or o_total) and args.fail_fast:
            break

    if args.json:
        out = {
            "root": root,
            "html_files_scanned": len(html_files),
            "total_refs": total_refs,
            "skipped_refs": skipped_refs,
            "ok_refs": ok_refs,
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
        return 1 if (missing_total or outside_total) else 0

    print(_c(f"Root: {root}", _Ansi.BOLD))
    print(f"HTML files scanned: {len(html_files)}")
    scope_note = f"scope={args.scope}"
    print(
        f"Refs: total={total_refs} ok={ok_refs} skipped={skipped_refs} "
        f"outside_root={outside_total} missing={missing_total} ({scope_note})"
    )

    if not missing_total and not outside_total:
        print()
        print(_c("All local references resolved.", _Ansi.GREEN))
        return 0

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

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
