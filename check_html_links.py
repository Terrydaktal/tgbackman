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
    if not os.path.commonpath([root_abs, abs_p]).startswith(root_abs):
        # This can happen with ../../.. links that escape root; still validate existence.
        note = (note + ",outside-root") if note else "outside-root"

    return abs_p, note


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


def scan_html_file_for_missing_refs(html_path: str, root: str) -> Tuple[int, int, int, List[MissingRef]]:
    """
    Returns: (total_refs, skipped_refs, ok_refs, missing_refs)
    """
    total = 0
    skipped = 0
    ok = 0
    missing: List[MissingRef] = []

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
                    resolved, note = _resolve_local_ref(url, html_path, root)
                    if resolved is None:
                        skipped += 1
                        continue
                    if os.path.exists(resolved):
                        ok += 1
                        continue
                    missing.append(MissingRef(html_path, ln, m.group(0).split("=", 1)[0].strip(), url, resolved, note))

                for m in ATTR_SRCSET_RE.finditer(line):
                    val = m.group("val")
                    for url in _iter_srcset_urls(val):
                        total += 1
                        resolved, note = _resolve_local_ref(url, html_path, root)
                        if resolved is None:
                            skipped += 1
                            continue
                        if os.path.exists(resolved):
                            ok += 1
                            continue
                        missing.append(MissingRef(html_path, ln, "srcset", url, resolved, note))

                for m in CSS_URL_RE.finditer(line):
                    url = m.group("url")
                    total += 1
                    resolved, note = _resolve_local_ref(url, html_path, root)
                    if resolved is None:
                        skipped += 1
                        continue
                    if os.path.exists(resolved):
                        ok += 1
                        continue
                    missing.append(MissingRef(html_path, ln, "url(...)", url, resolved, note))
    except Exception:
        # Treat unreadable files as having no refs; the caller can decide whether to care.
        return 0, 0, 0, []

    return total, skipped, ok, missing


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Root folder to scan (recursively)")
    ap.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    ap.add_argument("--max-missing", type=int, default=200, help="Limit missing refs printed (default: 200)")
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
    missing_refs: List[MissingRef] = []

    for p in html_files:
        t, s, ok, missing = scan_html_file_for_missing_refs(p, root)
        total_refs += t
        skipped_refs += s
        ok_refs += ok
        if missing:
            missing_refs.extend(missing)
            if args.fail_fast:
                break

    if args.json:
        out = {
            "root": root,
            "html_files_scanned": len(html_files),
            "total_refs": total_refs,
            "skipped_refs": skipped_refs,
            "ok_refs": ok_refs,
            "missing_refs": [
                {
                    "html_file": m.html_file,
                    "line_no": m.line_no,
                    "attr": m.attr,
                    "url": m.url,
                    "resolved_path": m.resolved_path,
                    "note": m.note,
                }
                for m in missing_refs[: args.max_missing]
            ],
            "missing_refs_total": len(missing_refs),
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 1 if missing_refs else 0

    print(_c(f"Root: {root}", _Ansi.BOLD))
    print(f"HTML files scanned: {len(html_files)}")
    print(f"Refs: total={total_refs} ok={ok_refs} skipped={skipped_refs} missing={len(missing_refs)}")

    if not missing_refs:
        print()
        print(_c("All local references resolved.", _Ansi.GREEN))
        return 0

    print()
    print(_c("Missing local references:", _Ansi.RED))
    lim = args.max_missing
    for i, m in enumerate(missing_refs[:lim], start=1):
        rel_html = os.path.relpath(m.html_file, root)
        note = f" [{m.note}]" if m.note else ""
        print(
            f"{i:>4}. {rel_html}:{m.line_no} {m.attr}=\"{m.url}\" -> {m.resolved_path}{note}"
        )
    if len(missing_refs) > lim:
        print(_c(f"... and {len(missing_refs) - lim} more", _Ansi.DIM))

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

