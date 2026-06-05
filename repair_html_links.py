#!/usr/bin/env python3
"""
repair_html_links.py

Scan all .html files under a Telegram backup folder and repair broken local links
containing unescaped hash ('#') characters in their filenames by URL-encoding them
in-place.

Usage:
  python3 repair_html_links.py /path/to/backup_folder [--dry-run]
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
import urllib.parse
from typing import Iterable, Optional, Tuple

LINK_ATTR_URL_RE = re.compile(
    r'(?P<prefix>\b(?:src|href|poster)\s*=\s*)(?P<q>["\'])(?P<url>[^"\']+)(?P=q)',
    re.IGNORECASE,
)
LINK_ATTR_SRCSET_RE = re.compile(
    r'(?P<prefix>\bsrcset\s*=\s*)(?P<q>["\'])(?P<val>[^"\']+)(?P=q)',
    re.IGNORECASE,
)
HTML_CSS_URL_RE = re.compile(
    r'url\(\s*(?P<q>["\']?)(?P<url>[^"\')]+)(?P=q)\s*\)',
    re.IGNORECASE,
)

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

def _iter_all_html_files(root: str) -> Iterable[str]:
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
    if lu.startswith(("http://", "https://", "mailto:", "javascript:", "data:", "tg:", "tel:")):
        return True
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", u):
        return True
    return False

def repair_local_html_links_in_place(root: str, *, apply_changes: bool) -> Tuple[int, int, int]:
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
            if not raw or "#" not in raw or "%23" in raw.lower():
                return None
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

def main() -> int:
    ap = argparse.ArgumentParser(description="Repair unescaped '#' in Telegram HTML backup links")
    ap.add_argument("path", help="Telegram export folder to scan recursively")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many files/links would change without rewriting files",
    )
    args = ap.parse_args()

    root = os.path.abspath(args.path)
    if not os.path.exists(root):
        print(f"Path does not exist: {root}", file=sys.stderr)
        return 2

    scanned, changed, rewritten = repair_local_html_links_in_place(
        root,
        apply_changes=not args.dry_run,
    )
    mode = "dry-run" if args.dry_run else "applied"
    print(f"Repair mode: {mode}")
    print(f"HTML files scanned: {scanned}")
    print(f"HTML files changed: {changed}")
    print(f"Links rewritten: {rewritten}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
