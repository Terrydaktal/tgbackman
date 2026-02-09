#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import sys
import urllib.parse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple


HTML_ATTR_URL_RE = re.compile(
    r'(?P<prefix>\b(?:src|href)\s*=\s*)(?P<q>["\'])(?P<url>[^"\']+)(?P=q)',
    re.IGNORECASE,
)
HTML_CSS_URL_RE = re.compile(
    r'url\(\s*(?P<q>["\']?)(?P<url>[^"\')]+)(?P=q)\s*\)',
    re.IGNORECASE,
)


def _is_ancestor_dir(parent: str, child: str) -> bool:
    parent = os.path.abspath(parent)
    child = os.path.abspath(child)
    try:
        common = os.path.commonpath([parent, child])
    except Exception:
        return False
    return common == parent


def _should_skip_url(url: str) -> bool:
    u = url.strip()
    if not u:
        return True
    if u.startswith("#"):
        return True
    lu = u.lower()
    if lu.startswith(("http://", "https://", "mailto:", "javascript:", "data:", "tg:", "tel:")):
        return True
    # Ignore any explicit scheme-like URLs (including file:).
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", u):
        return True
    return False


def _quote_url_path(rel_posix: str) -> str:
    # Quote reserved characters so file:// resolution hits the exact on-disk name.
    # Most important: literal '%' and '#'.
    return urllib.parse.quote(rel_posix, safe="/")


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

    # Try with and without fragment as well (Telegram sometimes emits an unescaped '#'
    # in the filename; browsers treat it as a fragment and break).
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


def _resolve_multi_chat_ref_to_abs(
    url: str,
    multi_root: str,
) -> Optional[Tuple[str, str]]:
    """
    If url contains chats/chat_XXX/... return (abs_path, rel_posix_from_multi_root).
    """
    if _should_skip_url(url):
        return None

    u0 = html.unescape(url).strip()
    u0 = u0.split("?", 1)[0].strip()
    if not u0:
        return None

    # We only handle multi-export layout refs.
    marker = "chats/chat_"
    idx = u0.find(marker)
    if idx < 0:
        return None

    rel0 = u0[idx:]
    # Consider fragment: for resolution, try both (some filenames include '#').
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


def _iter_chat_roots(single_root: str) -> Iterable[str]:
    """
    Yield per-chat export roots (directories that contain at least one messages*.html).
    """
    for dirpath, _dirnames, filenames in os.walk(single_root):
        if any(fn.startswith("messages") and fn.endswith(".html") for fn in filenames):
            yield dirpath


@dataclass
class FixStats:
    html_files_scanned: int = 0
    html_files_modified: int = 0
    urls_rewritten: int = 0
    media_files_copied: int = 0
    media_files_skipped_existing: int = 0


def fix_single_chat_export_in_place(
    chat_root: str,
    multi_root: str,
    *,
    dry_run: bool,
    overwrite: bool,
    verbose: bool,
    stats: FixStats,
) -> None:
    msg_files = sorted(
        os.path.join(chat_root, fn)
        for fn in os.listdir(chat_root)
        if fn.startswith("messages") and fn.endswith(".html")
    )
    if not msg_files:
        return

    media_root = os.path.join(chat_root, "media")

    # Cache: abs source -> (dest_abs_path, new_url)
    multi_copy_map: Dict[str, Tuple[str, str]] = {}

    def _ensure_multi_copied(abs_src: str, rel_posix: str) -> str:
        # Destination path keeps the multi-export relpath.
        dst_abs = os.path.join(media_root, rel_posix.replace("/", os.sep))
        dst_url = f"media/{_quote_url_path(rel_posix)}"
        if abs_src in multi_copy_map:
            return multi_copy_map[abs_src][1]
        multi_copy_map[abs_src] = (dst_abs, dst_url)
        return dst_url

    # First pass: discover multi-export refs and copy them.
    for p in msg_files:
        stats.html_files_scanned += 1
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "chats/chat_" not in line:
                        continue
                    for m in HTML_ATTR_URL_RE.finditer(line):
                        url = m.group("url")
                        resolved = _resolve_multi_chat_ref_to_abs(url, multi_root)
                        if not resolved:
                            continue
                        abs_src, rel_posix = resolved
                        _ensure_multi_copied(abs_src, rel_posix)
                    for m in HTML_CSS_URL_RE.finditer(line):
                        url = m.group("url")
                        resolved = _resolve_multi_chat_ref_to_abs(url, multi_root)
                        if not resolved:
                            continue
                        abs_src, rel_posix = resolved
                        _ensure_multi_copied(abs_src, rel_posix)
        except Exception:
            continue

    # Copy discovered media.
    if multi_copy_map:
        for abs_src, (dst_abs, _dst_url) in multi_copy_map.items():
            if os.path.isfile(dst_abs) and not overwrite:
                stats.media_files_skipped_existing += 1
                continue
            if dry_run:
                continue
            os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
            try:
                shutil.copy2(abs_src, dst_abs)
                stats.media_files_copied += 1
            except Exception:
                # best-effort
                continue

    # Second pass: rewrite HTML in-place.
    for p in msg_files:
        base_dir = os.path.dirname(p)
        tmp = p + ".tmp"
        changed = False

        def _rewrite_url(url: str) -> Optional[str]:
            # 1) Rewrite leftover multi-export refs to local media/...
            r = _resolve_multi_chat_ref_to_abs(url, multi_root)
            if r:
                abs_src, rel_posix = r
                return _ensure_multi_copied(abs_src, rel_posix)

            # 2) Fix broken local paths where Telegram emitted literal '%' or '#' in filenames.
            # Only attempt when the URL contains those characters (keeps diffs small).
            if ("%" in url) or ("#" in url):
                abs_local = _resolve_existing_under_root(url, base_dir, chat_root)
                if abs_local:
                    rel_local = os.path.relpath(abs_local, chat_root).replace(os.sep, "/")
                    return _quote_url_path(rel_local)

            return None

        def _sub_attr(m: re.Match) -> str:
            nonlocal changed
            prefix = m.group("prefix")
            q = m.group("q")
            url = m.group("url")
            nu = _rewrite_url(url)
            if not nu:
                return m.group(0)
            changed = True
            stats.urls_rewritten += 1
            return f"{prefix}{q}{nu}{q}"

        def _sub_css(m: re.Match) -> str:
            nonlocal changed
            q = m.group("q")
            url = m.group("url")
            nu = _rewrite_url(url)
            if not nu:
                return m.group(0)
            changed = True
            stats.urls_rewritten += 1
            return f"url({q}{nu}{q})"

        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as rf, open(
                tmp, "w", encoding="utf-8"
            ) as wf:
                for line in rf:
                    if ("href=" not in line) and ("src=" not in line) and ("url(" not in line) and ("chats/chat_" not in line):
                        wf.write(line)
                        continue
                    line2 = HTML_ATTR_URL_RE.sub(_sub_attr, line)
                    line2 = HTML_CSS_URL_RE.sub(_sub_css, line2)
                    wf.write(line2)

            if changed:
                stats.html_files_modified += 1
                if dry_run:
                    os.remove(tmp)
                else:
                    os.replace(tmp, p)
                    if verbose:
                        print(f"[fix] modified: {p}", file=sys.stderr)
            else:
                os.remove(tmp)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fix a split multi-chat HTML export in place by localizing leftover ../../chats/chat_XXX/... media refs."
    )
    ap.add_argument("single_root", help="Split output root (contains per-chat folders).")
    ap.add_argument(
        "--multi-root",
        required=True,
        help="Original multi-chat HTML export root (contains export_results.html and chats/).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Do not write/copy anything; just report.")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files under <chat>/media/ (default: skip existing).",
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose progress.")
    args = ap.parse_args()

    single_root = os.path.abspath(args.single_root)
    multi_root = os.path.abspath(args.multi_root)

    if not os.path.isdir(single_root):
        print(f"single_root is not a directory: {single_root}", file=sys.stderr)
        return 2
    if not os.path.isdir(multi_root):
        print(f"multi_root is not a directory: {multi_root}", file=sys.stderr)
        return 2
    if not os.path.isdir(os.path.join(multi_root, "chats")):
        print(f"multi_root does not look like a multi-export (missing chats/): {multi_root}", file=sys.stderr)
        return 2

    stats = FixStats()
    chat_roots = list(_iter_chat_roots(single_root))
    if not chat_roots:
        print("No per-chat exports found (no messages*.html).", file=sys.stderr)
        return 1

    for i, chat_root in enumerate(sorted(chat_roots), start=1):
        if args.verbose and (i == 1 or i % 50 == 0 or i == len(chat_roots)):
            print(f"[fix] chat {i}/{len(chat_roots)}: {chat_root}", file=sys.stderr)
        fix_single_chat_export_in_place(
            chat_root,
            multi_root,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            verbose=args.verbose,
            stats=stats,
        )

    print(f"Single root: {single_root}")
    print(f"Multi root:  {multi_root}")
    print(
        "Stats: "
        f"chat_roots={len(chat_roots)} "
        f"html_scanned={stats.html_files_scanned} "
        f"html_modified={stats.html_files_modified} "
        f"urls_rewritten={stats.urls_rewritten} "
        f"media_copied={stats.media_files_copied} "
        f"media_skipped_existing={stats.media_files_skipped_existing}"
    )
    if args.dry_run:
        print("(dry-run: no changes were written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

