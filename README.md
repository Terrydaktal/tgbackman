# tgbackman

Tools for inspecting, splitting, and validating Telegram exports.

`backman.py` recursively scans a folder for Telegram exports and prints (or emits JSON for) the discovered backups, including chat/message counts and message date ranges. It also includes action modes to convert multi-chat HTML exports into per-chat exports and to validate that exports are self-contained.

## Requirements

- Python 3 (no pip dependencies; uses the standard library)
- Optional (nice-to-have): `jq` (for viewing JSON output)

## Supported Backup Types

- Official Telegram exports (Telegram Desktop):
  - HTML multi-chat export: has `export_results.html` and `chats/chat_XXX/...`
  - HTML single-chat export: has `messages.html` (and sometimes `messages2.html`, etc.)
  - JSON export: has `result.json` / `results.json`
- Unofficial legacy backup (SQLite) layouts:
  - Detects `database.sqlite` under `.telegram_backup/...` and summarizes it.

Detection is heuristic; if you have an export that “should” be detected but isn’t, open an issue and include a minimal directory listing.

## Quick Start (Scan + Report)

Scan a folder and print a human-readable report:

```bash
python3 backman.py "/path/to/Telegram Backup"
```

Machine-readable JSON:

```bash
python3 backman.py --json "/path/to/Telegram Backup" | jq .
```

Speed knobs:

- `--no-inspect`: skips scanning message HTML/JSON bodies for ranges/counts (much faster)
- `--no-sizes`: skips computing on-disk sizes (faster on slow disks)

## Action Modes (One at a Time)

`backman.py` has a couple of action modes. Only one action flag may be used per invocation:

- `--split-multi-html` (convert multi-chat HTML to per-chat exports)
- `--check-links` (validate local references resolve on disk)

### Convert Multi-Chat HTML Export to Per-Chat Exports

This takes a multi-chat Telegram HTML export and produces per-chat exports where each chat has its own localized “shared media” copies.

```bash
python3 backman.py --split-multi-html "/path/to/Telegram Backup"
```

Optional flags:

- `--split-out /path/to/output_root`
- `--split-chat chat_001` (repeatable; only split certain chat folders)
- `--split-dry-run` (prints planned output root and exits)

#### Output Layout

The split output is a “collection” root containing per-chat folders. Each per-chat folder contains one or more range subfolders:

```
<out_root>/
  <Chat Name>/
    <START__END>/
      messages.html
      messages2.html
      ...
      photos/ files/ video_files/ ...
      media/                       # copies of shared media (from the original multi-export)
      css/ js/ images/             # Telegram HTML rendering assets
      .backman_export_meta.json    # marks the export as converted (see below)
```

`START__END` uses UTC timestamps:

`YYYY-MM-DDTHH-MM-SSZ__YYYY-MM-DDTHH-MM-SSZ`

### Validate Local Links Inside HTML Files

Checks that local `href=...`, `src=...`, `poster=...`, `srcset=...`, and CSS `url(...)` references resolve on disk.

```bash
python3 backman.py --check-links "/path/to/backup"
```

Common useful flags:

- `--check-scope media` (only check likely-media references)
- `--check-max-missing 200`
- `--check-fail-fast` (stop on first failing file)
- `--check-count-media-files` (counts files under `media/` vs chat-local media folders)
- `--json` (emit JSON summary)

This also detects “leftover multi-export links” like `../../chats/chat_XXX/...` inside `messages*.html`, which should not exist in properly localized split outputs.

## Converted vs Original Single-Chat Exports

Telegram’s “Export chat history” single-chat exports are labeled:

- `html_single_chat_export`

Per-chat exports created by `--split-multi-html` are labeled:

- `html_single_chat_export_converted`

The only signal used for “converted” is the presence of:

`<export_root>/.backman_export_meta.json`

## Safety / Non-Destructive Behavior

- The scan/report mode never modifies files.
- `--split-multi-html` copies data to a new output root (does not delete sources).

## Performance Notes

- `--no-inspect` can be dramatically faster on very large exports.
- On very large split roots, link-checking can take time because it scans all `.html` files.

## Troubleshooting

- If `--check-links` reports “Leftover multi-export links found” in a split output, re-run the split using the latest `--split-multi-html` logic (or fix the output manually).

## License

No license file has been added yet. If you want an OSS license (MIT/Apache-2.0/etc.), say which and I’ll add it.
