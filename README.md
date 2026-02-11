# tgbackman

`backman.py` is a single-file tool for Telegram backup workflows:

- discover and summarize backups under a root folder
- split official HTML multi-chat exports into per-chat self-contained exports
- validate local HTML links (`href/src/poster/srcset/url(...)`)
- repair broken local links that contain unescaped `#` in filenames

## Supported Backup Types

`backman.py` scans recursively and detects:

- Official Telegram HTML multi-chat exports
  - has `export_results.html` and `chats/chat_XXX/...`
- Official Telegram HTML single-chat exports
  - has `messages.html` plus `css/` and `js/`
- Official Telegram JSON exports
  - has `result.json` or `results.json`
- Unofficial legacy SQLite backups
  - `database.sqlite` with `messages/chats/users` tables

## Requirements

- Python 3 (stdlib only)
- optional: `dust` (faster directory sizing)
- optional: `jq` (pretty-print JSON output)

## Basic Usage

Scan and print human-readable summary:

```bash
python3 backman.py "/path/to/Telegram Backup"
```

Machine-readable output:

```bash
python3 backman.py --json "/path/to/Telegram Backup" | jq .
```

Performance knobs:

- `--no-inspect`: skip deep message scanning (counts/ranges)
- `--no-sizes`: skip size-on-disk calculations

## Action Modes

Only one action flag may be used at a time:

- `--split-multi-html`
- `--check-links`
- `--repair-html-links`

## Split HTML Multi-Chat Export

Convert one official HTML multi export into per-chat standalone folders.

```bash
python3 backman.py --split-multi-html "/path/to/Telegram Backup"
```

Options:

- `--split-out /path/to/output`
- `--split-chat chat_001` (repeatable)
- `--split-dry-run`

Default output root (if omitted):

- `<parent>/<basename>_single_chats`

Output structure:

```text
<out_root>/
  <Chat Name>/
    <START__END>/
      messages.html
      messages2.html
      ...
      photos/ files/ video_files/ ...
      media/
      css/ js/ images/
      .backman_export_meta.json
```

`START__END` is UTC and filesystem-safe:

- `YYYY-MM-DDTHH-MM-SSZ__YYYY-MM-DDTHH-MM-SSZ`

## Check HTML Links

Validate local references in all HTML files under a root.

```bash
python3 backman.py --check-links "/path/to/backup"
```

Key options:

- `--check-scope {all,media}`
- `--check-max-missing 200`
- `--check-fail-fast`
- `--check-allow-outside-root`
- `--check-no-split-leftovers-check`
- `--check-list-schemes "tg,file,http"`
- `--check-max-schemes 200`
- `--check-count-media-files`
- `--json`

Current text summary includes:

- `HTML files scanned: ...`
- `Ref classes (exclusive): total=... missing=... non_files=... cross_chat=... media_shared=... chat_local=... html_stuff=...`

When missing links are printed, each item is prefixed with a resolved chat name when available:

- official multi-chat: from `chats/chat_XXX/messages*.html`
- unofficial dialogs: from associated SQLite user/chat records

## Repair Broken Local HTML Links

Fixes only local links with unescaped `#` in filename segments by URL-encoding them.

```bash
python3 backman.py --repair-html-links "/path/to/backup"
```

Dry run:

```bash
python3 backman.py --repair-html-links --repair-dry-run "/path/to/backup"
```

It reports:

- HTML files scanned
- HTML files changed
- links rewritten

## Unofficial SQLite Deduping

If both canonical and duplicate unofficial DBs are present, you can suppress duplicate summaries:

```bash
python3 backman.py --dedupe-unofficial-sqlite "/path/to/Telegram Backup"
```

Behavior:

- prefers `database.sqlite` under `.telegram_backup/...`
- suppresses sibling/parent duplicate DB views under same tree

## Converted vs Original Single-Chat HTML

- Original Telegram single export: `html_single_chat_export`
- Split output from `--split-multi-html`: `html_single_chat_export_converted`

Detection is based on marker metadata written into converted chat range folders.

## Exit Codes

- `0`: success / no unresolved link failures
- `1`: link-check found missing/outside/split-leftover issues
- `2`: bad CLI usage or invalid input path
- `3`: split action failed

## Repository

Remote:

- `https://github.com/Terrydaktal/tgbackman.git`
