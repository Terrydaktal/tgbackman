# Architecture

The repository has two supported front ends and one canonical Python service
layer:

```text
src/tgbackup/
  cli.py                 console entry point (`tgbackman-backup`)
  config.py              paths, credentials, media/size parsing, permissions
  media_reorganize.py    dry-run/resumable legacy-to-per-chat reflink migration
  models.py              exporter value objects
  errors.py              public exporter exception types
  progress.py            journal-friendly progress and process locking
  db/
    connection.py        SQLite validation and connection policy
    schema.py             canonical tables, migrations, FTS, and statistics refresh
    archive.py            rich message conversion/upsert and provenance writes
    sources.py            immutable source registration and media integrity
    repository.py        target, active-chat, and blacklist queries
  legacy_tree/           filesystem-only legacy HTML/JSON/media operations
    split_html.py
    link_repair.py
    check_links.py
    media_relocator.py
    backfill_metadata.py
    inspect_tree.py       legacy export inspection (including embedded unofficial SQLite)
  database/               commands that open/create/reconcile SQLite
    importer.py           legacy files -> canonical database
    rebuild.py
    overlap_cli.py        explicit --db adapter for overlap analysis
    overlap_report.py
    range_repair.py
  telegram/client.py     Telethon authentication and peer resolution
  backup/
    media.py             primary-media selection, download, size/hash checks
    staging.py           durable failed-run staging and resume verification
    targets.py           stable peer identity and output-directory policy
    target_mapping.py    Telegram dialog mapping and migrated-peer links
    records.py           pure message-record/range helpers
  exporter.py            compatibility facade and transactional orchestration
```

The canonical backup command is the installed `tgbackman-backup` entry point
(or `python -m tgbackup`). New code should import `tgbackup` or use the
prefixed package commands; the implementation no longer depends on root-level
launcher files.

The Rust applications are split into crate-local modules. `tgbackman` keeps
only the egui frame loop and process entry point in `main.rs`; `app.rs` owns
GUI state/background workers, `database.rs` owns inventory/statistics/target
mapping, `matching.rs` owns overlap normalisation/comparison, `ui.rs` owns
shared rendering primitives, `viewer.rs` owns the integrated Telegram-style
conversation/search presentation, `model.rs` owns value objects, `cache.rs` owns cache
freshness/path policy, and `inventory.rs` owns union-find. `tgsearch` keeps its
CLI/query loop in `main.rs`, with `models.rs` for result rows and `render.rs`
for formatting, highlighting, and sanitisation.

The GUI does not duplicate exporter logic. It reads the canonical SQLite
database; a future background-run button should launch the `tgbackman-backup`
console process with an explicit database/config/session path and consume its
stdout as progress events.

The GUI persists a versioned inventory cache beside the database and validates
it against both the database and WAL modification times. A stale inventory is
rebuilt from denormalized chat statistics; only chats explicitly invalidated by
an importer are rescanned. Structural clustering is reused across ordinary API
increments and is explicitly invalidated by legacy imports.

The chat viewer is read-only. Conversation pages are limited to 400 rows and
move using the composite `(chat_id, timestamp_unix, message_id)` index. Sender
identity aliases are resolved once when a chat opens and reused for subsequent
page and exact-message navigation. Global and in-chat searches execute on
background workers against `messages_fts` and report the exact total. Global
results use a virtual list: a compact ordered row-ID index is retained, while
only visible 250-row database pages and visible egui rows are materialized;
nearby pages are fetched automatically during scrolling. Every match remains directly reachable through the scrollbar
without a “load more” control, even for result sets containing hundreds of
thousands of rows. In-chat search loads one compact result set with a single
count and sort. Selecting a match loads a bounded page around that message and
highlights the matching terms. Outgoing presentation uses stable Telegram sender IDs
where available and infers legacy sender aliases from linked overlapping
backups; reply previews may resolve their parent from elsewhere in the local
database without embedding or duplicating that parent.

The legacy/database boundary is intentional: a filesystem repair cannot
silently mutate the canonical message store. Database reconciliation is
explicit. The `database/` commands, direct API exporter, and the explicit
`media_reorganize.py` maintenance bridge are the only operations that update
the canonical message store.

## Lossless conversation metadata

The API exporter treats the Telegram TL object as the archival boundary. Each
message record contains an exact binary TL payload plus a complete JSON form;
the canonical row keeps both, with the binary hash, Telegram layer, and
Telethon version. Secondary requests capture every visible reactor and public
poll voter page. Content-addressed entity snapshots deduplicate message senders
and the basic/full chat responses, while reference tables connect those
snapshots to messages, chats, and immutable API-run manifests.

`backup/staging.py` verifies this metadata ledger before staged rows can be
reused or merged. `db/archive.py` stores it, and `database/importer.py` verifies
the immutable run records against the current presentation-facing rows. A
versioned staging key prevents an older partial record from being mistaken for
a lossless one. These dependencies remain acyclic: record serialization and
staging do not import the exporter or importer.

The contract is a current API-visible conversation snapshot. Values Telegram
withholds are represented by explicit `not_exposed` states. Deleted/expired
pre-capture content, historical edits and membership, secret chats, and
account/client state are not recoverable from Telegram history. Incremental
runs preserve this contract for fetched messages; a full rescan is required to
upgrade legacy rows and refresh mutable metadata on older messages.

## Media layout

The exporter’s normal database mode uses one stable directory per Telegram
peer and stores `media/<type>/<filename>` relative to that chat root in
`messages.media_path`. Older archives may instead contain JSON/HTML range
directories, and old database rows may point at a historical range that is no
longer in `chats.backup_path`. `tgbackman-media-reorganize` plans a separate
destination tree named with the peer identity, verifies every source with
size/SHA-256, creates Btrfs reflinks, and updates the database in one final
transaction. It never deletes the old tree. A manifest makes interrupted
copies resumable and is also the audit record needed before any later cleanup.

Use `tgbackman-legacy-*` for legacy-tree commands and `tgbackman-db-*` for
canonical database commands. The root script names remain compatibility
launchers for existing automation; they are not the implementation modules.

## Development commands

```bash
uv sync
uv run python -m pytest
cargo test --manifest-path tgbackman/Cargo.toml -- \
  --skip test_run_inventory_performance \
  --skip test_compute_media_stats_split_and_unofficial
cargo test --manifest-path tgsearch/Cargo.toml
```

The two skipped GUI tests are intentionally live-database performance tests;
run them on the mounted archive disk. All other tests use temporary databases
and do not require Telegram credentials or network access.
