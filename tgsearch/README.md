# tgsearch — Telegram Backup Database Search Utility

A high-performance, compiled Rust search utility designed for lightning-fast querying of consolidated Telegram backup SQLite databases. Features context window rendering, DST-aware chronological deduplication, multi-term logical FTS5 queries, automatic username aliasing, and output sanitisation.

## 🛠️ Components & Capabilities

### 1. `src/main.rs` (Primary Executable)
- **Role:** Performs sub-millisecond full-text searches using SQLite FTS5 index tables, matches context windows, removes overlaps, and renders outputs.
- **Inputs:**
  - `db_path`: Absolute path to a valid Telegram backup SQLite database (`telegram_backup.db`).
  - `query`: FTS5-compatible search string (supports unquoted `OR` operators, or quoted `"makeup OR alcohol"` literal phrases).
- **Outputs:** Streamlined, highly-formatted search results written directly to `stdout`.
- **Key CLI Options:**
  - `-c, --context <N>`: Context window size (lines before and after matches). Defaults to `3`.
  - `-l, --limit <N>`: Maximum matches to display. Set `-l -1` for unlimited.
  - `-c, --chat <substring>`: Filters results by chat name or ID.
  - `--dedupe`: Removes duplicate messages across overlapping backups. Matches timezone shifts (GMT vs BST) and normalizes aliases.
  - `--sanitise <rules-file>`: Loads target:replacement mappings from a file and anonymizes matching text, chat names, and senders.
  - `--no-header`: Suppresses the triple-line chat/message info boxes and shows a single clean divider line.
  - `--no-time`: Strips dates and times from displayed records.

### 2. `rules/` (Sanitisation Mapping)
- **`sanitise_rules.txt`**: Plain text file containing one name replacement rule per line (e.g. `username:REDACTED`). Supports both `:` and `=` delimiters and `#` comments.
- **`bad_lanaguge_rules.txt`**: Extends the standard rules by adding expletive-to-clean mappings (e.g. `fuck:frick`, `cunt:fool`) to allow sharing clean logs publicly.

### 3. `out/` (Export Folders)
- Standard repository folder for writing query output redirects (e.g., `tgsearch ... > out/output.txt`).

---

## 🔄 Execution Pipeline & Order of Operation

To compile and use the search utility, execute the steps in the following order:

### Step 1: Compilation
Compile the Rust project into a highly-optimized release binary. Run this command inside the project root:
```bash
cargo build --release
```
- **Result:** Produces a compiled native executable at `target/release/tgsearch`.

### Step 2: Formulate Sanitisation Rules (Optional)
If you wish to share log outputs publicly, inspect or modify the rule files inside `rules/` (such as `rules/sanitise_rules.txt`) to specify any names or phrases you want to anonymize.

### Step 3: Run Search Queries
Execute queries using the compiled binary.

#### Example A: Standard Deduplicated Context Search
```bash
target/release/tgsearch "/path/to/telegram_backup.db" "example-term" -c 1 -l 80 --dedupe
```
*Retrieves up to 80 unique matches with 1 line of preceding/succeeding context, deduplicating identical messages cleanly across backup overlaps.*

#### Example B: Sanitised and Compact Output (Perfect for Sharing)
```bash
target/release/tgsearch "/path/to/telegram_backup.db" "example-term OR another-term" --no-time -l -1 --dedupe --no-header --sanitise "rules/sanitise_rules.txt" > out/results.txt
```
*Queries unlimited matches containing either "makeup" or "alcohol", suppresses headers and timestamps, maps all real names and expletives to safe replacements, and writes a clean, anonymous result log directly to `out/makeup.txt`.*

---

## 🔒 Security & Privacy Notes
- **Local SQLite Access:** The database file is opened strictly in `read-only` mode (`SQLITE_OPEN_READ_ONLY | SQLITE_OPEN_URI`), preventing any accidental modifications to the source archive.
- **Leak Protection:** When utilizing `--sanitise`, every configured replacement is applied case-insensitively to chat names, senders, queries, and message text. Keep your local rules file outside version control when it contains private names.
