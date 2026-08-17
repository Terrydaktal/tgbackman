# Debuggability contract

This project uses a small, bounded diagnostic contract for long-running
Telegram exports and the Rust viewer. Diagnostics are supplementary evidence;
they never replace staged records, SQLite transactions, media hashes, or the
database verifier.

## Build identity

Every Python diagnostic event and Rust GUI state/crash record includes the
package version, source revision (when built from a Git checkout), dirty-tree
status, runtime version and build profile. Release GUI builds retain
source-line tables and must retain the matching binary and symbols together
with the release revision. A build with `revision=unknown` or `dirty=1` is
usable but not suitable for a post-mortem claim of exact source
correspondence.

## Evidence locations

- Python active export state: `$XDG_STATE_HOME/tgbackman/active-operation.json`
- GUI active state: `$XDG_STATE_HOME/tgbackman/gui-state.json`
- Uncaught Python exceptions: `$XDG_STATE_HOME/tgbackman/crashes/`
- Rust GUI panics: `$XDG_STATE_HOME/tgbackman/crashes/`
- Default structured lifecycle events: `$XDG_STATE_HOME/tgbackman/events.jsonl`
  (override with `TGBACKMAN_DIAGNOSTICS_FILE` or `--diagnostics-file`)
- Completed operation records: `$XDG_STATE_HOME/tgbackman/operations/`

The default state and crash files are mode `0600`. They contain operation
metadata, error types, bounded tracebacks and build identity, but not message
bodies or known Telegram credentials. Active snapshots also include process
RSS and thread count. Chat names, filenames and paths can still be personal
information and must be handled accordingly. Redaction covers credential
values loaded by the application; it is not a guarantee against an arbitrary
third-party exception string containing a secret.

## Collection

While an export is running:

```text
tgbackman-backup snapshot
```

After an unclean shutdown, inspect the database and then explicitly reap
attempts that have been running longer than the chosen lease:

```bash
uv run tgbackman-backup reap-stale --older-than 86400
```

Structured lifecycle events are enabled by default. Set
`TGBACKMAN_DIAGNOSTICS_FILE` or pass `--diagnostics-file PATH` to select another
private destination. Diagnostic sinks are best effort; a full or unavailable
sink must never fail an export. The JSONL sink rotates at 4 MiB and retains one
previous segment. Terminal operation snapshots are retained in a bounded set of
32 files.

## Performance tiers

- **Minimal (default):** durable run/attempt error fields, build identity,
  atomic active-state snapshot, panic/uncaught-exception file and a rotating
  JSONL lifecycle stream. The stream is written at operation/chat boundaries,
  not per message.
- **Observable destination:** set `TGBACKMAN_DIAGNOSTICS_FILE` or
  `--diagnostics-file PATH` to route that same bounded stream to a collector.
- **Progress:** terminal progress is independently controlled by the run
  flags. `--no-progress` removes message and media progress callbacks from the
  hot path.

The repository does not claim a universal zero-cost guarantee. Release
performance checks should compare the minimal, progress and observable tiers on
the representative archive workload before changing the evidence contract.
`uv run python tools/benchmark_diagnostics.py` exercises record construction,
JSON serialization, progress accounting and bounded lifecycle JSONL events. It
does not model Telegram/network throughput, media hashing, or a real archive
filesystem.

## Failure contract

Failed runs retain their staging rows whenever possible, keep a concise human
error for existing tooling, and add exception type, phase, bounded traceback,
causal chain, operation ID and build revision to the run and attempt records.
The canonical database also retains lifecycle rows in
`telegram_backup_diagnostic_events`; purge operations do not remove those
event rows. GUI active/blacklist mutations append equivalent local events,
including failed mutation attempts. Events record the configured actor
(`TGBACKMAN_ACTOR`, then the OS user), host, process, writer role, reason and
outcome. Python-written events include a bounded previous-hash/integrity field
that detects ordinary accidental edits within the local event chain; it is not
a cryptographic external trust anchor and does not prevent a privileged user
from rewriting the database.
`verify --check-db` validates the Python event hash chain when checking an
export ledger. The SQLite event ledger retains the newest 100,000 rows; when
older rows are evicted, the first retained hash is treated as an external
predecessor anchor rather than as a fully verifiable history.
Unexpected process termination may leave an attempt in `running`; the active
snapshot and process ID distinguish that case from a clean failure. Staging
boundaries update a durable attempt heartbeat, and `reap-stale` acquires the
same exporter lock before marking attempts abandoned. Choose a lease longer
than the longest expected media/API stall.

## Supported diagnostic scenarios

The assurance suite covers data correctness and recovery. Diagnostic-specific
checks must cover network interruption, FloodWait, media truncation, SQLite
busy/disk-full behavior, worker panic/disconnect, redaction, evidence
truncation and minimal-vs-observable overhead. Release artifacts must be
symbolizable using the recorded revision and matching symbol files.
