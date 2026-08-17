#!/usr/bin/env python3
"""Measure diagnostic overhead on a representative offline export path.

The benchmark deliberately does not contact Telegram.  It exercises the hot
operations performed for each staged record (record construction and JSON
serialization), then compares progress accounting and operation-boundary JSONL
events.  Use the output as a deployment-machine comparison, not as a network
or disk-throughput guarantee.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

from tgbackup.diagnostics import DiagnosticEvent, DiagnosticRecorder
from tgbackup.progress import ProgressReporter


def _record(message_id: int) -> dict[str, object]:
    return {
        "id": message_id,
        "date_unixtime": str(message_id),
        "from": "benchmark",
        "from_id": "user:1",
        "text": "synthetic staged message",
        "media_type": None,
    }


def _run_export_like(
    count: int,
    *,
    progress: Optional[ProgressReporter] = None,
    recorder: Optional[DiagnosticRecorder] = None,
) -> None:
    if recorder is not None:
        recorder.emit(
            DiagnosticEvent(
                event="backup_started",
                component="benchmark",
                fields={"scope": "synthetic", "messages": count},
            )
        )
    for message_id in range(1, count + 1):
        record = _record(message_id)
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if progress is not None:
            progress.observe(record, None)
    if recorder is not None:
        recorder.emit(
            DiagnosticEvent(
                event="backup_finished",
                component="benchmark",
                fields={"status": "completed", "messages": count},
            )
        )


def _measure(fn: Callable[[], None]) -> float:
    started = time.perf_counter()
    fn()
    return time.perf_counter() - started


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
    return {
        "median_seconds": statistics.median(samples),
        "p95_seconds": ordered[p95_index],
        "min_seconds": min(samples),
        "max_seconds": max(samples),
    }


def _samples(count: int, repeats: int, mode: str, directory: Path) -> list[float]:
    results: list[float] = []
    for repeat in range(repeats):
        recorder: Optional[DiagnosticRecorder] = None
        progress: Optional[ProgressReporter] = None
        if mode == "progress":
            progress = ProgressReporter(
                "benchmark",
                enabled=True,
                every=count + 1,
                output=lambda _: None,
            )
        elif mode == "observable":
            recorder = DiagnosticRecorder(directory / f"events-{repeat}.jsonl")
        results.append(_measure(lambda: _run_export_like(count, progress=progress, recorder=recorder)))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.messages <= 0:
        parser.error("--messages must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    with tempfile.TemporaryDirectory(prefix="tgbackman-diagnostics-") as raw_directory:
        directory = Path(raw_directory)
        samples = {
            mode: _samples(args.messages, args.repeats, mode, directory)
            for mode in ("minimal", "progress", "observable")
        }
        summary = {mode: _summary(values) for mode, values in samples.items()}
    baseline = summary["minimal"]["median_seconds"]
    print(
        json.dumps(
            {
                "messages": args.messages,
                "repeats": args.repeats,
                "path": "record construction + JSON serialization; no Telegram/network",
                "modes": summary,
                "median_ratio_to_minimal": {
                    mode: (values["median_seconds"] / baseline if baseline else None)
                    for mode, values in summary.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
