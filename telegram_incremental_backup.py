#!/usr/bin/env python3
"""Compatibility launcher for the packaged ``tgbackup`` CLI.

The implementation lives under ``src/tgbackup`` so it can be imported by the
GUI worker, tests, and installed console entry points. Existing systemd units
and user commands can continue invoking this file unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import tgbackup.exporter as _implementation  # noqa: E402
from tgbackup.exporter import *  # noqa: F401,F403,E402
from tgbackup.exporter import main  # noqa: E402
from tgbackup.exporter import (
    _document_media_type,
    _filename_from_file,
    _legacy_media_type_for,
    _photo_size_expected_size,
    _photo_size_sort_key,
)


def _sync_compatibility_patches(*names: str) -> None:
    """Forward test/integration monkey-patches to the packaged module.

    The historical root module was the implementation module, so callers could
    patch globals such as ``connect_client`` directly.  It is now a launcher;
    keeping this small bridge means existing plugins and tests continue to
    patch the same public surface while the implementation remains modular.
    """

    for name in names:
        if name in globals():
            setattr(_implementation, name, globals()[name])


async def run_exports(*args, **kwargs):
    _sync_compatibility_patches("connect_client", "invoke_indexer")
    return await _implementation.run_exports(*args, **kwargs)


async def map_targets(*args, **kwargs):
    _sync_compatibility_patches("connect_client", "entity_description")
    return await _implementation.map_targets(*args, **kwargs)


def upsert_target(*args, **kwargs):
    _sync_compatibility_patches("entity_description")
    return _implementation.upsert_target(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
