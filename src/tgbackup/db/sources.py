"""Source-file provenance and media-integrity operations.

These writes are shared by JSON, HTML, unofficial SQLite, and API imports but
do not depend on any parser.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import time
import urllib.parse
import zlib
from typing import Optional

def archive_source_file(
    conn: sqlite3.Connection,
    path: str,
    source_format: str,
    *,
    archive_payload: bool,
) -> str:
    """Register a source and optionally embed its exact bytes in SQLite."""
    digest = hashlib.sha256()
    compressor = zlib.compressobj(level=9)
    spool = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024)
    size = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
            if archive_payload:
                spool.write(compressor.compress(chunk))
    compressed_size = 0
    compression = "none"
    if archive_payload:
        spool.write(compressor.flush())
        compressed_size = spool.tell()
        compression = "zlib"
    content_sha256 = digest.hexdigest()
    source_key = hashlib.sha256(
        f"{source_format}\0{content_sha256}".encode("utf-8")
    ).hexdigest()
    conn.execute(
        """INSERT INTO backup_sources (
               source_key, source_format, original_path, content_sha256,
               content_size, compressed_size, compression, payload, imported_unix
           ) VALUES (?, ?, ?, ?, ?, 0, 'none', X'', ?)
           ON CONFLICT(source_key) DO UPDATE SET
               original_path=excluded.original_path,
               imported_unix=excluded.imported_unix""",
        (
            source_key, source_format, os.path.abspath(path), content_sha256,
            size, int(time.time()),
        ),
    )
    if archive_payload:
        existing = conn.execute(
            "SELECT rowid, compression, compressed_size FROM backup_sources WHERE source_key=?",
            (source_key,),
        ).fetchone()
        if existing is None:
            spool.close()
            raise RuntimeError(f"failed to register source {path}")
        rowid, old_compression, old_size = int(existing[0]), str(existing[1]), int(existing[2])
        # Re-write the payload on every archived import.  Size/compression
        # metadata alone cannot detect same-size bit rot in an existing BLOB.
        del old_compression, old_size
        conn.execute(
            "UPDATE backup_sources SET compression=?, compressed_size=?, payload=zeroblob(?) WHERE source_key=?",
            (compression, compressed_size, compressed_size, source_key),
        )
        spool.seek(0)
        if hasattr(conn, "blobopen"):
            with conn.blobopen("backup_sources", "payload", rowid, readonly=False) as blob:
                for chunk in iter(lambda: spool.read(1024 * 1024), b""):
                    blob.write(chunk)
        else:  # pragma: no cover - Python 3.11+ provides incremental blobs.
            conn.execute(
                "UPDATE backup_sources SET payload=? WHERE source_key=?",
                (sqlite3.Binary(spool.read()), source_key),
            )
    spool.close()
    return source_key


def record_import(
    conn: sqlite3.Connection,
    source_key: str,
    source_format: str,
    path: str,
    chat_id: Optional[str],
    expected: int,
    imported: int,
    skipped: int = 0,
) -> None:
    conn.execute(
        """
        INSERT INTO backup_imports (
            source_key, source_format, original_path, chat_id,
            expected_messages, imported_messages, skipped_records, completed_unix
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_key) DO UPDATE SET
            chat_id=excluded.chat_id,
            expected_messages=excluded.expected_messages,
            imported_messages=excluded.imported_messages,
            skipped_records=excluded.skipped_records,
            completed_unix=excluded.completed_unix
        """,
        (
            source_key,
            source_format,
            os.path.abspath(path),
            chat_id,
            expected,
            imported,
            skipped,
            int(time.time()),
        ),
    )
    conn.execute(
        """INSERT INTO backup_import_files(
               source_key, original_path, source_format, chat_id,
               expected_messages, imported_messages, skipped_records, completed_unix
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(source_key, original_path) DO UPDATE SET
               chat_id=excluded.chat_id,
               expected_messages=excluded.expected_messages,
               imported_messages=excluded.imported_messages,
               skipped_records=excluded.skipped_records,
               completed_unix=excluded.completed_unix""",
        (
            source_key, os.path.abspath(path), source_format, chat_id,
            expected, imported, skipped, int(time.time()),
        ),
    )


def backfill_source_media_integrity(conn: sqlite3.Connection, source_key: str) -> None:
    """Hash every locally available attachment referenced by one imported source."""
    rows = conn.execute(
        """SELECT m.chat_id, m.message_id, m.media_path, m.media_size, m.media_sha256
           FROM messages AS m JOIN message_sources AS s
             ON s.chat_id=m.chat_id AND s.message_id=m.message_id
           WHERE s.source_key=? AND m.media_type IS NOT NULL""",
        (source_key,),
    ).fetchall()
    for chat_id, message_id, media_path, existing_size, existing_sha256 in rows:
        media_size: int | None = int(existing_size) if existing_size is not None else None
        media_sha256: str | None = str(existing_sha256) if existing_sha256 else None
        media_status = "missing"
        if media_path and not urllib.parse.urlparse(str(media_path)).scheme:
            local_path = os.path.abspath(str(media_path))
            if os.path.isfile(local_path):
                media_size = os.path.getsize(local_path)
                digest = hashlib.sha256()
                with open(local_path, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                media_sha256 = digest.hexdigest()
                media_status = "downloaded"
        conn.execute(
            """INSERT INTO message_source_media(
                   source_key, chat_id, message_id, media_path, media_size,
                   media_sha256, media_status, checked_unix
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(source_key, chat_id, message_id) DO UPDATE SET
                   media_path=excluded.media_path, media_size=excluded.media_size,
                   media_sha256=excluded.media_sha256, media_status=excluded.media_status,
                   checked_unix=excluded.checked_unix""",
            (source_key, chat_id, message_id, media_path, media_size, media_sha256, media_status, int(time.time())),
        )
        conn.execute(
            "UPDATE messages SET media_size=?, media_sha256=?, media_status=? WHERE chat_id=? AND message_id=?",
            (media_size, media_sha256, media_status, chat_id, message_id),
        )
