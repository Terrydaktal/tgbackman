"""Errors exposed by the backup services and CLI."""


class ExportError(RuntimeError):
    """Expected exporter failure with a user-facing message."""
