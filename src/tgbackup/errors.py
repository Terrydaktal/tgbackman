"""Errors exposed by the backup services and CLI."""


class ExportError(RuntimeError):
    """Expected exporter failure with a user-facing message."""

    def __init__(self, message: str, *, code: str = "export_error") -> None:
        super().__init__(message)
        self.code = code
