class IngestionError(RuntimeError):
    """Base error for safe document ingestion failures."""


class SourcePathError(IngestionError):
    """A requested path is outside the configured source directory."""


class DocumentPairingError(IngestionError):
    """Source files cannot be paired deterministically."""


class PageMappingError(IngestionError):
    """Citation-safe Markdown-to-PDF page mapping could not be established."""

    def __init__(
        self,
        message: str,
        *,
        unresolved_pages: list[int] | None = None,
        duplicate_pages: list[int] | None = None,
        page_count_mismatch: bool = False,
    ) -> None:
        super().__init__(message)
        self.unresolved_pages = unresolved_pages or []
        self.duplicate_pages = duplicate_pages or []
        self.page_count_mismatch = page_count_mismatch
