from collections.abc import Sequence

from backend.app.ingestion.models import DocumentChunk, DocumentPage


class UnsafeExtractionError(ValueError):
    """Unverified or incomplete provenance reached an indexing boundary."""


def validate_page_for_chunking(page: DocumentPage) -> None:
    """Reject unverified OCR and any non-indexed page carrying text."""
    method = str(page.extraction_method)
    if method == "unverified_ocr":
        raise UnsafeExtractionError("Unverified OCR text cannot be chunked")
    if page.text.strip() and not page.coverage_status.startswith(("indexed_", "approved_")):
        raise UnsafeExtractionError(
            f"Non-indexed page {page.page_number} cannot provide chunk text"
        )


def validate_chunks_for_indexing(chunks: Sequence[DocumentChunk]) -> None:
    """Enforce citation-safe provenance before embedding or Qdrant upload."""
    for chunk in chunks:
        metadata = chunk.metadata
        if str(metadata.extraction_method) == "unverified_ocr":
            raise UnsafeExtractionError("Unverified OCR chunk cannot be indexed")
        if metadata.page_number < 1:
            raise UnsafeExtractionError("Indexed chunk has an invalid PDF page number")
        if not metadata.coverage_status.startswith(("indexed_", "approved_")):
            raise UnsafeExtractionError("Indexed chunk has a non-indexed coverage status")
        if not metadata.citation_snippet or metadata.citation_snippet not in chunk.text:
            raise UnsafeExtractionError("Citation snippet is not verbatim chunk text")
        if not metadata.source_checksum:
            raise UnsafeExtractionError("Indexed chunk is missing its source checksum")
