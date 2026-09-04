from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

IndexedExtractionMethod = Literal[
    "provided_markdown", "pymupdf4llm_fallback", "approved_manual_transcription"
]
PageExtractionMethod = Literal[
    "provided_markdown",
    "pymupdf4llm_fallback",
    "approved_manual_transcription",
    "unverified_ocr",
]
PageCoverageStatus = Literal[
    "indexed_provided_markdown",
    "indexed_pymupdf4llm_fallback",
    "excluded_blank",
    "excluded_decorative",
    "excluded_unverified_visual",
    "failed_page_mapping",
    "approved_manual_transcription",
]
ExclusionCoverageStatus = Literal[
    "excluded_blank", "excluded_decorative", "excluded_unverified_visual"
]


class DocumentManifest(BaseModel):
    """Portable identity and provenance for one source document."""

    document_id: str
    title: str
    region: str
    pdf_path: str
    markdown_path: str | None = None
    source_checksum: str
    page_count: int = Field(gt=0)
    ingestion_version: str
    extraction_method: list[IndexedExtractionMethod]
    created_at: datetime


class ManifestOverride(BaseModel):
    """Optional explicit pairing and descriptive metadata override."""

    document_id: str
    title: str
    region: str
    pdf_filename: str
    markdown_filename: str | None = None


class DocumentPair(BaseModel):
    """Resolved source files before extraction."""

    document_id: str
    title: str
    region: str
    pdf_path: Path
    markdown_path: Path | None = None


class DocumentPage(BaseModel):
    """One citation-safe, one-based source page."""

    document_id: str
    page_number: int = Field(gt=0)
    text: str
    extraction_method: PageExtractionMethod | None
    coverage_status: PageCoverageStatus


class PageCoverageDecision(BaseModel):
    """Checksum-bound decision to exclude one reviewed authoritative PDF page."""

    document_id: str
    page_number: int = Field(gt=0)
    coverage_status: ExclusionCoverageStatus
    reason: str = Field(min_length=1)
    ocr_confidence: float | None = Field(default=None, ge=0, le=100)
    review_status: Literal["approved_exclusion", "unverified"]
    source_checksum: str = Field(min_length=64, max_length=64)


class ManualTranscriptionRecord(BaseModel):
    """Human-reviewed extension point; never populated by automated extraction."""

    document_id: str
    page_number: int = Field(gt=0)
    transcription: str = Field(min_length=1)
    structured_values: dict[str, str | int | float | bool | None] | None = None
    reviewer: str = Field(min_length=1)
    reviewed_at: datetime
    source_checksum: str = Field(min_length=64, max_length=64)
    verification_notes: str = Field(min_length=1)
    status: Literal["approved"]

    @field_validator("reviewer")
    @classmethod
    def reviewer_must_be_human_identity(cls, value: str) -> str:
        if value.strip().casefold() in {"codex", "gemini", "tesseract", "ai", "llm"}:
            raise ValueError("reviewer must identify a human reviewer")
        return value.strip()

    @field_validator("reviewed_at")
    @classmethod
    def reviewed_at_must_include_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone")
        return value


class PageCoverageEntry(BaseModel):
    """Exactly one coverage disposition for one one-based PDF page."""

    page_number: int = Field(gt=0)
    status: PageCoverageStatus
    reason: str | None = None


class DocumentCoverageReport(BaseModel):
    """Complete page-level coverage without implying excluded content was absent."""

    document_id: str
    pdf_page_count: int = Field(gt=0)
    indexed_pages: int = Field(ge=0)
    blank_decorative_pages: int = Field(ge=0)
    excluded_visual_pages: int = Field(ge=0)
    failed_mappings: int = Field(ge=0)
    percentage_pages_indexed: float = Field(ge=0, le=100)
    page_lists_by_status: dict[PageCoverageStatus, list[int]]
    pages: list[PageCoverageEntry]


class CoverageLimitation(BaseModel):
    """Typed limitation exposed to later agent layers."""

    document_id: str
    excluded_pages: list[int]
    statuses: list[PageCoverageStatus]
    message: str


class ChunkMetadata(BaseModel):
    """Citation and filtering metadata stored in Qdrant payloads."""

    document_id: str
    document_title: str
    region: str
    page_number: int = Field(gt=0)
    section_path: list[str]
    chunk_index: int = Field(ge=0)
    citation_snippet: str
    citation_limitation: str | None = None
    extraction_method: IndexedExtractionMethod
    coverage_status: PageCoverageStatus
    source_checksum: str


class DocumentChunk(BaseModel):
    """A retrievable unit that never crosses a page boundary."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata


class CitationEvidence(BaseModel):
    """Minimal verbatim evidence required to support a citation."""

    document_id: str
    document_title: str
    page_number: int = Field(gt=0)
    citation_snippet: str


class DocumentIngestionResult(BaseModel):
    document_id: str
    pdf_filename: str = ""
    markdown_filename: str | None = None
    paired: bool = False
    pages: int = 0
    chunks: int = 0
    dense_embeddings: int = 0
    sparse_embeddings: int = 0
    provided_markdown_pages: list[int] = Field(default_factory=list)
    fallback_pages: list[int] = Field(default_factory=list)
    unresolved_pages: list[int] = Field(default_factory=list)
    duplicate_pages: list[int] = Field(default_factory=list)
    empty_pages: list[int] = Field(default_factory=list)
    excluded_pages: list[int] = Field(default_factory=list)
    coverage: DocumentCoverageReport | None = None
    limitations: list[CoverageLimitation] = Field(default_factory=list)
    page_count_mismatch: bool = False
    citation_ready_chunks: int = 0
    chunks_missing_provenance: int = 0
    skipped_unchanged: bool = False
    failures: list[str] = Field(default_factory=list)


class IngestionReport(BaseModel):
    discovered_documents: int = 0
    processed_documents: int = 0
    skipped_unchanged_documents: int = 0
    pages: int = 0
    chunks: int = 0
    dense_embedding_count: int = 0
    sparse_embedding_count: int = 0
    dense_embedding_request_count: int = 0
    qdrant_upsert_operations: int = 0
    qdrant_upserted_points: int = 0
    fallback_pages: int = 0
    indexed_pages: int = 0
    excluded_pages: int = 0
    failed_page_mappings: int = 0
    coverage_limitations: list[CoverageLimitation] = Field(default_factory=list)
    citation_ready_chunks: int = 0
    chunks_missing_provenance: int = 0
    review_required: bool = False
    failures: list[str] = Field(default_factory=list)
    documents: list[DocumentIngestionResult] = Field(default_factory=list)
    estimated_dense_embedding_requests: int = 0
    elapsed_seconds: float = 0
    dry_run: bool = False


class PageMappingResult(BaseModel):
    pages: list[DocumentPage]
    markdown_pages: list[int] = Field(default_factory=list)
    fallback_pages: list[int] = Field(default_factory=list)
    unresolved_pages: list[int] = Field(default_factory=list)
    duplicate_pages: list[int] = Field(default_factory=list)
    empty_pages: list[int] = Field(default_factory=list)
    coverage: DocumentCoverageReport
    page_count_mismatch: bool = False
