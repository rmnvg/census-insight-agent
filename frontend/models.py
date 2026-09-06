from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SessionResponse(StrictModel):
    session_id: str
    created_at: datetime
    updated_at: datetime


class ChatRequest(StrictModel):
    session_id: str
    message: str = Field(min_length=1)


class EvidenceSpan(StrictModel):
    evidence_id: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)


class Citation(StrictModel):
    citation_id: str
    document_id: str
    document_title: str
    page_number: int = Field(gt=0)
    snippet: str
    chunk_id: str
    section_path: list[str]
    evidence_span: EvidenceSpan


class ClaimDerivation(StrictModel):
    operation: Literal["sum", "difference", "percentage_difference"]
    operands: list[float]
    result: float
    unit: str
    input_claim_ids: list[str]


class Claim(StrictModel):
    claim_id: str
    text: str
    citation_ids: list[str] = Field(default_factory=list)
    document_derived: bool = False
    metric: str | None = None
    region: str | None = None
    year: int | None = None
    population_scope: str | None = None
    residence_scope: str | None = None
    value: float | None = None
    unit: str | None = None
    derivation: ClaimDerivation | None = None


class ArtifactSummary(StrictModel):
    artifact_id: str
    artifact_type: Literal["chart", "table"]
    title: str
    filename: str
    media_type: str
    byte_size: int = Field(ge=0)
    sha256: str
    session_id: str
    run_id: str
    source_manifest_path: str
    download_url: str


class PendingArtifact(StrictModel):
    status: Literal["capability_pending"]
    artifact_type: Literal["chart", "table"]
    message: str


class ChatResponse(StrictModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    artifacts: list[ArtifactSummary | PendingArtifact] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    refusal: bool = False
    trace_id: str


class OperationalApiError(StrictModel):
    error_code: str
    message: str
    session_id: str | None = None
    trace_id: str | None = None
    retryable: bool = False


class BackendHealth(StrictModel):
    status: Literal["ok", "error"]
    service: str


class ExecutorHealth(StrictModel):
    status: Literal["ok", "error"]
    detail: str
    heartbeat_age_seconds: float | None = None


class QdrantHealth(StrictModel):
    status: Literal["ok", "error"]
    collection: str
    detail: str


class DocumentSummary(StrictModel):
    document_id: str
    title: str
    region: str
    source_checksum: str


class PageCoverageEntry(StrictModel):
    page_number: int = Field(gt=0)
    status: str
    reason: str | None = None


class CoverageReport(StrictModel):
    document_id: str
    pdf_page_count: int = Field(gt=0)
    indexed_pages: int = Field(ge=0)
    blank_decorative_pages: int = Field(ge=0)
    excluded_visual_pages: int = Field(ge=0)
    failed_mappings: int = Field(ge=0)
    percentage_pages_indexed: float = Field(ge=0, le=100)
    page_lists_by_status: dict[str, list[int]]
    pages: list[PageCoverageEntry]


class CoverageLimitation(StrictModel):
    document_id: str
    excluded_pages: list[int]
    statuses: list[str]
    message: str


class CoverageSummary(StrictModel):
    document: DocumentSummary
    coverage: CoverageReport
    limitations: list[CoverageLimitation] = Field(default_factory=list)


class SidebarSnapshot(StrictModel):
    backend_status: Literal["ok", "error"]
    executor_status: Literal["ok", "error"]
    document_status: Literal["ok", "error"]
    documents: list[DocumentSummary] = Field(default_factory=list)
    coverage_by_document: dict[str, CoverageSummary] = Field(default_factory=dict)


class ArtifactListing(StrictModel):
    artifacts: list[ArtifactSummary]


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timestamp: datetime
    event: str
    node: str | None = None
    details: dict[str, object] = Field(default_factory=dict)
    latency_ms: float | None = None


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tool_name: str
    arguments: dict[str, object] = Field(default_factory=dict)
    status: str
    result_count: int = 0
    error_type: str | None = None
    latency_ms: float = 0


class TraceResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: str
    run_id: str
    events: list[TraceEvent] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    status: str = "success"
    error_code: str | None = None
    run_status: str = "completed"
    answer_status: str = "answered"
    refusal_reason: str | None = None


class TraceEventView(StrictModel):
    event: str
    node: str | None = None
    details: dict[str, object] = Field(default_factory=dict)
    latency_ms: float | None = None


class ArtifactFile(StrictModel):
    filename: str
    media_type: str
    content: bytes
