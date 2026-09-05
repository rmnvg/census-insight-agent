from typing import Literal

from pydantic import BaseModel, Field

from backend.app.ingestion.models import IndexedExtractionMethod, PageCoverageStatus


class SparseVectorData(BaseModel):
    indices: list[int]
    values: list[float]


class RetrievedEvidence(BaseModel):
    chunk_id: str
    text: str
    document_title: str
    document_id: str
    region: str
    page_number: int = Field(gt=0)
    citation_snippet: str
    citation_limitation: str | None = None
    section_path: list[str]
    extraction_method: IndexedExtractionMethod
    coverage_status: PageCoverageStatus
    source_checksum: str = ""
    retrieval_score: float


class RetrievalCandidate(BaseModel):
    chunk_id: str
    score: float


class RetrievalDiagnostics(BaseModel):
    dense_candidates: list[RetrievalCandidate]
    sparse_candidates: list[RetrievalCandidate]
    fused_ranking: list[RetrievalCandidate]
    applied_filters: dict[str, list[str]]
    retrieval_time_ms: float


class RetrievalSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    document_ids: list[str] | None = None
    regions: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    debug: bool = False


class EvidenceSufficiency(BaseModel):
    status: Literal["candidate_evidence", "insufficient_evidence"]
    rule: str
    requires_claim_validation: bool = True


class RetrievalSearchResponse(BaseModel):
    evidence: list[RetrievedEvidence]
    evidence_sufficiency: EvidenceSufficiency
    debug: RetrievalDiagnostics | None = None


class QdrantHealthResponse(BaseModel):
    status: Literal["ok", "error"]
    collection: str
    detail: str


class DocumentSummary(BaseModel):
    document_id: str
    title: str
    region: str
    source_checksum: str
