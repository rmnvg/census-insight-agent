from datetime import UTC, datetime
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from backend.app.execution.contracts import (
    ArtifactDataset,
    ArtifactDescriptor,
    ExecutionRequest,
    ExecutionResult,
    SourceRecord,
)
from backend.app.retrieval.models import RetrievedEvidence

TaskType = Literal[
    "lookup",
    "summary",
    "comparison",
    "inconsistency_analysis",
    "artifact_chart",
    "artifact_table",
    "source_support",
    "clarification",
    "out_of_scope",
]


class ArtifactDataRequirement(BaseModel):
    """Source-data requirement for an artifact, separate from its presentation intent."""

    artifact_type: Literal["chart", "table"]
    metric: str = Field(min_length=1)
    year: int | None = Field(default=None, ge=1800, le=2200)
    regions: list[str] = Field(default_factory=list)
    population_scope: str | None = None
    residence_scope: str | None = None
    residence_scopes: list[Literal["total", "rural", "urban"]] = Field(default_factory=list)
    comparison: bool = False
    rank_all: bool = False
    rank_direction: Literal["max", "min"] | None = None


class TaskClassification(BaseModel):
    task_type: TaskType
    regions: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    referent: str | None = None
    clarification_question: str | None = None
    artifact_requirement: ArtifactDataRequirement | None = None
    reason: str = Field(min_length=1)


class ResolvedQuery(BaseModel):
    query: str
    requires_clarification: bool = False
    clarification_question: str | None = None
    task_type: TaskType | None = None
    regions: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    artifact_requirement: ArtifactDataRequirement | None = None


EvidenceRelevance = Literal[
    "direct_answer",
    "supporting_definition",
    "related_non_answering",
    "compatible_rounding",
    "conflicting",
    "insufficient",
]


class EvidenceAssessmentItem(BaseModel):
    evidence_id: str
    relevance: EvidenceRelevance
    entity_match: bool
    metric_match: bool
    year_match: bool
    has_explicit_value: bool
    has_unit: bool
    population_scope_match: bool = True
    residence_scope_match: bool = True
    unit_compatible: bool = True
    reason: str


class EvidenceAssessment(BaseModel):
    items: list[EvidenceAssessmentItem] = Field(default_factory=list)
    selected_evidence_ids: list[str] = Field(default_factory=list)
    sufficient: bool
    coverage_limitation_material: bool = False
    explanation: str


class AgentPlan(BaseModel):
    steps: list[str]
    retrieval_top_k: int = Field(default=10, ge=1, le=20)
    capability_pending: bool = False


ClaimUnit = Literal["percent", "percentage_points", "count", "ratio", "other"]


class ClaimDerivation(BaseModel):
    operation: Literal["sum", "difference", "percentage_difference"]
    operands: list[float] = Field(min_length=2, max_length=20)
    result: float
    unit: ClaimUnit
    input_claim_ids: list[str] = Field(min_length=2)


class DraftClaim(BaseModel):
    claim_id: str
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    document_derived: bool = False
    metric: str | None = None
    region: str | None = None
    year: int | None = None
    population_scope: str | None = None
    residence_scope: str | None = None
    value: float | None = None
    unit: ClaimUnit | None = None
    derivation: ClaimDerivation | None = None


class DraftAnswer(BaseModel):
    answer_markdown: str
    claims: list[DraftClaim] = Field(default_factory=list)
    refusal: bool = False
    limitations: list[str] = Field(default_factory=list)


class SupportAssessment(BaseModel):
    supported_claim_ids: list[str] = Field(default_factory=list)
    unsupported_claim_ids: list[str] = Field(default_factory=list)
    explanation: str


class CalculationRequest(BaseModel):
    description: str
    operation: Literal["sum", "difference", "percentage_difference"]
    values: list[float] = Field(min_length=2, max_length=20)
    evidence_ids: list[str] = Field(min_length=1)


class CalculationPlan(BaseModel):
    calculations: list[CalculationRequest] = Field(default_factory=list)


class CalculationResult(BaseModel):
    description: str
    operation: Literal["sum", "difference", "percentage_difference"]
    values: list[float]
    result: float
    evidence_ids: list[str]
    unit: ClaimUnit


class EvidenceSpan(BaseModel):
    evidence_id: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)


class Citation(BaseModel):
    citation_id: str
    document_id: str
    document_title: str
    page_number: int = Field(gt=0)
    snippet: str
    chunk_id: str
    section_path: list[str]
    evidence_span: EvidenceSpan


class AnswerClaim(BaseModel):
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
    unit: ClaimUnit | None = None
    derivation: ClaimDerivation | None = None


class ValidatedClaimRecord(BaseModel):
    """Bounded, JSON-safe claim memory; source text is deliberately excluded."""

    claim_id: str
    text: str
    metric: str | None = None
    region: str | None = None
    year: int | None = None
    population_scope: str | None = None
    residence_scope: str | None = None
    value: float | None = None
    displayed_value: str | None = None
    unit: ClaimUnit | None = None
    document_derived: bool = False
    derivation: ClaimDerivation | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    source_checksums: list[str | None] = Field(default_factory=list)


class ValidatedClaimTurn(BaseModel):
    run_id: str
    task_type: TaskType
    claims: list[ValidatedClaimRecord] = Field(default_factory=list)


class EvidenceReference(BaseModel):
    evidence_id: str
    document_id: str
    page_number: int = Field(gt=0)
    source_checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ArtifactResult(BaseModel):
    status: Literal["capability_pending"]
    artifact_type: Literal["chart", "table"]
    message: str


class AgentResponse(BaseModel):
    answer_markdown: str = Field(serialization_alias="answer")
    claims: list[AnswerClaim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    artifacts: list[ArtifactResult | ArtifactDescriptor] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    refusal: bool = False
    trace_id: str


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments: dict[str, object]
    status: Literal["ok", "error", "timeout", "limit_exceeded"]
    result_count: int = 0
    error_type: str | None = None
    latency_ms: float = 0


class TraceEvent(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event: str
    node: str | None = None
    details: dict[str, object] = Field(default_factory=dict)
    latency_ms: float | None = None


class RunTrace(BaseModel):
    session_id: str
    run_id: str
    events: list[TraceEvent]
    tool_calls: list[ToolCallRecord]
    errors: list[str]
    status: Literal["success", "failed"] = "success"
    error_code: str | None = None
    run_status: Literal["completed", "failed"] = "completed"
    answer_status: Literal["answered", "partial", "refused"] = "answered"
    refusal_reason: str | None = None


class AgentErrorResponse(BaseModel):
    error_code: str
    message: str
    session_id: str
    trace_id: str
    retryable: bool


class SessionRecord(BaseModel):
    session_id: str
    created_at: datetime
    updated_at: datetime


class SessionContextStatus(BaseModel):
    session_id: str
    validated_turn_count: int
    has_validated_comparison: bool


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    session_id: str
    thread_id: str
    run_id: str
    user_query: str
    resolved_query: str
    task_type: TaskType
    classification: TaskClassification
    artifact_requirement: ArtifactDataRequirement | None
    selected_skill: str | None
    skill_instructions: str | None
    plan: AgentPlan
    retrieved_evidence: list[RetrievedEvidence]
    selected_evidence: list[RetrievedEvidence]
    evidence_assessment: EvidenceAssessment | None
    available_limitations: list[str]
    evidence_sufficient: bool
    validation_error_codes: list[str]
    support_assessment: SupportAssessment | None
    draft_answer: DraftAnswer | None
    answer_claims: list[AnswerClaim]
    citations: list[Citation]
    limitations: list[str]
    artifacts: list[ArtifactResult | ArtifactDescriptor]
    tool_calls: list[ToolCallRecord]
    errors: list[str]
    retry_count: int
    final_response: AgentResponse | None
    trace_events: list[TraceEvent]
    conversation_summary: str
    calculations: list[CalculationResult]
    artifact_dataset: ArtifactDataset | None
    ranking_winner: SourceRecord | None
    generated_code: str | None
    execution_request: ExecutionRequest | None
    execution_result: ExecutionResult | None
    artifact_descriptors: list[ArtifactDescriptor]
    artifact_attempt: int
    artifact_errors: list[str]
    validated_claim_history: list[ValidatedClaimTurn]
    source_support_claims: list[ValidatedClaimRecord]
    checkpoint_schema_version: int
