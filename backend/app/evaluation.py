from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field

from backend.app.retrieval.models import RetrievalSearchResponse


class RetrievalGoldTarget(BaseModel):
    document_id: str
    page_number: int = Field(gt=0)
    source_snippet: str = Field(min_length=1)
    verification_note: str = Field(min_length=1)


class RetrievalEvaluationCase(BaseModel):
    case_id: str
    category: Literal["numeric", "definition", "cross_document", "filtered", "unanswerable"]
    query: str
    expected_targets: list[RetrievalGoldTarget] = Field(default_factory=list)
    answerable: bool = True
    document_ids: list[str] | None = None
    regions: list[str] | None = None
    top_k: int = Field(default=10, gt=0, le=50)


class CaseEvaluation(BaseModel):
    case_id: str
    retrieved: int
    matched_targets_at_5: int
    matched_targets_at_10: int
    target_count: int
    reciprocal_rank: float
    filter_correct: bool
    citation_metadata_correct: bool
    evidence_sufficiency: str
    dense_candidates: int
    sparse_candidates: int
    fused_candidates: int


class RetrievalEvaluationMetrics(BaseModel):
    cases: int
    answerable_cases: int
    targets: int
    recall_at_5: float
    recall_at_10: float
    mean_reciprocal_rank: float
    citation_page_accuracy: float
    document_filter_accuracy: float
    region_filter_accuracy: float
    unsafe_evidence_count: int
    insufficient_out_of_corpus_count: int
    per_case: list[CaseEvaluation]


def evaluate_retrieval(
    cases: list[RetrievalEvaluationCase],
    search: Callable[[RetrievalEvaluationCase], RetrievalSearchResponse],
) -> RetrievalEvaluationMetrics:
    """Evaluate retrieval against manually source-verified document/page targets."""
    total_targets = sum(len(case.expected_targets) for case in cases if case.answerable)
    hits_5 = hits_10 = 0
    reciprocal_ranks: list[float] = []
    citation_checks: list[bool] = []
    document_filter_checks: list[bool] = []
    region_filter_checks: list[bool] = []
    unsafe = insufficient = 0
    details: list[CaseEvaluation] = []
    for case in cases:
        response = search(case)
        evidence = response.evidence
        target_keys = {(item.document_id, item.page_number) for item in case.expected_targets}
        ranks = [
            rank
            for rank, item in enumerate(evidence, start=1)
            if (item.document_id, item.page_number) in target_keys
        ]
        matched_5 = len(
            target_keys & {(item.document_id, item.page_number) for item in evidence[:5]}
        )
        matched_10 = len(
            target_keys & {(item.document_id, item.page_number) for item in evidence[:10]}
        )
        if case.answerable:
            hits_5 += matched_5
            hits_10 += matched_10
            reciprocal_ranks.append(1 / min(ranks) if ranks else 0.0)
        citation_ok = all(
            item.page_number > 0 and item.citation_snippet in item.text for item in evidence
        )
        citation_checks.append(citation_ok)
        unsafe += sum(
            item.extraction_method == "unverified_ocr"
            or item.coverage_status.startswith("excluded_")
            for item in evidence
        )
        doc_ok = not case.document_ids or all(
            item.document_id in case.document_ids for item in evidence
        )
        region_ok = not case.regions or all(item.region in case.regions for item in evidence)
        if case.document_ids:
            document_filter_checks.append(doc_ok)
        if case.regions:
            region_filter_checks.append(region_ok)
        if not case.answerable and response.evidence_sufficiency.status == "insufficient_evidence":
            insufficient += 1
        diagnostics = response.debug
        details.append(
            CaseEvaluation(
                case_id=case.case_id,
                retrieved=len(evidence),
                matched_targets_at_5=matched_5,
                matched_targets_at_10=matched_10,
                target_count=len(target_keys),
                reciprocal_rank=1 / min(ranks) if ranks else 0.0,
                filter_correct=doc_ok and region_ok,
                citation_metadata_correct=citation_ok,
                evidence_sufficiency=response.evidence_sufficiency.status,
                dense_candidates=len(diagnostics.dense_candidates) if diagnostics else 0,
                sparse_candidates=len(diagnostics.sparse_candidates) if diagnostics else 0,
                fused_candidates=len(diagnostics.fused_ranking) if diagnostics else 0,
            )
        )
    answerable_count = sum(case.answerable for case in cases)
    return RetrievalEvaluationMetrics(
        cases=len(cases),
        answerable_cases=answerable_count,
        targets=total_targets,
        recall_at_5=hits_5 / (total_targets or 1),
        recall_at_10=hits_10 / (total_targets or 1),
        mean_reciprocal_rank=sum(reciprocal_ranks) / (len(reciprocal_ranks) or 1),
        citation_page_accuracy=sum(citation_checks) / (len(citation_checks) or 1),
        document_filter_accuracy=sum(document_filter_checks) / (len(document_filter_checks) or 1),
        region_filter_accuracy=sum(region_filter_checks) / (len(region_filter_checks) or 1),
        unsafe_evidence_count=unsafe,
        insufficient_out_of_corpus_count=insufficient,
        per_case=details,
    )
