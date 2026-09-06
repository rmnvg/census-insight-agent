from backend.app.evaluation import (
    RetrievalEvaluationCase,
    RetrievalGoldTarget,
    evaluate_retrieval,
)
from backend.app.retrieval.models import (
    EvidenceSufficiency,
    RetrievalDiagnostics,
    RetrievalSearchResponse,
    RetrievedEvidence,
)


def test_retrieval_evaluation_measures_metrics_and_safety() -> None:
    case = RetrievalEvaluationCase(
        case_id="numeric",
        category="numeric",
        query="population",
        expected_targets=[
            RetrievalGoldTarget(
                document_id="synthetic",
                page_number=2,
                source_snippet="population is 150",
                verification_note="fixture",
            )
        ],
    )
    evidence = RetrievedEvidence(
        chunk_id="chunk",
        text="The population is 150.",
        document_title="Synthetic",
        document_id="synthetic",
        region="Test",
        page_number=2,
        citation_snippet="The population is 150.",
        section_path=["Population"],
        extraction_method="provided_markdown",
        coverage_status="indexed_provided_markdown",
        source_checksum="a" * 64,
        retrieval_score=1.0,
    )
    response = RetrievalSearchResponse(
        evidence=[evidence],
        evidence_sufficiency=EvidenceSufficiency(status="candidate_evidence", rule="test"),
        debug=RetrievalDiagnostics(
            dense_candidates=[],
            sparse_candidates=[],
            fused_ranking=[],
            applied_filters={},
            retrieval_time_ms=1,
        ),
    )

    metrics = evaluate_retrieval([case], lambda _: response)

    assert metrics.recall_at_5 == metrics.recall_at_10 == 1.0
    assert metrics.mean_reciprocal_rank == 1.0
    assert metrics.citation_page_accuracy == 1.0
    assert metrics.unsafe_evidence_count == 0
