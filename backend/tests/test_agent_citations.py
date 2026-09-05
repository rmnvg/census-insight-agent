from backend.app.agent.citations import validate_and_materialize_citations
from backend.app.agent.models import DraftAnswer, DraftClaim
from backend.app.retrieval.models import RetrievedEvidence


def evidence(**updates: object) -> RetrievedEvidence:
    values: dict[str, object] = {
        "chunk_id": "chunk-1",
        "text": "## Literacy\n\nThe literacy rate was 75.36 per cent.",
        "document_title": "Karnataka Census",
        "document_id": "karnataka",
        "region": "Karnataka",
        "page_number": 52,
        "citation_snippet": "The literacy rate was 75.36 per cent.",
        "section_path": ["Literacy"],
        "extraction_method": "provided_markdown",
        "coverage_status": "indexed_provided_markdown",
        "retrieval_score": 1.0,
    }
    values.update(updates)
    return RetrievedEvidence.model_validate(values)


def draft(evidence_id: str = "chunk-1") -> DraftAnswer:
    return DraftAnswer(
        answer_markdown="The literacy rate was 75.36 per cent.",
        claims=[
            DraftClaim(claim_id="claim-1", text="Literacy was 75.36%.", evidence_ids=[evidence_id])
        ],
    )


def test_materializes_metadata_from_trusted_evidence() -> None:
    result = validate_and_materialize_citations(draft(), [evidence()])
    assert result.valid
    assert result.citations[0].page_number == 52
    assert result.citations[0].snippet in evidence().text


def test_invented_evidence_id_is_rejected() -> None:
    result = validate_and_materialize_citations(draft("invented"), [evidence()])
    assert not result.valid
    assert "unknown evidence" in result.errors[0]


def test_non_verbatim_and_excluded_evidence_are_rejected() -> None:
    invalid_snippet = validate_and_materialize_citations(
        draft(), [evidence(citation_snippet="normalized missing text")]
    )
    excluded = validate_and_materialize_citations(
        draft(), [evidence(coverage_status="excluded_unverified_visual")]
    )
    assert not invalid_snippet.valid
    assert not excluded.valid


def test_heading_only_evidence_is_rejected() -> None:
    result = validate_and_materialize_citations(
        draft(), [evidence(text="## Literacy", citation_snippet="## Literacy")]
    )
    assert not result.valid
