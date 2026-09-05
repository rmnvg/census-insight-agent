import pytest

from backend.app.agent.citations import validate_and_materialize_citations
from backend.app.agent.models import DraftAnswer, DraftClaim, EvidenceSpan
from backend.app.agent.quotes import (
    quote_supports_claim,
    resolve_evidence_span,
    select_evidence_span_with_diagnostic,
)
from backend.app.retrieval.models import RetrievedEvidence


def evidence(text: str, *, page_number: int = 10) -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk_id=f"chunk-{page_number}",
        text=text,
        document_title="Primary Census Abstract Data Highlights: Karnataka",
        document_id="census-2011-karnataka-pca-highlights",
        region="Karnataka",
        page_number=page_number,
        citation_snippet=text[: min(40, len(text))],
        section_path=["Literacy"],
        extraction_method="provided_markdown",
        coverage_status="indexed_provided_markdown",
        retrieval_score=1.0,
    )


def claim(text: str, evidence_id: str = "chunk-10", claim_id: str = "claim-1") -> DraftClaim:
    return DraftClaim(claim_id=claim_id, text=text, evidence_ids=[evidence_id])


def test_definition_first_chunk_produces_exact_value_bearing_quote() -> None:
    item = evidence(
        "- Effective literacy rate means literates aged 7 and above.\n"
        "- The Literacy Rate of the State increased from 66.64 per cent in 2001 "
        "to 75.36 per cent in 2011.\n"
    )
    draft = DraftAnswer(
        answer_markdown="Karnataka literacy was 75.36 percent in 2011.",
        claims=[claim("Karnataka literacy was 75.36 percent in 2011.")],
    )
    result = validate_and_materialize_citations(draft, [item])
    citation = result.citations[0]

    assert result.valid
    assert "75.36" in citation.snippet
    assert (
        citation.snippet
        == item.text[citation.evidence_span.start_offset : citation.evidence_span.end_offset]
    )


def test_fabricated_and_wrong_provenance_spans_are_rejected() -> None:
    item = evidence("Karnataka literacy was 75.36 percent in 2011.")
    with pytest.raises(ValueError, match="wrong chunk"):
        resolve_evidence_span(
            EvidenceSpan(evidence_id="invented", start_offset=0, end_offset=5), item
        )
    with pytest.raises(ValueError, match="outside"):
        resolve_evidence_span(
            EvidenceSpan(evidence_id=item.chunk_id, start_offset=0, end_offset=999), item
        )


def test_wrong_year_or_category_numeric_match_is_rejected() -> None:
    item = evidence("Karnataka rural literacy was 75.36 percent in 2001.")
    assert not quote_supports_claim(
        claim("Karnataka rural literacy was 75.36 percent in 2011."), item.text, item
    )
    urban = evidence("Karnataka urban literacy was 75.36 percent in 2011.")
    assert not quote_supports_claim(
        claim("Karnataka rural literacy was 75.36 percent in 2011."), urban.text, urban
    )


def test_table_quote_keeps_contiguous_header_and_relevant_row() -> None:
    item = evidence(
        "| State | Literacy Rate 2011 |\n"
        "|---|---|\n"
        "| KARNATAKA | 75.36 percent |\n"
        "| Belgaum | 73.48 percent |\n"
    )
    draft = DraftAnswer(
        answer_markdown="Karnataka literacy was 75.36 percent in 2011.",
        claims=[claim("Karnataka literacy was 75.36 percent in 2011.")],
    )
    result = validate_and_materialize_citations(draft, [item])
    assert result.valid
    assert "Literacy Rate 2011" in result.citations[0].snippet
    assert "KARNATAKA | 75.36" in result.citations[0].snippet


def test_two_claims_from_one_chunk_receive_distinct_quotes() -> None:
    item = evidence(
        "Karnataka literacy was 75.36 percent in 2011. "
        "Karnataka urban literacy was 85.78 percent in 2011."
    )
    draft = DraftAnswer(
        answer_markdown="Two findings",
        claims=[
            claim("Karnataka literacy was 75.36 percent in 2011.", claim_id="total"),
            claim("Karnataka urban literacy was 85.78 percent in 2011.", claim_id="urban"),
        ],
    )
    result = validate_and_materialize_citations(draft, [item])
    assert result.valid
    assert result.citations[0].snippet != result.citations[1].snippet


def test_wrong_document_page_cannot_be_invented() -> None:
    item = evidence("Karnataka literacy was 75.36 percent in 2011.", page_number=50)
    draft = DraftAnswer(
        answer_markdown="Finding",
        claims=[claim("Karnataka literacy was 75.36 percent in 2011.", "chunk-50")],
    )
    result = validate_and_materialize_citations(draft, [item])
    assert result.citations[0].document_id == item.document_id
    assert result.citations[0].page_number == 50


def test_structured_filler_wording_uses_metadata_and_table_context() -> None:
    item = evidence(
        "| State | Literacy Rate | 2011 Total |\n"
        "|---|---|---|\n"
        "| <b>KARNATAKA</b> | Persons | 75.36 |\n"
    )
    structured = DraftClaim(
        claim_id="structured",
        text="For the same year, Karnataka had a total literacy rate of 75.36% for persons.",
        evidence_ids=[item.chunk_id],
        metric="literacy_rate",
        region="Karnataka",
        year=2011,
        population_scope="persons",
        residence_scope="total",
        value=75.36,
        unit="percent",
    )
    span, diagnostic = select_evidence_span_with_diagnostic(structured, item)
    assert span is not None
    assert diagnostic.match_type == "table"
    quote = item.text[span.start_offset : span.end_offset]
    assert quote in item.text
    assert "<b>KARNATAKA</b>" in quote


def test_prose_per_cent_supports_structured_percent_unit_via_region_metadata() -> None:
    item = evidence("The literacy rate increased to 75.36 per cent in 2011.")
    structured = DraftClaim(
        claim_id="structured",
        text="Karnataka literacy was 75.36% in 2011.",
        evidence_ids=[item.chunk_id],
        metric="literacy_rate",
        region="Karnataka",
        year=2011,
        value=75.36,
        unit="percent",
    )
    assert quote_supports_claim(structured, item.text, item)
