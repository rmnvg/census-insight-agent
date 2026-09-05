from backend.app.agent.calculations import (
    add_deterministic_derived_claims,
    validated_calculation,
)
from backend.app.agent.citations import validate_and_materialize_citations
from backend.app.agent.models import CalculationRequest, DraftAnswer, DraftClaim
from backend.app.retrieval.models import RetrievedEvidence


def rate_evidence(region: str, value: float) -> RetrievedEvidence:
    text = f"| State | Literacy Rate 2011 |\n|---|---|\n| {region.upper()} | {value}% |\n"
    return RetrievedEvidence(
        chunk_id=f"chunk-{region.casefold()}",
        text=text,
        document_title=f"Census 2011 {region}",
        document_id=f"doc-{region.casefold()}",
        region=region,
        page_number=50,
        citation_snippet=text,
        section_path=["Literates and Literacy Rate by residence : 2011 (PERSONS)"],
        extraction_method="provided_markdown",
        coverage_status="indexed_provided_markdown",
        retrieval_score=1.0,
    )


def request() -> CalculationRequest:
    return CalculationRequest(
        description="Compare rates",
        operation="percentage_difference",
        values=[75.36, 72.9],
        evidence_ids=["chunk-karnataka", "chunk-odisha"],
    )


def source_claims() -> list[DraftClaim]:
    return [
        DraftClaim(
            claim_id="karnataka",
            text="Karnataka literacy was 75.36% in 2011.",
            evidence_ids=["chunk-karnataka"],
            metric="literacy_rate",
            region="Karnataka",
            year=2011,
            population_scope="persons",
            residence_scope="total",
            value=75.36,
            unit="percent",
        ),
        DraftClaim(
            claim_id="odisha",
            text="Odisha literacy was 72.9% in 2011.",
            evidence_ids=["chunk-odisha"],
            metric="literacy_rate",
            region="Odisha",
            year=2011,
            population_scope="persons",
            residence_scope="total",
            value=72.9,
            unit="percent",
        ),
    ]


def test_rate_comparison_defaults_to_percentage_points() -> None:
    evidence = [rate_evidence("Karnataka", 75.36), rate_evidence("Odisha", 72.9)]
    result = validated_calculation("Compare Karnataka with Odisha", request(), evidence)
    assert result is not None
    assert result.operation == "difference"
    assert result.result == 2.46
    assert result.unit == "percentage_points"


def test_explicit_relative_percent_request_uses_percentage_difference() -> None:
    evidence = [rate_evidence("Karnataka", 75.36), rate_evidence("Odisha", 72.9)]
    result = validated_calculation(
        "How many percent higher was Karnataka than Odisha?", request(), evidence
    )
    assert result is not None
    assert result.operation == "percentage_difference"
    assert round(result.result, 2) == 3.37
    assert result.unit == "percent"


def test_derived_claim_is_programmatic_and_requires_both_input_citations() -> None:
    evidence = [rate_evidence("Karnataka", 75.36), rate_evidence("Odisha", 72.9)]
    calculation = validated_calculation("Compare the rates", request(), evidence)
    assert calculation is not None
    draft = add_deterministic_derived_claims(
        DraftAnswer(answer_markdown="ignored", claims=source_claims()), [calculation]
    )
    derived = draft.claims[-1]
    assert derived.text == "Karnataka is higher than Odisha by 2.46 percentage points."
    assert derived.derivation is not None
    assert derived.derivation.input_claim_ids == ["karnataka", "odisha"]
    result = validate_and_materialize_citations(draft, evidence, [calculation])
    assert result.valid
    assert len(result.claims[-1].citation_ids) == 2

    altered = draft.model_copy(
        update={
            "claims": [
                *draft.claims[:-1],
                derived.model_copy(update={"value": 3.37, "unit": "percent"}),
            ]
        }
    )
    invalid = validate_and_materialize_citations(altered, evidence, [calculation])
    assert not invalid.valid
    assert "INVALID_DERIVATION" in invalid.error_codes
