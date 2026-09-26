import asyncio
import json
from pathlib import Path
from typing import Any, cast

from backend.app.agent.calculations import (
    add_deterministic_derived_claims,
    validated_calculation,
)
from backend.app.agent.citations import validate_and_materialize_citations
from backend.app.agent.evidence import unit_known
from backend.app.agent.graph import AgentGraph
from backend.app.agent.models import (
    CalculationRequest,
    DraftAnswer,
    DraftClaim,
    EvidenceAssessment,
    EvidenceAssessmentItem,
    TaskClassification,
)
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
        source_checksum="a" * 64,
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


_LIVE_CHUNKS = Path(__file__).parent / "fixtures" / "live_sex_ratio_comparison_chunks.json"
ODISHA, MP_TABLE, MP_PROSE, ODISHA_SECOND = (
    "7be0f079-a4ec-58bd-af08-db39c8bc075f",
    "3ba06715-cdf3-5901-b498-4df21a9a386c",
    "548fcc7d-2abc-5d2f-9c84-7c303c005ae8",
    "a09186ef-00dd-55a5-ba67-a569bb9dae55",
)


def live_chunks() -> list[RetrievedEvidence]:
    return [
        RetrievedEvidence.model_validate({**payload, "retrieval_score": 1.0})
        for payload in json.loads(_LIVE_CHUNKS.read_text(encoding="utf-8"))
    ]


def ratio_evidence(chunk_id: str, region: str, text: str) -> RetrievedEvidence:
    return rate_evidence(region, 0).model_copy(
        update={"chunk_id": chunk_id, "text": text, "citation_snippet": text}
    )


def test_count_comparison_never_says_percent_and_prunes_non_supporting_citations() -> None:
    """Regression from a live Odisha vs Madhya Pradesh sex-ratio comparison (2026-09-24).

    Real Qdrant chunks and the model's real draft claims. The draft was right (979 vs 931), but
    the Madhya Pradesh claim also cited a prose chunk the quote selector rejects, which
    invalidated the whole answer; the calculation cited a second Odisha chunk no claim cited;
    and the app-rendered derived claim said "by 48 percent" for a count difference.
    """
    evidence = live_chunks()
    claims = [
        DraftClaim(
            claim_id=ODISHA,
            text="The sex ratio in Odisha was 979 females per 1000 males in 2011.",
            evidence_ids=[ODISHA],
            metric="Sex Ratio",
            region="Odisha",
            year=2011,
            population_scope="Total",
            residence_scope="total",
            value=979.0,
            unit="count",
        ),
        DraftClaim(
            claim_id=ODISHA_SECOND,
            text="The sex ratio in Madhya Pradesh was 931 females per 1000 males in 2011.",
            evidence_ids=[MP_TABLE, MP_PROSE],
            metric="Sex Ratio",
            region="Madhya Pradesh",
            year=2011,
            population_scope="Total",
            residence_scope="total",
            value=931.0,
            unit="count",
        ),
    ]
    calculation = validated_calculation(
        "Compare the sex ratio of Odisha and Madhya Pradesh.",
        CalculationRequest(
            description="Compare the sex ratio of Odisha and Madhya Pradesh in 2011.",
            operation="difference",
            values=[979.0, 931.0],
            evidence_ids=[ODISHA, ODISHA_SECOND, MP_TABLE, MP_PROSE],
        ),
        evidence,
    )
    assert calculation is not None and calculation.unit == "count"
    draft = add_deterministic_derived_claims(
        DraftAnswer(answer_markdown="ignored", claims=claims), [calculation]
    )
    derived = draft.claims[-1]
    assert derived.text == "Odisha is higher than Madhya Pradesh by 48."

    result = validate_and_materialize_citations(draft, evidence, [calculation])
    assert result.valid, result.errors
    assert result.pruned_citations == [(ODISHA_SECOND, MP_PROSE)]
    assert {citation.chunk_id for citation in result.citations} == {ODISHA, MP_TABLE}
    assert set(result.claims[-1].citation_ids) == {
        *result.claims[0].citation_ids,
        *result.claims[1].citation_ids,
    }


def test_per_residence_comparisons_each_match_their_own_calculation() -> None:
    """Regression from a live cross-replica follow-up (2026-09-26): "How does that compare with
    Madhya Pradesh?" after Odisha's sex ratio.

    The model's draft was right: total, rural, and urban values for both states and three
    correct differences. All three calculations cite the same two table chunks, and the validator
    paired every derived claim with the first calculation whose evidence it cited, the total one.
    The rural and urban differences were rejected as altered, and the repair lost a target, so a
    correct comparison was refused.
    """
    evidence = live_chunks()
    values = {
        "Madhya Pradesh": {"total": 931.0, "rural": 936.0, "urban": 918.0},
        "Odisha": {"total": 979.0, "rural": 989.0, "urban": 932.0},
    }
    chunk = {"Madhya Pradesh": MP_TABLE, "Odisha": ODISHA_SECOND}
    claims = [
        DraftClaim(
            claim_id=f"{region}-{scope}",
            text=f"In 2011, the {scope} sex ratio in {region} was {value:g} per 1000 males.",
            evidence_ids=[chunk[region]],
            metric="Sex Ratio",
            region=region,
            year=2011,
            residence_scope=scope,
            value=value,
            unit="count",
        )
        for region, scopes in values.items()
        for scope, value in scopes.items()
    ]
    calculations = []
    for scope in ("total", "rural", "urban"):
        calculation = validated_calculation(
            "How does that compare with Madhya Pradesh?",
            CalculationRequest(
                description=f"Compare the {scope} sex ratio of Odisha and Madhya Pradesh.",
                operation="difference",
                values=[values["Odisha"][scope], values["Madhya Pradesh"][scope]],
                evidence_ids=[ODISHA_SECOND, MP_TABLE],
            ),
            evidence,
        )
        assert calculation is not None
        calculations.append(calculation)
    draft = add_deterministic_derived_claims(
        DraftAnswer(answer_markdown="ignored", claims=claims), calculations
    )
    derived = [claim for claim in draft.claims if claim.derivation is not None]
    assert [claim.text for claim in derived] == [
        "Odisha is higher than Madhya Pradesh by 48.",
        "In rural areas, Odisha is higher than Madhya Pradesh by 53.",
        "In urban areas, Odisha is higher than Madhya Pradesh by 14.",
    ]

    result = validate_and_materialize_citations(draft, evidence, calculations)
    assert result.valid, result.errors

    # Matching by operands never lets a derived claim through with a value no calculation has.
    forged = draft.model_copy(
        update={
            "claims": [
                *draft.claims[:-1],
                draft.claims[-1].model_copy(update={"value": 15.0}),
            ]
        }
    )
    invalid = validate_and_materialize_citations(forged, evidence, calculations)
    assert "INVALID_DERIVATION" in invalid.error_codes


def test_claim_with_no_supporting_citation_still_fails() -> None:
    evidence = [
        ratio_evidence("mp-prose", "Madhya Pradesh", "The sex ratio of the state has improved."),
    ]
    draft = DraftAnswer(
        answer_markdown="x",
        claims=[
            DraftClaim(
                claim_id="mp",
                text="The sex ratio in Madhya Pradesh was 931 in 2011.",
                evidence_ids=["mp-prose"],
                region="Madhya Pradesh",
                year=2011,
                value=931,
                unit="count",
            )
        ],
    )
    result = validate_and_materialize_citations(draft, evidence, [])
    assert not result.valid
    assert "CLAIM_QUOTE_NOT_FOUND" in result.error_codes
    assert result.pruned_citations == []


class UnitBlindAssessor:
    """Replays the live assessor verdict: right table, every flag true except has_unit."""

    async def assess_evidence(self, query: str, task_type: str, evidence: list[Any]) -> Any:
        del query, task_type
        return EvidenceAssessment(
            items=[
                EvidenceAssessmentItem(
                    evidence_id=item.chunk_id,
                    relevance="direct_answer",
                    entity_match=True,
                    metric_match=True,
                    year_match=True,
                    has_explicit_value=True,
                    has_unit=False,
                    reason="Madhya Pradesh sex ratio row",
                )
                for item in evidence
            ],
            selected_evidence_ids=[item.chunk_id for item in evidence],
            sufficient=False,
            explanation="Direct value present; unit not stated in the cell.",
        )


def _lookup_state(evidence: list[RetrievedEvidence]) -> dict[str, Any]:
    return {
        "resolved_query": "What was the sex ratio in Madhya Pradesh in 2011?",
        "task_type": "lookup",
        "classification": TaskClassification(
            task_type="lookup", regions=["Madhya Pradesh"], reason="lookup"
        ),
        "artifact_requirement": None,
        "retrieved_evidence": evidence,
        "available_limitations": [],
        "trace_events": [],
    }


def test_unit_known_in_table_heading_satisfies_has_unit() -> None:
    # Live 2026-09-24: "What was the sex ratio in Madhya Pradesh in 2011?" was declined because
    # the assessor marked the real row has_unit=false. The chunk's breadcrumb is mis-attributed
    # and never says "per 1000 males"; its "Sex Ratio 2011" heading fixes the unit by definition.
    mp_table = next(item for item in live_chunks() if item.chunk_id == MP_TABLE)
    assert unit_known(mp_table.text)
    graph = AgentGraph(UnitBlindAssessor(), object())  # type: ignore[arg-type]
    result = asyncio.run(graph.assess_evidence(_lookup_state([mp_table])))  # type: ignore[arg-type]
    assert result["evidence_sufficient"] is True
    selected = cast(list[RetrievedEvidence], result["selected_evidence"])
    assert [item.chunk_id for item in selected] == [MP_TABLE]


def test_missing_unit_still_blocks_when_the_chunk_states_none() -> None:
    bare = live_chunks()[0].model_copy(
        update={
            "chunk_id": "bare",
            "text": "| MADHYA PRADESH | 931 |",
            "citation_snippet": "| MADHYA PRADESH | 931 |",
        }
    )
    assert not unit_known(bare.text)
    graph = AgentGraph(UnitBlindAssessor(), object())  # type: ignore[arg-type]
    result = asyncio.run(graph.assess_evidence(_lookup_state([bare])))  # type: ignore[arg-type]
    assert result["evidence_sufficient"] is False
