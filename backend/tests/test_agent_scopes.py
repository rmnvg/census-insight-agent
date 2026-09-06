import pytest

from backend.app.agent.calculations import enrich_source_claims
from backend.app.agent.graph import build_artifact_requirement
from backend.app.agent.models import (
    ArtifactDataRequirement,
    DraftAnswer,
    DraftClaim,
    TaskClassification,
)
from backend.app.agent.scopes import canonical_scopes, canonicalize_artifact_requirement
from backend.tests.test_artifact_proposals import table_evidence


@pytest.mark.parametrize(
    ("query", "population", "residence"),
    [
        ("total persons literacy rate", "persons", "total"),
        ("rural persons literacy rate", "persons", "rural"),
        ("urban persons literacy rate", "persons", "urban"),
        ("female rural literacy rate", "female", "rural"),
        ("male urban literacy rate", "male", "urban"),
    ],
)
def test_canonical_scope_semantics(query: str, population: str, residence: str) -> None:
    assert canonical_scopes(query, None, None) == (population, residence)


def test_artifact_requirement_infers_total_residence_from_total_persons() -> None:
    classified = ArtifactDataRequirement(
        artifact_type="table",
        metric="Literacy Rate",
        year=2011,
        regions=["North", "South"],
        population_scope="Total Persons",
        residence_scope=None,
        comparison=True,
    )
    result = build_artifact_requirement(
        TaskClassification(
            task_type="artifact_chart",
            regions=["North", "South"],
            artifact_requirement=classified,
            reason="chart",
        ),
        "Create a bar chart comparing 2011 total persons literacy rates",
        "artifact_chart",
    )
    assert result is not None
    assert result.artifact_type == "chart"
    assert result.population_scope == "persons"
    assert result.residence_scope == "total"


@pytest.mark.parametrize(
    "metric", ["literacy rate", "Literacy Rate", "literacy rates", "Effective Literacy Rates"]
)
def test_artifact_requirement_canonicalizes_literacy_metric(metric: str) -> None:
    requirement = ArtifactDataRequirement(
        artifact_type="chart",
        metric=metric,
        year=2011,
        regions=["Karnataka", "Odisha"],
        population_scope="persons",
        residence_scope="total",
        comparison=True,
    )
    result = canonicalize_artifact_requirement(requirement, "2011 literacy rates")
    assert result.metric == "literacy rate"


def test_prompt4_claim_enrichment_uses_the_same_scope_semantics() -> None:
    trusted = table_evidence("North", "71.50", "north-evidence")
    draft = DraftAnswer(
        answer_markdown="North was 71.5%.",
        claims=[
            DraftClaim(
                claim_id="claim-1",
                text="The 2011 total persons literacy rate for North was 71.5%.",
                evidence_ids=[trusted.chunk_id],
                value=71.5,
                unit="percent",
            )
        ],
    )
    result = enrich_source_claims(draft, [trusted], "Compare the 2011 total persons literacy rates")
    assert result.claims[0].population_scope == "persons"
    assert result.claims[0].residence_scope == "total"
