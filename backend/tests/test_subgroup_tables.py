"""Regression: a population-subgroup table must never answer a whole-population question.

Live 2026-09-24, real Karnataka chunks: "Statement 17" is the Scheduled Tribes sex-ratio table,
but its title is a separate chunk, so every table fragment's breadcrumb said only
"Statement 17". The agent reported the ST value (990) as Karnataka's sex ratio (973).
"""

import json
from pathlib import Path

import pytest

from backend.app.agent.citations import validate_and_materialize_citations
from backend.app.agent.models import ArtifactDataRequirement, DraftAnswer, DraftClaim
from backend.app.agent.tools import _title_index, with_table_title
from backend.app.execution.contracts import ArtifactDatasetProposal, ArtifactRowProposal
from backend.app.execution.hydration import ProposalHydrationError, hydrate_artifact_dataset
from backend.app.retrieval.models import RetrievedEvidence

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "live_karnataka_sex_ratio_tables.json").read_text()
)


def evidence(key: str) -> RetrievedEvidence:
    return RetrievedEvidence.model_validate({**_FIXTURE[key], "retrieval_score": 1.0})


def titled(key: str) -> RetrievedEvidence:
    titles = _title_index(list(_FIXTURE.values()))
    return with_table_title(evidence(key), titles)


def test_statement_title_is_attached_to_its_table_fragments_only() -> None:
    table = titled("st_table")
    assert table.section_path[-1].startswith(
        "Sex Ratio (number of females per 1000 males) among Scheduled Tribes"
    )
    assert table.text == evidence("st_table").text
    persons = titled("all_persons_table")
    assert persons.section_path[-1] == (
        "Sex Ratio (number of females per 1000 males) by residence : 2001-2011"
    )
    # Heading-only title chunks are not themselves re-titled.
    assert titled("st_title").section_path == evidence("st_title").section_path


def claim(text: str, evidence_id: str, value: float) -> DraftClaim:
    return DraftClaim(
        claim_id="c1",
        text=text,
        evidence_ids=[evidence_id],
        metric="sex ratio",
        region="Karnataka",
        year=2011,
        residence_scope="total",
        value=value,
        unit="count",
    )


def test_subgroup_table_cannot_support_a_whole_population_claim() -> None:
    table = titled("st_table")
    draft = DraftAnswer(
        answer_markdown="x",
        claims=[claim("The sex ratio of Karnataka in 2011 was 990.", table.chunk_id, 990)],
    )
    result = validate_and_materialize_citations(draft, [table], [])
    assert not result.valid
    assert "SUBGROUP_TABLE_FOR_WHOLE_POPULATION_CLAIM" in result.error_codes

    scoped = DraftAnswer(
        answer_markdown="x",
        claims=[
            claim(
                "The sex ratio among Scheduled Tribes in Karnataka in 2011 was 990.",
                table.chunk_id,
                990,
            )
        ],
    )
    scoped_result = validate_and_materialize_citations(scoped, [table], [])
    assert "SUBGROUP_TABLE_FOR_WHOLE_POPULATION_CLAIM" not in scoped_result.error_codes


def requirement() -> ArtifactDataRequirement:
    return ArtifactDataRequirement(
        artifact_type="chart", metric="sex ratio", year=2011, regions=["Karnataka"], comparison=True
    )


def proposal(value: float, evidence_id: str) -> ArtifactDatasetProposal:
    return ArtifactDatasetProposal(
        title="Sex ratio",
        rows=[
            ArtifactRowProposal(
                label="KARNATAKA",
                unit="females per 1000 males",
                value=value,
                evidence_id=evidence_id,
                year=2011,
                residence_scope="Total",
                series="Total",
            )
        ],
    )


def test_hydration_refuses_subgroup_cells_and_binds_the_all_persons_row() -> None:
    st_table, persons = titled("st_table"), titled("all_persons_table")
    with pytest.raises(ProposalHydrationError, match="SUBGROUP_TABLE"):
        hydrate_artifact_dataset(proposal(990, st_table.chunk_id), requirement(), [st_table], "q")
    dataset = hydrate_artifact_dataset(
        proposal(973, persons.chunk_id), requirement(), [persons], "q"
    )
    assert dataset.source_records[0].raw_value == "973"
    assert dataset.source_records[0].page_number == persons.page_number
