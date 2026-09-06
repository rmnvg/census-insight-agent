import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from backend.app.agent.errors import AgentOperationalError
from backend.app.agent.graph import AgentGraph
from backend.app.agent.models import (
    AgentPlan,
    AgentState,
    ArtifactDataRequirement,
    TaskClassification,
)
from backend.app.agent.skills import SkillRegistry
from backend.app.agent.tools import AgentTools
from backend.app.execution.contracts import (
    ArtifactDataset,
    ArtifactDatasetProposal,
    ArtifactDescriptor,
    ArtifactRowProposal,
    SourceRecord,
)
from backend.app.execution.hydration import count_table_entity_rows
from backend.app.execution.presentation import artifact_response
from backend.app.retrieval.models import RetrievedEvidence
from backend.tests.test_agent_service import FakeTools


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


# Mirrors the real "Statement 6: Sex Ratio ... by residence" table structure in
# data/source/markdown/PCA Data Highlights MP.md: a two-row forward-filled header
# (Sex Ratio 2011 / Total|Rural|Urban), a state aggregate row, then district rows.
DISTRICT_TABLE_TEXT = (
    "Statement 6: Sex Ratio (number of females per 1000 males) by residence: 2001-2011\n\n"
    "| State/District Code | State/District | Sex Ratio 2011 | | |\n"
    "|---|---|---|---|---|\n"
    "| | | Total | Rural | Urban |\n"
    "| 23 | MADHYA PRADESH | 800 | 805 | 795 |\n"
    "| 418 | Sheopur | 908 | 901 | 903 |\n"
    "| 419 | Balaghat | 1021 | 1024 | 1000 |\n"
    "| 420 | Bhind | 828 | 828 | 828 |\n"
)


def district_evidence(chunk_id: str = "chunk-sex-ratio-1") -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk_id=chunk_id,
        text=DISTRICT_TABLE_TEXT,
        document_title="Madhya Pradesh PCA Highlights",
        document_id="doc-mp",
        region="Madhya Pradesh",
        page_number=32,
        citation_snippet=DISTRICT_TABLE_TEXT,
        section_path=["Statement 6", "Sex Ratio"],
        extraction_method="provided_markdown",
        coverage_status="indexed_provided_markdown",
        source_checksum="a" * 64,
        retrieval_score=1,
    )


def ranking_requirement(**overrides: Any) -> ArtifactDataRequirement:
    base = ArtifactDataRequirement(
        artifact_type="table",
        metric="sex ratio",
        year=2011,
        regions=[],
        population_scope=None,
        residence_scope="total",
        rank_all=True,
        rank_direction="max",
    )
    return base.model_copy(update=overrides)


def proposed_row(
    label: str, value: float, evidence_id: str = "chunk-sex-ratio-1"
) -> ArtifactRowProposal:
    return ArtifactRowProposal(
        label=label,
        value=value,
        unit="ratio",
        evidence_id=evidence_id,
        year=2011,
        population_scope=None,
        residence_scope="total",
    )


class RankingModel:
    """Minimal AgentModel stub: prepare_artifact only calls propose_artifact_dataset."""

    def __init__(self, rows: list[ArtifactRowProposal]) -> None:
        self.rows = rows

    async def propose_artifact_dataset(
        self, query: str, task_type: str, values: list[Any], *, rank_all: bool = False
    ) -> ArtifactDatasetProposal:
        del query, task_type, values, rank_all
        return ArtifactDatasetProposal(title="Sex ratio by district", rows=self.rows)


def base_state(req: ArtifactDataRequirement) -> dict[str, Any]:
    return {
        "session_id": "11111111-1111-4111-8111-111111111111",
        "run_id": "22222222-2222-4222-8222-222222222222",
        "resolved_query": "Which district had the highest sex ratio in Madhya Pradesh?",
        "task_type": "artifact_table",
        "classification": TaskClassification(task_type="artifact_table", reason="ranking"),
        "selected_evidence": [district_evidence()],
        "artifact_requirement": req,
        "selected_skill": "table",
        "skill_instructions": "Preserve citations and values.",
        "trace_events": [],
        "artifact_attempt": 0,
        "artifact_errors": [],
        "plan": AgentPlan(steps=["artifact"]),
    }


def make_graph(model: RankingModel, tmp_path: Path) -> AgentGraph:
    return AgentGraph(model, FakeTools(SkillRegistry(tmp_path / "skills")))  # type: ignore[arg-type]


def test_prepare_artifact_selects_highest_district_row(tmp_path: Path) -> None:
    model = RankingModel(
        [
            proposed_row("Sheopur", 908),
            proposed_row("Balaghat", 1021),
            proposed_row("Bhind", 828),
        ]
    )
    value = make_graph(model, tmp_path)
    state = base_state(ranking_requirement())
    result = run(value.prepare_artifact(cast(AgentState, state)))
    winner = result["ranking_winner"]
    assert isinstance(winner, SourceRecord)
    assert winner.region == "Balaghat"
    assert winner.normalized_numeric_value == 1021
    dataset = result["artifact_dataset"]
    assert isinstance(dataset, ArtifactDataset)
    assert dataset.rows[0]["label"] == "Balaghat"


def test_prepare_artifact_selects_lowest_row_when_direction_is_min(tmp_path: Path) -> None:
    model = RankingModel(
        [
            proposed_row("Sheopur", 908),
            proposed_row("Balaghat", 1021),
            proposed_row("Bhind", 828),
        ]
    )
    value = make_graph(model, tmp_path)
    state = base_state(ranking_requirement(rank_direction="min"))
    result = run(value.prepare_artifact(cast(AgentState, state)))
    winner = result["ranking_winner"]
    assert isinstance(winner, SourceRecord)
    assert winner.region == "Bhind"
    assert winner.normalized_numeric_value == 828


def test_prepare_artifact_rejects_incomplete_row_coverage(tmp_path: Path) -> None:
    model = RankingModel(
        [
            proposed_row("Sheopur", 908),
            proposed_row("Balaghat", 1021),
        ]
    )
    value = make_graph(model, tmp_path)
    state = base_state(ranking_requirement())
    with pytest.raises(AgentOperationalError) as excinfo:
        run(value.prepare_artifact(cast(AgentState, state)))
    assert excinfo.value.diagnostics["reason_code"] == "RANKING_COVERAGE_INCOMPLETE"


def test_prepare_artifact_rejects_state_aggregate_winner(tmp_path: Path) -> None:
    model = RankingModel(
        [
            proposed_row("Madhya Pradesh", 800),
            proposed_row("Sheopur", 908),
            proposed_row("Balaghat", 1021),
            proposed_row("Bhind", 828),
        ]
    )
    value = make_graph(model, tmp_path)
    state = base_state(ranking_requirement(rank_direction="min"))
    with pytest.raises(AgentOperationalError) as excinfo:
        run(value.prepare_artifact(cast(AgentState, state)))
    assert excinfo.value.diagnostics["reason_code"] == "RANKING_INCLUDES_AGGREGATE_ROW"


def test_count_table_entity_rows_excludes_headers_and_state_total() -> None:
    assert count_table_entity_rows([district_evidence()]) == 3


def test_artifact_response_prepends_ranking_sentence_with_citation() -> None:
    source = SourceRecord(
        source_record_id="source-1",
        row_id="row-1",
        field="value",
        raw_value="1021",
        normalized_numeric_value=1021,
        unit="",
        metric="sex ratio",
        region="Balaghat",
        year=2011,
        population_scope="persons",
        residence_scope="total",
        document_title="Madhya Pradesh PCA Highlights",
        document_id="doc-mp",
        page_number=32,
        chunk_id="chunk-sex-ratio-1",
        exact_supporting_quote="1021",
        source_checksum="a" * 64,
    )
    dataset = ArtifactDataset(
        title="Sex ratio by district",
        task_type="artifact_table",
        rows=[{"row_id": "row-1", "label": "Balaghat", "series": "", "value": 1021}],
        columns=["row_id", "label", "series", "value"],
        units={"value": ""},
        source_records=[source],
        requested_output="Which district had the highest sex ratio?",
    )
    descriptor = ArtifactDescriptor(
        artifact_id="33333333-3333-4333-8333-333333333333",
        artifact_type="table",
        title="Sex ratio by district",
        filename="table.md",
        media_type="text/markdown",
        byte_size=10,
        sha256="b" * 64,
        session_id="11111111-1111-4111-8111-111111111111",
        run_id="22222222-2222-4222-8222-222222222222",
        source_manifest_path="/tmp/source-manifest.json",
        download_url="/artifacts/table.md",
    )
    response = artifact_response(
        dataset,
        descriptor,
        [district_evidence()],
        "run-1",
        ranking_winner=source,
        rank_direction="max",
    )
    assert response.answer_markdown.startswith(
        "Balaghat recorded the highest sex ratio (1021) among Sex ratio by district."
    )
    assert any(citation.snippet == "1021" for citation in response.citations)


class ScrollPoint:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload


class ScrollClient:
    def __init__(self, points: list[ScrollPoint]) -> None:
        self.points = points

    def scroll(self, collection_name: str, **kwargs: Any) -> tuple[list[ScrollPoint], None]:
        del collection_name, kwargs
        return self.points, None


class ScrollStore:
    def __init__(self, points: list[ScrollPoint]) -> None:
        self.client = ScrollClient(points)
        self.collection_name = "census"


def _payload(evidence_item: RetrievedEvidence) -> dict[str, Any]:
    return evidence_item.model_dump(mode="json")


def test_collect_metric_table_rows_filters_by_metric_keywords() -> None:
    matching = district_evidence()
    non_matching = district_evidence(chunk_id="chunk-literacy-1").model_copy(
        update={
            "text": "Literacy rate by residence: 2011",
            "section_path": ["Statement 11", "Literacy"],
        }
    )
    store = ScrollStore([ScrollPoint(_payload(matching)), ScrollPoint(_payload(non_matching))])
    tools = AgentTools(
        retrieval=cast(Any, None),
        store=cast(Any, store),
        skills=cast(Any, None),
        data_root=Path("."),
    )
    result = run(tools.collect_metric_table_rows("doc-mp", "sex ratio"))
    assert [item.chunk_id for item in result] == ["chunk-sex-ratio-1"]
