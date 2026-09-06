import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from backend.app.agent.errors import AgentOperationalError
from backend.app.agent.evidence import pack_targeted_assessment_evidence
from backend.app.agent.graph import (
    AgentGraph,
    build_artifact_requirement,
    format_artifact_target_query,
)
from backend.app.agent.models import (
    AgentPlan,
    ArtifactDataRequirement,
    EvidenceAssessment,
    EvidenceAssessmentItem,
    TaskClassification,
    ToolCallRecord,
)
from backend.app.agent.skills import SkillRegistry
from backend.app.execution.contracts import ArtifactDataset, SourceRecord
from backend.app.execution.lineage import validate_dataset
from backend.app.retrieval.models import RetrievedEvidence
from backend.tests.test_agent_service import FakeTools


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def requirement() -> ArtifactDataRequirement:
    return ArtifactDataRequirement(
        artifact_type="chart",
        metric="literacy rate",
        year=2011,
        regions=["Karnataka", "Odisha"],
        population_scope="persons",
        residence_scope="total",
        comparison=True,
    )


def candidate(
    region: str,
    page: int,
    chunk_id: str,
    kind: str,
    score: float,
) -> RetrievedEvidence:
    if kind == "direct":
        value = "71.00" if region == "Karnataka" else "72.00"
        text = (
            "| Region | Year | Residence | Population | Metric | Value | Unit |\n"
            "|---|---:|---|---|---|---:|---|\n"
            f"| {region} | 2011 | Total | Persons | Literacy rate | {value} | percent |"
        )
    elif kind in {"map", "graph"}:
        text = f"{kind.title()} description for {region} with district labels. " + "x" * 1300
    elif kind == "contents":
        text = "Table of contents with page references 1 2 3. " + "x" * 1300
    elif kind == "definition":
        text = (
            "Literacy rate is a percentage definition for population age 7 and above. " + "x" * 1300
        )
    else:
        text = "Related district material with value 60.00 percent. " + "x" * 1300
    return RetrievedEvidence(
        chunk_id=chunk_id,
        text=text,
        document_title=f"Synthetic {region} report",
        document_id=f"fixture-{region.casefold().replace(' ', '-')}",
        region=region,
        page_number=page,
        citation_snippet=text[:80],
        section_path=["Literacy"],
        extraction_method="provided_markdown",
        coverage_status="indexed_provided_markdown",
        source_checksum=("a" if region == "Karnataka" else "b") * 64,
        retrieval_score=score,
    )


def failed_trace_fixture() -> list[RetrievedEvidence]:
    path = Path(__file__).parent / "fixtures" / "prompt5_failed_candidates.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        candidate(row["region"], row["page"], row["id"], row["kind"], row["score"]) for row in rows
    ]


class AssessBothModel:
    def __init__(self) -> None:
        self.assessment_ids: list[str] = []

    async def assess_evidence(
        self, query: str, task_type: str, evidence: list[RetrievedEvidence]
    ) -> EvidenceAssessment:
        del query, task_type
        self.assessment_ids = [item.chunk_id for item in evidence]
        items = []
        for item in evidence:
            direct = "state-table" in item.chunk_id or item.chunk_id == "ka-table"
            items.append(
                EvidenceAssessmentItem(
                    evidence_id=item.chunk_id,
                    relevance="direct_answer" if direct else "related_non_answering",
                    entity_match=direct,
                    metric_match=direct,
                    year_match=direct,
                    has_explicit_value=direct,
                    has_unit=direct,
                    population_scope_match=direct,
                    residence_scope_match=direct,
                    unit_compatible=direct,
                    reason="Synthetic direct row" if direct else "Not direct evidence",
                )
            )
        selected = [item.evidence_id for item in items if item.relevance == "direct_answer"]
        return EvidenceAssessment(
            items=items,
            selected_evidence_ids=selected,
            sufficient={item.region for item in evidence if item.chunk_id in selected}
            == {"Karnataka", "Odisha"},
            explanation="Synthetic comparison assessment",
        )


class ArtifactClassifier:
    async def classify(self, query: str, context: list[Any]) -> TaskClassification:
        del query, context
        return TaskClassification(
            task_type="artifact_chart",
            regions=requirement().regions,
            artifact_requirement=requirement(),
            reason="Chart requested",
        )


class RecordingTools(FakeTools):
    def __init__(self, skills: SkillRegistry) -> None:
        super().__init__(skills)
        self.requests: list[Any] = []
        self.qdrant_writes = 0

    async def search_documents(self, value: Any) -> list[RetrievedEvidence]:
        self.requests.append(value)
        assert value.regions is not None and len(value.regions) == 1
        region = value.regions[0]
        return [item for item in failed_trace_fixture() if item.region == region]


def test_chart_intent_has_independent_typed_data_requirement() -> None:
    classification = TaskClassification(
        task_type="artifact_chart",
        regions=["Karnataka", "Odisha"],
        artifact_requirement=requirement(),
        reason="Chart requested",
    )
    result = build_artifact_requirement(
        classification,
        "Create a bar chart comparing the 2011 literacy rates",
        "artifact_chart",
    )
    assert classification.task_type == "artifact_chart"
    assert result == requirement()
    assert result is not None and result.metric == "literacy rate"

    graph = AgentGraph(ArtifactClassifier(), object())  # type: ignore[arg-type]
    classified = run(
        graph.classify_task(
            {
                "user_query": "Create a comparison chart",
                "messages": [],
                "trace_events": [],
            }
        )
    )
    assert classified["task_type"] == "artifact_chart"


def test_artifact_query_formatting_never_joins_conjunction() -> None:
    queries = [
        format_artifact_target_query(requirement(), target) for target in requirement().regions
    ]
    assert queries == [
        "2011 total persons literacy rate Karnataka",
        "2011 total persons literacy rate Odisha",
    ]
    assert all("Karnatakaand" not in query for query in queries)


def test_comparison_artifact_searches_targets_independently_with_filters(tmp_path: Path) -> None:
    tools = RecordingTools(SkillRegistry(tmp_path / "skills"))
    graph = AgentGraph(AssessBothModel(), tools, assessment_max_characters=12_000)  # type: ignore[arg-type]
    state = {
        "resolved_query": "2011 total persons literacy rates for Karnataka and Odisha",
        "task_type": "artifact_chart",
        "classification": TaskClassification(
            task_type="artifact_chart",
            regions=["Karnataka", "Odisha"],
            artifact_requirement=requirement(),
            reason="Chart",
        ),
        "artifact_requirement": requirement(),
        "plan": AgentPlan(steps=["retrieve"], retrieval_top_k=10),
        "tool_calls": [
            ToolCallRecord(tool_name="list_skills", status="ok", arguments={}),
            ToolCallRecord(tool_name="read_skill", status="ok", arguments={}),
        ],
        "trace_events": [],
        "errors": [],
    }
    result = run(graph.call_tools(state))  # type: ignore[arg-type]
    assert [request.regions for request in tools.requests] == [["Karnataka"], ["Odisha"]]
    assert [request.query for request in tools.requests] == [
        "2011 total persons literacy rate Karnataka",
        "2011 total persons literacy rate Odisha",
    ]
    assert len(result["tool_calls"]) == 6
    assert tools.qdrant_writes == 0


def test_trace_shape_reserves_rank_ten_odisha_direct_evidence() -> None:
    evidence = failed_trace_fixture()
    packed = pack_targeted_assessment_evidence(
        evidence,
        required_targets=["Karnataka", "Odisha"],
        max_characters=12_000,
        max_chunks=12,
    )
    assert len(evidence) == 10
    assert packed.reserved_by_target == {
        "Karnataka": "ka-table",
        "Odisha": "od-state-table",
    }
    assert "od-state-table" in {item.chunk_id for item in packed.included}
    assert sum(len(item.text) for item in packed.included) <= 12_000


def test_artifact_assessment_requires_direct_evidence_for_both_targets() -> None:
    model = AssessBothModel()
    graph = AgentGraph(model, object(), assessment_max_characters=12_000)  # type: ignore[arg-type]
    state = {
        "task_type": "artifact_chart",
        "resolved_query": "synthetic comparison",
        "classification": TaskClassification(
            task_type="artifact_chart", regions=requirement().regions, reason="Chart"
        ),
        "artifact_requirement": requirement(),
        "retrieved_evidence": failed_trace_fixture(),
        "available_limitations": ["Some unrelated visual pages were excluded."],
        "trace_events": [],
    }
    result = run(graph.assess_evidence(state))  # type: ignore[arg-type]
    assert result["evidence_sufficient"] is True
    assert {item.region for item in result["selected_evidence"]} == {"Karnataka", "Odisha"}
    assert result["limitations"] == []
    assert "od-state-table" in model.assessment_ids

    state["retrieved_evidence"] = [
        item for item in failed_trace_fixture() if item.region == "Karnataka"
    ]
    result = run(graph.assess_evidence(state))  # type: ignore[arg-type]
    assert result["evidence_sufficient"] is False


def test_required_target_that_cannot_fit_returns_typed_budget_error() -> None:
    graph = AgentGraph(AssessBothModel(), object(), assessment_max_characters=100)  # type: ignore[arg-type]
    state = {
        "task_type": "artifact_chart",
        "resolved_query": "synthetic comparison",
        "classification": TaskClassification(
            task_type="artifact_chart", regions=requirement().regions, reason="Chart"
        ),
        "artifact_requirement": requirement(),
        "retrieved_evidence": failed_trace_fixture(),
        "trace_events": [],
    }
    with pytest.raises(AgentOperationalError) as caught:
        run(graph.assess_evidence(state))  # type: ignore[arg-type]
    assert caught.value.code == "EVIDENCE_BUDGET_INSUFFICIENT"


def test_artifact_dataset_retains_complete_target_provenance() -> None:
    direct = [item for item in failed_trace_fixture() if "| Region |" in item.text]
    records = []
    rows = []
    for index, item in enumerate(direct, 1):
        raw = "71.00" if item.region == "Karnataka" else "72.00"
        records.append(
            SourceRecord(
                source_record_id=f"source-{index}",
                row_id=item.region.casefold(),
                field="literacy_rate",
                raw_value=raw,
                normalized_numeric_value=float(raw),
                unit="percent",
                metric="literacy rate",
                region=item.region,
                year=2011,
                population_scope="persons",
                residence_scope="total",
                document_title=item.document_title,
                document_id=item.document_id,
                page_number=item.page_number,
                chunk_id=item.chunk_id,
                exact_supporting_quote=item.text,
                source_checksum=item.source_checksum,
            )
        )
        rows.append({"row_id": item.region.casefold(), "literacy_rate": float(raw)})
    dataset = ArtifactDataset(
        title="Synthetic literacy comparison",
        task_type="artifact_chart",
        rows=rows,
        columns=["row_id", "literacy_rate"],
        units={"literacy_rate": "percent"},
        source_records=records,
        requested_output="bar chart",
    )
    validate_dataset(dataset, direct, requirement())
    assert {record.region for record in dataset.source_records} == {"Karnataka", "Odisha"}
