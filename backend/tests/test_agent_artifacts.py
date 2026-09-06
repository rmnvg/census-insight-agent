import asyncio
from pathlib import Path
from typing import Any, cast

from backend.app.agent.graph import AgentGraph
from backend.app.agent.models import (
    AgentPlan,
    AgentState,
    ArtifactDataRequirement,
    TaskClassification,
)
from backend.app.agent.skills import SkillRegistry
from backend.app.execution.client import ArtifactStore, ExecutionQueueClient
from backend.app.execution.contracts import (
    ArtifactDatasetProposal,
    ArtifactRowProposal,
    GeneratedProgram,
)
from backend.tests.test_agent_service import FakeTools
from backend.tests.test_executor import evidence
from executor.worker import Worker


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


TABLE_PROGRAM = """
import json
from pathlib import Path
payload = json.loads(Path("input.json").read_text(encoding="utf-8"))
columns = payload["dataset"]["columns"]
rows = payload["dataset"]["rows"]
csv_text = ",".join(columns) + "\\n"
csv_text += "\\n".join(",".join(str(row[column]) for column in columns) for row in rows) + "\\n"
Path("output/table.csv").write_text(csv_text, encoding="utf-8")
markdown = "| " + " | ".join(columns) + " |\\n"
markdown += "|" + "|".join("---" for column in columns) + "|\\n"
markdown += "\\n".join(
    "| " + " | ".join(str(row[column]) for column in columns) + " |"
    for row in rows
) + "\\n"
Path("output/table.md").write_text(markdown, encoding="utf-8")
manifest = json.dumps(payload["source_manifest"])
Path("output/source-manifest.json").write_text(manifest, encoding="utf-8")
""".strip()


class ArtifactModel:
    def __init__(self, code: str = TABLE_PROGRAM, repaired: str = TABLE_PROGRAM) -> None:
        self.code = code
        self.repaired = repaired
        self.repairs = 0

    async def propose_artifact_dataset(
        self, query: str, task_type: str, values: list[Any], *, rank_all: bool = False
    ) -> ArtifactDatasetProposal:
        del query, task_type, values, rank_all
        return ArtifactDatasetProposal(
            title="Literacy",
            chart_kind=None,
            x_label="Region",
            y_label="Literacy rate (percent)",
            rows=[
                ArtifactRowProposal(
                    label="Karnataka",
                    value=75.36,
                    unit="percent",
                    evidence_id="chunk-karnataka",
                    year=2011,
                    population_scope="persons",
                    residence_scope="total",
                )
            ],
        )

    async def generate_artifact_code(self, value: Any, skill: str) -> GeneratedProgram:
        del value, skill
        return GeneratedProgram(code=self.code)

    async def repair_artifact_code(
        self,
        value: Any,
        code: str,
        skill: str,
        error_code: str,
        stderr: str,
    ) -> GeneratedProgram:
        del value, code, skill, error_code, stderr
        self.repairs += 1
        return GeneratedProgram(code=self.repaired)


class ImmediateQueue(ExecutionQueueClient):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.submissions = 0

    def submit(self, request: Any) -> Path:
        self.submissions += 1
        path = super().submit(request)
        Worker(self.root).run_once()
        return path


def base_state(tmp_path: Path) -> dict[str, Any]:
    text = (
        "Literacy Rate by residence: 2011 (Persons)\n\n"
        "| Region | Literacy Rate | | |\n"
        "|---|---|---|---|\n"
        "| | 2011 | | |\n"
        "| | Total | Rural | Urban |\n"
        "| Karnataka | 75.36 | 68.73 | 85.78 |"
    )
    trusted = evidence().model_copy(
        update={
            "text": text,
            "citation_snippet": text,
            "region": "Karnataka",
            "section_path": ["Literacy Rate by residence", "2011", "Persons"],
        }
    )
    requirement = ArtifactDataRequirement(
        artifact_type="table",
        metric="literacy rate",
        year=2011,
        regions=["Karnataka"],
        population_scope="persons",
        residence_scope="total",
    )
    return {
        "session_id": "11111111-1111-4111-8111-111111111111",
        "run_id": "22222222-2222-4222-8222-222222222222",
        "resolved_query": "Create a literacy table",
        "task_type": "artifact_table",
        "classification": TaskClassification(task_type="artifact_table", reason="table"),
        "selected_evidence": [trusted],
        "artifact_requirement": requirement,
        "selected_skill": "table",
        "skill_instructions": "Preserve citations and values.",
        "trace_events": [],
        "artifact_attempt": 0,
        "artifact_errors": [],
        "plan": AgentPlan(steps=["artifact"]),
    }


def graph(tmp_path: Path, model: ArtifactModel) -> AgentGraph:
    queue_root = tmp_path / "workspace" / "execution-queue"
    return AgentGraph(
        model,  # type: ignore[arg-type]
        FakeTools(SkillRegistry(tmp_path / "skills")),
        execution_queue=ImmediateQueue(queue_root),
        artifact_store=ArtifactStore(tmp_path / "workspace", queue_root),
        execution_timeout_seconds=2,
    )


def test_chart_and_table_skills_are_selected(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    skills.mkdir()
    for name in ("chart", "table"):
        (skills / f"{name}.md").write_text(
            f"---\nname: {name}\ndescription: {name}\ntask_types: [artifact_{name}]\n---\n"
            "Use citation-safe data.\n",
            encoding="utf-8",
        )
    value = AgentGraph(ArtifactModel(), FakeTools(SkillRegistry(skills)))  # type: ignore[arg-type]
    for task in ("artifact_chart", "artifact_table"):
        result = run(value.load_skill({"task_type": task, "tool_calls": [], "trace_events": []}))
        assert result["selected_skill"] == task.removeprefix("artifact_")


def test_evidence_refusal_prevents_artifact_execution() -> None:
    assert (
        AgentGraph._evidence_route({"task_type": "artifact_chart", "evidence_sufficient": False})
        == "stop"
    )


def test_table_artifact_executes_and_persists_with_citations(tmp_path: Path) -> None:
    value = graph(tmp_path, ArtifactModel())
    state = base_state(tmp_path)
    state.update(run(value.prepare_artifact(cast(AgentState, state))))
    state.update(run(value.generate_artifact_code(cast(AgentState, state))))
    state.update(run(value.execute_artifact(cast(AgentState, state))))
    state.update(run(value.inspect_artifact(cast(AgentState, state))))
    response = state["final_response"]
    assert response.artifacts
    assert response.citations
    assert all(claim.citation_ids for claim in response.claims)
    assert isinstance(value.execution_queue, ImmediateQueue)
    assert value.execution_queue.submissions == 1
    events = {event.event for event in state["trace_events"]}
    assert {"executor_submission", "executor_result"} <= events
    assert AgentGraph._artifact_result_route(cast(AgentState, state)) == "done"


def test_one_runtime_repair_succeeds_and_second_failure_stops(tmp_path: Path) -> None:
    model = ArtifactModel(code="raise RuntimeError('fix me')")
    value = graph(tmp_path, model)
    state = base_state(tmp_path)
    state.update(run(value.prepare_artifact(cast(AgentState, state))))
    state.update(run(value.generate_artifact_code(cast(AgentState, state))))
    state.update(run(value.execute_artifact(cast(AgentState, state))))
    state.update(run(value.inspect_artifact(cast(AgentState, state))))
    assert AgentGraph._artifact_result_route(cast(AgentState, state)) == "repair"
    state.update(run(value.repair_artifact(cast(AgentState, state))))
    state.update(run(value.execute_artifact(cast(AgentState, state))))
    state.update(run(value.inspect_artifact(cast(AgentState, state))))
    assert model.repairs == 1
    assert state["final_response"].artifacts

    terminal = {"artifact_attempt": 2, "artifact_errors": ["EXECUTION_FAILED"]}
    assert AgentGraph._artifact_result_route(cast(AgentState, terminal)) == "stop"


def test_policy_violation_is_never_repaired() -> None:
    state = {"artifact_attempt": 1, "artifact_errors": ["CODE_POLICY_VIOLATION"]}
    assert AgentGraph._artifact_result_route(cast(AgentState, state)) == "stop"
