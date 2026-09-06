import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from backend.app.agent.graph import AgentGraph
from backend.app.agent.models import AgentState, DraftAnswer, DraftClaim, SupportAssessment
from backend.app.agent.quotes import select_evidence_span
from backend.app.config import Settings
from backend.app.main import app
from backend.tests.test_agent_quotes import evidence
from backend.tests.test_executor import request
from executor.runner import STREAM_LIMIT, execute_request


@pytest.mark.parametrize(
    ("value", "year", "population", "supported"),
    [
        (75.36, 2011, "Persons", True),
        (82.47, 2011, "Persons", False),
        (82.47, 2011, "Male", True),
        (75.36, 2001, "Persons", False),
    ],
)
def test_table_values_are_bound_to_their_column(
    value: float, year: int, population: str, supported: bool
) -> None:
    item = evidence(
        "| Year | Persons literacy rate | Male literacy rate |\n"
        "|---|---|---|\n| 2011 | 75.36 | 82.47 |\n"
    )
    claim = DraftClaim(
        claim_id="c1",
        text=f"Karnataka {population} literacy was {value}% in {year}.",
        evidence_ids=[item.chunk_id],
        metric="literacy rate",
        region="Karnataka",
        year=year,
        population_scope=population,
        value=value,
        unit="percent",
    )
    assert (select_evidence_span(claim, item) is not None) == supported


@pytest.mark.parametrize("approved,rejected", [([], []), (["other"], []), (["c1"], ["c1"])])
def test_incomplete_semantic_assessment_cannot_publish(
    approved: list[str], rejected: list[str]
) -> None:
    item = evidence("Karnataka literacy was 75.36 percent in 2011.")

    class Assessor:
        async def assess_support(self, *_: object) -> SupportAssessment:
            return SupportAssessment(
                supported_claim_ids=approved, unsupported_claim_ids=rejected, explanation="probe"
            )

    graph = object.__new__(AgentGraph)
    graph.model = cast(Any, Assessor())
    state = cast(
        AgentState,
        {
            "task_type": "lookup",
            "run_id": "probe",
            "evidence_sufficient": True,
            "selected_evidence": [item],
            "draft_answer": DraftAnswer(
                answer_markdown="Finding",
                claims=[DraftClaim(claim_id="c1", text=item.text, evidence_ids=[item.chunk_id])],
            ),
        },
    )
    result = asyncio.run(graph.validate_citations(state))
    assert "final_response" not in result
    assert "INCOMPLETE_SUPPORT_ASSESSMENT" in cast(list[str], result["validation_error_codes"])


def test_unbounded_stdout_stops_before_the_execution_deadline(tmp_path: Path) -> None:
    result = execute_request(request("while True:\n    print('x' * 4096)"), tmp_path / "noisy")
    assert result.error_code == "OUTPUT_LIMIT_EXCEEDED"
    assert result.output_truncated
    assert len(result.stdout.encode()) <= STREAM_LIMIT
    assert not result.timed_out


def test_http_ingestion_is_disabled_before_live_components_load(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "backend.app.api.get_settings", lambda: Settings(google_cloud_project="test")
    )
    response = TestClient(app).post("/admin/ingest", json={"rebuild": True})
    assert response.status_code == 403
