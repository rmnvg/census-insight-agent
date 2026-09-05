import asyncio

import pytest
from fastapi.testclient import TestClient

from backend.app.agent.evidence import (
    bound_assessment_evidence,
    contains_document_level_table_row,
)
from backend.app.agent.provider import GeminiAgentModel, ProviderCallTimeout
from backend.app.agent.service import AgentService
from backend.app.agent.skills import SkillRegistry
from backend.app.main import app
from backend.app.retrieval.models import RetrievedEvidence
from backend.tests.test_agent_service import (
    FakeTools,
    TimeoutAfterSuccessModel,
    census_evidence,
    run,
    settings,
)


class SlowRunnable:
    def __init__(self) -> None:
        self.calls = 0
        self.cancellations = 0

    def with_structured_output(self, schema: object) -> "SlowRunnable":
        del schema
        return self

    async def ainvoke(self, messages: object) -> object:
        del messages
        self.calls += 1
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            self.cancellations += 1
            raise
        return {}


def test_provider_deadline_bounds_one_retry_and_cancels_work() -> None:
    runnable = SlowRunnable()
    model = GeminiAgentModel(runnable, timeout_seconds=0.04, max_retries=1)  # type: ignore[arg-type]
    with pytest.raises(ProviderCallTimeout) as caught:
        run(model.assess_evidence("question", "comparison", []))
    assert runnable.calls == 2
    assert runnable.cancellations == 2
    assert caught.value.retry_count == 1
    assert caught.value.elapsed_seconds < 0.1


def test_assessment_budget_is_deduplicated_and_fair_between_documents() -> None:
    evidence: list[RetrievedEvidence] = []
    for region in ("Karnataka", "Odisha"):
        for page in range(1, 8):
            item = census_evidence(region, page)
            evidence.append(item.model_copy(update={"text": item.text + "x" * 200}))
    evidence.append(evidence[0])
    bounded = bound_assessment_evidence(evidence, max_characters=1800, max_chunks=6)
    assert len({item.chunk_id for item in bounded}) == len(bounded)
    assert {item.region for item in bounded} == {"Karnataka", "Odisha"}
    assert sum(len(item.text) for item in bounded) <= 1800


def test_assessment_budget_promotes_expanded_document_total_row() -> None:
    evidence: list[RetrievedEvidence] = []
    for region in ("Karnataka", "Odisha"):
        for page in range(1, 11):
            item = census_evidence(region, page)
            evidence.append(
                item.model_copy(
                    update={
                        "text": (
                            "| District | Literacy Rate 2011 |\n"
                            f"| District {page} | {60 + page}.0 |"
                        ),
                        "retrieval_score": 1.0 - page / 100,
                    }
                )
            )
        total = census_evidence(region, 82).model_copy(
            update={
                "chunk_id": f"expanded-total-{region.casefold()}",
                "text": (
                    f"| State | 2011 Total Literacy Rate |\n| <b>{region.upper()}</b> | 72.9 |"
                ),
                "retrieval_score": 0.0,
            }
        )
        evidence.append(total)

    bounded = bound_assessment_evidence(evidence, max_characters=700, max_chunks=6)

    assert {item.chunk_id for item in bounded} >= {
        "expanded-total-karnataka",
        "expanded-total-odisha",
    }
    assert all(
        contains_document_level_table_row(item)
        for item in bounded
        if item.chunk_id.startswith("expanded-total-")
    )


def test_chat_timeout_returns_structured_504_and_trace_is_fetchable(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))
    model = TimeoutAfterSuccessModel()
    service = AgentService(
        settings(root),
        model,
        FakeTools(SkillRegistry(root / "skills")),  # type: ignore[arg-type]
    )
    session = run(service.create_session())
    run(service.chat(session.session_id, "What is Karnataka literacy?"))
    model.fail_assessment = True
    monkeypatch.setattr("backend.app.api.get_agent_service", lambda: service)
    client = TestClient(app)
    response = client.post(
        "/chat",
        json={"session_id": session.session_id, "message": "How does that compare with Odisha?"},
    )
    body = response.json()
    assert response.status_code == 504
    assert body["error_code"] == "EVIDENCE_ASSESSMENT_TIMEOUT"
    assert body["retryable"] is True
    assert body["session_id"] == session.session_id
    trace = client.get(f"/runs/{body['trace_id']}/trace")
    assert trace.status_code == 200
    assert trace.json()["status"] == "failed"
    assert trace.json()["run_status"] == "failed"
    assert trace.json()["answer_status"] == "refused"
    assert trace.json()["refusal_reason"] == "operational_error"
