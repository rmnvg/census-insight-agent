import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.app.agent.models import ResearchPlan, ResearchSectionPlan
from backend.app.agent.service import AgentChatError
from backend.app.main import app
from backend.tests.test_agent_service import run
from backend.tests.test_sessions_transcript import _events, build


class FakePlanner:
    def __init__(self, plan: ResearchPlan | None = None, fail: bool = False) -> None:
        self.plan = plan
        self.fail = fail
        self.topics: list[str] = []

    async def plan_research(self, topic: str) -> ResearchPlan:
        self.topics.append(topic)
        if self.fail:
            raise RuntimeError("provider unavailable")
        assert self.plan is not None
        return self.plan


def plan() -> ResearchPlan:
    return ResearchPlan(
        title="Literacy in Karnataka",
        sections=[
            ResearchSectionPlan(heading="Headline", question="What is Karnataka literacy?"),
            ResearchSectionPlan(heading="Context", question="Summarize Karnataka"),
            ResearchSectionPlan(heading="Out of scope", question="What was the GDP of France?"),
            ResearchSectionPlan(heading="Duplicate", question="what is  Karnataka literacy?"),
        ],
    )


def test_research_answers_each_section_with_a_validated_turn(tmp_path: Path) -> None:
    service, _ = build(tmp_path)
    session = run(service.create_session())
    events: list[tuple[str, dict[str, Any]]] = []

    async def emit(event: str, data: dict[str, object]) -> None:
        events.append((event, data))

    report = run(
        service.research(
            session.session_id, "  Literacy in   Karnataka ", emit=emit, planner=FakePlanner(plan())
        )
    )

    assert [section.heading for section in report.sections] == [
        "Headline",
        "Context",
        "Out of scope",
    ]
    assert [section.status for section in report.sections] == ["answered", "answered", "declined"]
    assert report.verified_claim_count > 0 and report.citation_count > 0
    answered = report.sections[0].response
    assert answered is not None and all(claim.citation_ids for claim in answered.claims)
    names = [name for name, _ in events]
    assert names[0] == "plan" and names.count("section_done") == 3
    assert "progress" in names and names.count("section_started") == 3

    transcript = run(service.get_transcript(session.session_id))
    assert [(item.role, item.mode) for item in transcript.messages] == [
        ("user", "research"),
        ("assistant", "chat"),
    ]
    assert transcript.messages[0].content == "Literacy in Karnataka"
    assert transcript.messages[1].report == report
    # Section child sessions never appear in the chat history.
    assert [item.session_id for item in run(service.list_sessions())] == [session.session_id]
    assert len({section.session_id for section in report.sections}) == 3


def test_deleting_a_chat_removes_its_research_sessions(tmp_path: Path) -> None:
    service, _ = build(tmp_path)
    session = run(service.create_session())

    async def emit(event: str, data: dict[str, object]) -> None:
        del event, data

    report = run(
        service.research(session.session_id, "Literacy", emit=emit, planner=FakePlanner(plan()))
    )
    children = {section.session_id for section in report.sections}
    run(service.delete_session(session.session_id))
    for child in children:
        assert run(service.get_session(child)) is None
    with sqlite3.connect(service.settings.workspace_root / "checkpoints.sqlite") as connection:
        placeholders = ",".join("?" * len(children))
        remaining = connection.execute(
            f"SELECT COUNT(*) FROM checkpoints WHERE thread_id IN ({placeholders})",
            tuple(children),
        ).fetchone()[0]
    assert remaining == 0


def test_out_of_scope_topic_produces_an_explained_empty_brief(tmp_path: Path) -> None:
    service, _ = build(tmp_path)
    session = run(service.create_session())

    async def emit(event: str, data: dict[str, object]) -> None:
        del event, data

    planner = FakePlanner(
        ResearchPlan(in_scope=False, title="GDP", reason="Census has no GDP data.")
    )
    report = run(
        service.research(session.session_id, "India GDP growth", emit=emit, planner=planner)
    )
    assert not report.in_scope and report.sections == []
    assert report.reason == "Census has no GDP data."


def test_planning_failure_is_recorded_and_raised(tmp_path: Path) -> None:
    service, _ = build(tmp_path)
    session = run(service.create_session())

    async def emit(event: str, data: dict[str, object]) -> None:
        del event, data

    with pytest.raises(AgentChatError) as caught:
        run(
            service.research(
                session.session_id, "Literacy", emit=emit, planner=FakePlanner(fail=True)
            )
        )
    assert caught.value.error.code == "RESEARCH_PLANNING_FAILED"
    last = run(service.get_transcript(session.session_id)).messages[-1]
    assert last.error is not None and last.error.error_code == "RESEARCH_PLANNING_FAILED"


def test_http_research_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service, _ = build(tmp_path)
    planner = FakePlanner(plan())
    original = service.research

    async def research(session_id: str, topic: str, *, emit: Any) -> Any:
        return await original(session_id, topic, emit=emit, planner=planner)

    monkeypatch.setattr(service, "research", research)
    monkeypatch.setattr("backend.app.api.get_agent_service", lambda: service)
    client = TestClient(app)
    session_id = client.post("/sessions").json()["session_id"]
    response = client.post("/research/stream", json={"session_id": session_id, "topic": "Literacy"})
    assert response.status_code == 200
    events = _events(response.text)
    assert [name for name, _ in events][:2] == ["started", "plan"]
    progress = next(data for name, data in events if name == "progress")
    assert {"index", "node", "label"} <= set(progress)
    name, result = events[-1]
    assert name == "result" and result["title"] == "Literacy in Karnataka"
    assert json.dumps(result).count('"answer"') >= 2
    assert (
        client.post("/research/stream", json={"session_id": session_id, "topic": "x"}).status_code
        == 422
    )


def test_step_limit_is_a_typed_retryable_error(tmp_path: Path) -> None:
    # Live: an artifact that needed its one repair and still failed took exactly the old
    # 16-step limit and escaped as an untyped GraphRecursionError.
    from backend.app.agent.service import AgentService
    from backend.app.agent.skills import SkillRegistry
    from backend.tests.test_agent_service import FakeModel, FakeTools, settings

    limited = settings(tmp_path).model_copy(update={"agent_max_steps": 3})
    service = AgentService(limited, FakeModel(), FakeTools(SkillRegistry(tmp_path / "skills")))  # type: ignore[arg-type]
    session = run(service.create_session())
    with pytest.raises(AgentChatError) as caught:
        run(service.chat(session.session_id, "What is Karnataka literacy?"))
    assert caught.value.error.code == "AGENT_STEP_LIMIT"
    assert caught.value.error.retryable
    assert service.get_trace(caught.value.run_id) is not None


def test_one_crashing_section_does_not_sink_the_brief(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _ = build(tmp_path)
    session = run(service.create_session())
    original = service._isolated_turn

    async def flaky(session_id: str, message: str, on_progress: Any) -> Any:
        if "Summarize" in message:
            raise RuntimeError("unexpected")
        return await original(session_id, message, on_progress)

    monkeypatch.setattr(service, "_isolated_turn", flaky)

    async def emit(event: str, data: dict[str, object]) -> None:
        del event, data

    report = run(
        service.research(session.session_id, "Literacy", emit=emit, planner=FakePlanner(plan()))
    )
    assert [section.status for section in report.sections] == ["answered", "failed", "declined"]
    failed = report.sections[1]
    assert failed.error is not None and failed.error.error_code == "INTERNAL_ERROR"
