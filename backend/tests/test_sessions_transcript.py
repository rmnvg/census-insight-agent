import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.agent.persistence import _default_title
from backend.app.agent.service import AgentChatError, AgentService
from backend.app.agent.skills import SkillRegistry
from backend.app.main import app
from backend.tests.test_agent_service import (
    FakeModel,
    FakeTools,
    TimeoutAfterSuccessModel,
    run,
    settings,
)


def build(tmp_path: Path, model: object | None = None) -> tuple[AgentService, FakeTools]:
    tools = FakeTools(SkillRegistry(tmp_path / "skills"))
    service = AgentService(settings(tmp_path), model or FakeModel(), tools)  # type: ignore[arg-type]
    return service, tools


def test_chat_records_a_round_trippable_transcript_and_titles_the_session(
    tmp_path: Path,
) -> None:
    service, _ = build(tmp_path)
    session = run(service.create_session())
    response = run(service.chat(session.session_id, "  What is Karnataka literacy?  "))

    transcript = run(service.get_transcript(session.session_id))
    assert [entry.role for entry in transcript.messages] == ["user", "assistant"]
    assert transcript.messages[0].content == "What is Karnataka literacy?"
    assert transcript.messages[1].response == response
    assert transcript.session.title == "What is Karnataka literacy?"


def test_session_list_hides_empty_sessions_and_orders_by_activity(tmp_path: Path) -> None:
    service, _ = build(tmp_path)
    older = run(service.create_session())
    run(service.create_session())  # never used: must not clutter history
    newer = run(service.create_session())
    run(service.chat(older.session_id, "What is Karnataka literacy?"))
    run(service.chat(newer.session_id, "What is Karnataka literacy?"))

    listed = run(service.list_sessions())
    assert [item.session_id for item in listed] == [newer.session_id, older.session_id]
    assert all(item.message_count == 2 for item in listed)


def test_transcript_is_display_only_and_memory_survives_its_removal(tmp_path: Path) -> None:
    model = FakeModel()
    service, tools = build(tmp_path, model)
    session = run(service.create_session())
    run(service.chat(session.session_id, "What is Karnataka literacy?"))
    with sqlite3.connect(service.settings.workspace_root / "checkpoints.sqlite") as connection:
        connection.execute("DELETE FROM app_messages")

    follow_up = run(service.chat(session.session_id, "How does that compare?"))
    assert {citation.document_id for citation in follow_up.citations} == {
        "doc-karnataka",
        "doc-odisha",
    }
    assert dict(model.context_sizes)["How does that compare?"] > 0


def test_failed_turn_is_recorded_as_an_error_entry(tmp_path: Path) -> None:
    model = TimeoutAfterSuccessModel()
    service, _ = build(tmp_path, model)
    session = run(service.create_session())
    run(service.chat(session.session_id, "What is Karnataka literacy?"))
    model.fail_assessment = True
    with pytest.raises(AgentChatError) as caught:
        run(service.chat(session.session_id, "How does that compare with Odisha?"))

    last = run(service.get_transcript(session.session_id)).messages[-1]
    assert last.role == "assistant" and last.response is None
    assert last.error is not None
    assert last.error.error_code == "EVIDENCE_ASSESSMENT_TIMEOUT"
    assert last.error.trace_id == caught.value.run_id


def test_progress_callback_reports_nodes_and_matches_plain_invocation(tmp_path: Path) -> None:
    service, _ = build(tmp_path)
    plain_session = run(service.create_session())
    streamed_session = run(service.create_session())
    nodes: list[str] = []

    async def record(node: str) -> None:
        nodes.append(node)

    plain = run(service.chat(plain_session.session_id, "What is Karnataka literacy?"))
    streamed = run(
        service.chat(streamed_session.session_id, "What is Karnataka literacy?", on_progress=record)
    )
    assert nodes[0] == "load_memory" and nodes[-1] == "persist_result"
    assert "validate_citations" in nodes
    assert streamed.answer_markdown == plain.answer_markdown
    assert streamed.citations == plain.citations


def test_rename_and_delete_remove_every_session_record(tmp_path: Path) -> None:
    service, _ = build(tmp_path)
    session = run(service.create_session())
    response = run(service.chat(session.session_id, "What is Karnataka literacy?"))
    renamed = run(service.rename_session(session.session_id, "  Literacy   check "))
    assert renamed.title == "Literacy check"
    session_dir = tmp_path / "workspace" / "sessions" / session.session_id
    assert session_dir.is_dir()

    run(service.delete_session(session.session_id))
    assert run(service.get_session(session.session_id)) is None
    assert service.get_trace(response.trace_id) is None
    assert not session_dir.exists()
    with sqlite3.connect(service.settings.workspace_root / "checkpoints.sqlite") as connection:
        remaining = connection.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (session.session_id,)
        ).fetchone()[0]
    assert remaining == 0


def test_default_title_cuts_long_questions_at_a_word_boundary() -> None:
    title = _default_title(
        "Compare the literacy rates of Karnataka and Odisha across rural and urban areas please"
    )
    assert title.endswith("…") and len(title) <= 61
    assert not title[:-1].endswith(" ")


def _events(body: str) -> list[tuple[str, dict[str, object]]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        if "event" in lines:
            events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_http_session_endpoints_and_streamed_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _ = build(tmp_path)
    monkeypatch.setattr("backend.app.api.get_agent_service", lambda: service)
    client = TestClient(app)
    session_id = client.post("/sessions").json()["session_id"]

    streamed = client.post(
        "/chat/stream", json={"session_id": session_id, "message": "What is Karnataka literacy?"}
    )
    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("text/event-stream")
    events = _events(streamed.text)
    assert events[0][0] == "started"
    progress = [data for name, data in events if name == "progress"]
    assert progress[0] == {"node": "load_memory", "label": "Loading validated conversation memory"}
    name, result = events[-1]
    assert name == "result" and result["answer"] and result["citations"]

    listed = client.get("/sessions").json()
    assert [item["session_id"] for item in listed] == [session_id]
    messages = client.get(f"/sessions/{session_id}/messages").json()["messages"]
    assert messages[1]["response"]["answer"] == result["answer"]

    assert client.patch(f"/sessions/{session_id}", json={"title": "Renamed"}).json()["title"] == (
        "Renamed"
    )
    assert client.delete(f"/sessions/{session_id}").status_code == 204
    assert client.get(f"/sessions/{session_id}/messages").status_code == 404
    assert client.post(
        "/chat/stream", json={"session_id": session_id, "message": "hi"}
    ).status_code == (404)
    assert client.delete("/sessions/not-a-session").status_code == 422
