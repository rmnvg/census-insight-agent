"""Langfuse observability, checked against the exact spans the SDK would export.

A real Langfuse client with an in-memory OpenTelemetry exporter: no server, no network.
"""

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langfuse import Langfuse
from langfuse import LangfuseOtelSpanAttributes as Attr
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from backend.app.agent.service import AgentService, UnknownRunError
from backend.app.agent.skills import SkillRegistry
from backend.app.main import app
from backend.app.observability import (
    LangfuseObservability,
    Observability,
    TurnObservation,
    build_observability,
    trace_id_for,
)
from backend.tests.test_agent_service import FakeModel, FakeTools, settings

QUESTION = "What is Karnataka literacy?"


class Recorded:
    def __init__(self, exporter: InMemorySpanExporter, scores: list[dict[str, Any]]) -> None:
        self.exporter = exporter
        self.scores = scores

    def spans(self, client: Langfuse) -> list[ReadableSpan]:
        client.flush()
        return list(self.exporter.get_finished_spans())


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> Iterator[Recorded]:
    exporter = InMemorySpanExporter()
    scores: list[dict[str, Any]] = []
    monkeypatch.setattr(Langfuse, "create_score", lambda self, **kwargs: scores.append(kwargs))
    yield Recorded(exporter, scores)


def observed(tmp_path: Path, exporter: InMemorySpanExporter, **overrides: Any) -> AgentService:
    # The SDK keeps one tracer per public key, so each test needs its own key to get its own
    # exporter.
    public_key = f"pk-lf-{uuid4().hex}"
    configured = settings(tmp_path).model_copy(
        update={
            "langfuse_base_url": "http://langfuse.invalid",
            "langfuse_public_key": public_key,
            **overrides,
        }
    )
    client = Langfuse(
        base_url="http://langfuse.invalid",
        public_key=public_key,
        secret_key="sk-lf-test",
        span_exporter=exporter,
        flush_at=1,
    )
    service = AgentService(
        configured, FakeModel(), cast(Any, FakeTools(SkillRegistry(tmp_path / "skills")))
    )
    service.observability = LangfuseObservability(configured, client=client)
    return service


def attributes(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def test_disabled_by_default_and_needs_all_three_settings(tmp_path: Path) -> None:
    assert type(build_observability(settings(tmp_path))) is Observability
    partial = settings(tmp_path).model_copy(update={"langfuse_base_url": "http://x"})
    assert type(build_observability(partial)) is Observability


def test_turn_is_one_trace_per_run_grouped_by_session_without_content(
    tmp_path: Path, recorded: Recorded
) -> None:
    service = observed(tmp_path, recorded.exporter)
    session = asyncio.run(service.create_session())
    response = asyncio.run(service.chat(session.session_id, QUESTION))
    client = cast(LangfuseObservability, service.observability).client

    [turn] = [span for span in recorded.spans(client) if span.name == "agent_turn"]
    values = attributes(turn)
    assert format(turn.context.trace_id, "032x") == trace_id_for(response.trace_id)
    assert values[Attr.TRACE_SESSION_ID] == session.session_id
    assert values[Attr.OBSERVATION_TYPE] == "agent"
    # Default: no prompt, question, or answer text leaves the process.
    assert Attr.OBSERVATION_INPUT not in values and Attr.OBSERVATION_OUTPUT not in values
    assert not any(QUESTION in str(value) for value in values.values())
    metadata = {key: value for key, value in values.items() if "metadata" in key}
    assert any('"answered"' in str(value) or value == "answered" for value in metadata.values())
    [score] = recorded.scores
    assert (score["name"], score["value"]) == ("answer_status", "answered")
    assert score["trace_id"] == trace_id_for(response.trace_id)


def test_content_is_sent_only_when_capture_is_enabled(tmp_path: Path, recorded: Recorded) -> None:
    service = observed(tmp_path, recorded.exporter, langfuse_capture_content=True)
    session = asyncio.run(service.create_session())
    asyncio.run(service.chat(session.session_id, QUESTION))
    client = cast(LangfuseObservability, service.observability).client
    [turn] = [span for span in recorded.spans(client) if span.name == "agent_turn"]
    assert QUESTION in str(attributes(turn)[Attr.OBSERVATION_INPUT])
    assert "75.36" in str(attributes(turn)[Attr.OBSERVATION_OUTPUT])


def test_each_model_call_becomes_a_generation_with_token_usage(
    tmp_path: Path, recorded: Recorded
) -> None:
    service = observed(tmp_path, recorded.exporter)
    observability = cast(LangfuseObservability, service.observability)
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="classified",
                    usage_metadata={
                        "input_tokens": 120,
                        "output_tokens": 30,
                        "total_tokens": 190,
                        "output_token_details": {"reasoning": 40},
                    },
                )
            ]
        )
    )

    async def turn() -> None:
        with observability.turn(session_id="a" * 32, run_id="r-1", question=QUESTION) as handle:
            config = {"callbacks": handle.callbacks, "metadata": {"langgraph_node": "classify"}}
            await model.ainvoke([HumanMessage(content=QUESTION)], config=cast(Any, config))

    asyncio.run(turn())
    spans = recorded.spans(observability.client)
    [generation] = [span for span in spans if span.name == "classify"]
    [parent] = [span for span in spans if span.name == "agent_turn"]
    values = attributes(generation)
    assert values[Attr.OBSERVATION_TYPE] == "generation"
    assert generation.parent is not None and generation.parent.span_id == parent.context.span_id
    assert json.loads(values[Attr.OBSERVATION_USAGE_DETAILS]) == {
        "input": 120,
        "output": 30,
        "total": 190,
        "output_reasoning": 40,
    }
    assert Attr.OBSERVATION_INPUT not in values and Attr.OBSERVATION_OUTPUT not in values


def test_failed_turn_is_marked_as_an_error(tmp_path: Path, recorded: Recorded) -> None:
    service = observed(tmp_path, recorded.exporter)
    session = asyncio.run(service.create_session())

    class Boom(RuntimeError):
        pass

    async def explode(*args: Any, **kwargs: Any) -> Any:
        raise Boom("graph failed")

    service._observed_turn = explode  # type: ignore[method-assign]
    with pytest.raises(Boom):
        asyncio.run(service.chat(session.session_id, QUESTION))
    client = cast(LangfuseObservability, service.observability).client
    [turn] = [span for span in recorded.spans(client) if span.name == "agent_turn"]
    assert attributes(turn)[Attr.OBSERVATION_LEVEL] == "ERROR"
    assert [score["value"] for score in recorded.scores] == ["failed"]


def test_monitoring_failure_never_breaks_an_answer(tmp_path: Path) -> None:
    service = AgentService(
        settings(tmp_path), FakeModel(), cast(Any, FakeTools(SkillRegistry(tmp_path / "skills")))
    )

    class BrokenClient:
        def start_as_current_observation(self, **kwargs: Any) -> Any:
            raise ConnectionError("langfuse unreachable")

        def create_score(self, **kwargs: Any) -> None:
            raise ConnectionError("langfuse unreachable")

    broken = LangfuseObservability(
        settings(tmp_path).model_copy(update={"langfuse_capture_content": False}),
        client=cast(Any, BrokenClient()),
    )
    service.observability = broken
    session = asyncio.run(service.create_session())
    response = asyncio.run(service.chat(session.session_id, QUESTION))
    assert response.claims
    asyncio.run(service.record_feedback(response.trace_id, positive=True))


def test_feedback_is_stored_beside_the_trace_and_scored(tmp_path: Path, recorded: Recorded) -> None:
    service = observed(tmp_path, recorded.exporter)
    session = asyncio.run(service.create_session())
    response = asyncio.run(service.chat(session.session_id, QUESTION))
    recorded.scores.clear()

    asyncio.run(service.record_feedback(response.trace_id, positive=False))
    stored = tmp_path / "workspace" / "sessions" / session.session_id / "feedback"
    record = json.loads((stored / f"{response.trace_id}.json").read_text(encoding="utf-8"))
    assert record["positive"] is False
    [score] = recorded.scores
    assert score["name"] == "user_feedback" and score["value"] == 0
    assert score["trace_id"] == trace_id_for(response.trace_id)
    assert str(score["data_type"]).endswith("BOOLEAN")

    with pytest.raises(UnknownRunError):
        asyncio.run(service.record_feedback("00000000-0000-4000-8000-000000000000", positive=True))
    asyncio.run(service.delete_session(session.session_id))
    assert not stored.exists()


def test_http_feedback_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = AgentService(
        settings(tmp_path), FakeModel(), cast(Any, FakeTools(SkillRegistry(tmp_path / "skills")))
    )
    monkeypatch.setattr("backend.app.api.get_agent_service", lambda: service)
    session = asyncio.run(service.create_session())
    response = asyncio.run(service.chat(session.session_id, QUESTION))
    client = TestClient(app)

    ok = client.post(f"/runs/{response.trace_id}/feedback", json={"rating": "up"})
    assert ok.status_code == 204
    unknown = client.post(
        "/runs/00000000-0000-4000-8000-000000000000/feedback", json={"rating": "up"}
    )
    assert unknown.status_code == 404
    assert client.post("/runs/not-a-run/feedback", json={"rating": "up"}).status_code == 422
    invalid = client.post(f"/runs/{response.trace_id}/feedback", json={"rating": "meh"})
    assert invalid.status_code == 422


def test_no_op_observation_has_no_callbacks() -> None:
    with Observability().turn(session_id="a" * 32, run_id="r", question="q") as handle:
        assert isinstance(handle, TurnObservation) and handle.callbacks == []
