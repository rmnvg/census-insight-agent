"""Optional Langfuse observability: agent turns, model calls, user feedback, and evaluations.

Enabled when LANGFUSE_BASE_URL, LANGFUSE_PUBLIC_KEY, and LANGFUSE_SECRET_KEY are all set; otherwise
every hook is a no-op and nothing is sent.

What is sent by default, per turn: a trace grouped by session, with the outcome, refusal reason,
claim and citation counts, and latency. For each Gemini call inside the turn: a generation with
the graph node, model, token usage (including reasoning tokens), latency, and error status.

Prompt, evidence, and answer text are not sent unless LANGFUSE_CAPTURE_CONTENT=true, matching the
local trace contract. Langfuse is a monitoring sink, never an input: nothing read from it affects
an answer.
"""

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langfuse.api import ScoreDataType

from backend.app.agent.models import AgentResponse
from backend.app.config import Settings

if TYPE_CHECKING:
    from langfuse import Langfuse, LangfuseAgent, LangfuseGeneration

logger = logging.getLogger(__name__)


class TurnObservation:
    """Handle for one agent turn; the no-op base records nothing."""

    callbacks: list[BaseCallbackHandler] = []

    def answered(self, response: AgentResponse, *, refusal_reason: str | None) -> None:
        return None

    def failed(self, error_code: str) -> None:
        return None


class Observability:
    enabled = False

    @contextmanager
    def turn(self, *, session_id: str, run_id: str, question: str) -> Iterator[TurnObservation]:
        del session_id, run_id, question
        yield TurnObservation()

    def record_feedback(self, run_id: str, *, positive: bool) -> None:
        return None

    def shutdown(self) -> None:
        return None


def trace_id_for(run_id: str) -> str:
    """The Langfuse trace ID of an agent run, derivable from the run ID alone."""
    from langfuse import Langfuse

    return Langfuse.create_trace_id(seed=run_id)


def _usage(response: LLMResult) -> dict[str, int]:
    for generations in response.generations:
        for generation in generations:
            if not isinstance(generation, ChatGeneration):
                continue
            usage = getattr(generation.message, "usage_metadata", None) or {}
            details = {
                "input": usage.get("input_tokens"),
                "output": usage.get("output_tokens"),
                "total": usage.get("total_tokens"),
                "output_reasoning": (usage.get("output_token_details") or {}).get("reasoning"),
                "input_cache_read": (usage.get("input_token_details") or {}).get("cache_read"),
            }
            return {key: int(value) for key, value in details.items() if value}
    return {}


class _ModelCallRecorder(AsyncCallbackHandler):
    """Records each chat-model call under the turn's trace as a Langfuse generation."""

    raise_error = False

    def __init__(self, turn: "LangfuseAgent", *, capture_content: bool) -> None:
        self._turn = turn
        self._capture_content = capture_content
        self._open: dict[UUID, LangfuseGeneration] = {}

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        meta = metadata or {}
        self._open[run_id] = self._turn.start_observation(
            name=str(meta.get("langgraph_node") or "model_call"),
            as_type="generation",
            model=meta.get("ls_model_name"),
            input=(
                [[message.model_dump() for message in batch] for batch in messages]
                if self._capture_content
                else None
            ),
            metadata={
                "provider": meta.get("ls_provider"),
                "graph_step": meta.get("langgraph_step"),
            },
        )

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        generation = self._open.pop(run_id, None)
        if generation is None:
            return
        output = (
            [[item.text for item in batch] for batch in response.generations]
            if self._capture_content
            else None
        )
        generation.update(usage_details=_usage(response) or None, output=output)
        generation.end()

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        generation = self._open.pop(run_id, None)
        if generation is None:
            return
        # The error type only: provider messages can echo request content.
        generation.update(level="ERROR", status_message=type(error).__name__)
        generation.end()

    def close_open(self) -> None:
        for generation in self._open.values():
            generation.update(level="WARNING", status_message="Call did not report completion")
            generation.end()
        self._open.clear()


class _LangfuseTurn(TurnObservation):
    def __init__(self, span: "LangfuseAgent", recorder: _ModelCallRecorder, capture: bool) -> None:
        self._span = span
        self._recorder = recorder
        self._capture = capture
        self.callbacks = [recorder]

    def answered(self, response: AgentResponse, *, refusal_reason: str | None) -> None:
        status = "refused" if response.refusal else "answered"
        self._span.update(
            output=response.answer_markdown if self._capture else None,
            level="WARNING" if response.refusal else "DEFAULT",
            metadata={
                "answer_status": status,
                "refusal_reason": refusal_reason,
                "claims": len(response.claims),
                "citations": len(response.citations),
                "artifacts": len(response.artifacts),
            },
        )
        self._span.score_trace(
            name="answer_status", value=status, data_type=ScoreDataType.CATEGORICAL
        )

    def failed(self, error_code: str) -> None:
        self._span.update(level="ERROR", status_message=error_code)
        self._span.score_trace(
            name="answer_status", value="failed", data_type=ScoreDataType.CATEGORICAL
        )


class LangfuseObservability(Observability):
    enabled = True

    def __init__(self, settings: Settings, client: "Langfuse | None" = None) -> None:
        from langfuse import Langfuse

        self._capture = settings.langfuse_capture_content
        self._model = settings.gemini_chat_model
        self.client = client or Langfuse(
            base_url=settings.langfuse_base_url,
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            environment=settings.langfuse_environment,
        )

    @contextmanager
    def turn(self, *, session_id: str, run_id: str, question: str) -> Iterator[TurnObservation]:
        from langfuse import propagate_attributes

        try:
            span_context = self.client.start_as_current_observation(
                name="agent_turn",
                as_type="agent",
                trace_context={"trace_id": trace_id_for(run_id)},
                input=question if self._capture else None,
                metadata={"run_id": run_id, "chat_model": self._model},
            )
            span = span_context.__enter__()
        except Exception:
            # Monitoring must never take an answer down with it.
            logger.exception("Langfuse turn could not start; continuing without it")
            yield TurnObservation()
            return
        recorder = _ModelCallRecorder(span, capture_content=self._capture)
        try:
            with propagate_attributes(session_id=session_id, trace_name="agent_turn"):
                yield _LangfuseTurn(span, recorder, self._capture)
        finally:
            recorder.close_open()
            span_context.__exit__(None, None, None)

    def record_feedback(self, run_id: str, *, positive: bool) -> None:
        self.client.create_score(
            name="user_feedback",
            value=1 if positive else 0,
            data_type="BOOLEAN",
            trace_id=trace_id_for(run_id),
        )

    def shutdown(self) -> None:
        self.client.shutdown()


def build_observability(settings: Settings) -> Observability:
    configured = (
        settings.langfuse_base_url,
        settings.langfuse_public_key,
        settings.langfuse_secret_key.get_secret_value(),
    )
    if all(configured):
        return LangfuseObservability(settings)
    return Observability()


def callbacks_config(observation: TurnObservation) -> dict[str, Sequence[BaseCallbackHandler]]:
    return {"callbacks": observation.callbacks} if observation.callbacks else {}
