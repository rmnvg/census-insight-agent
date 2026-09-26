import asyncio
import logging
import shutil
import time
from collections.abc import Awaitable, Callable
from functools import lru_cache, partial
from typing import Any, Protocol, cast
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from qdrant_client import QdrantClient

from backend.app.agent.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    checkpoint_safe,
    hydrate_agent_state,
)
from backend.app.agent.errors import AgentOperationalError
from backend.app.agent.graph import AgentGraph
from backend.app.agent.models import (
    AgentErrorResponse,
    AgentResponse,
    AgentState,
    ResearchPlan,
    ResearchReport,
    ResearchSection,
    RunTrace,
    SessionContextStatus,
    SessionRecord,
    SessionSummary,
    SessionTranscript,
)
from backend.app.agent.persistence import (
    TraceStore,
    safe_trace_details,
    validate_session_id,
)
from backend.app.agent.provider import (
    AgentModel,
    GeminiAgentModel,
    reset_request_deadline,
    set_request_deadline,
)
from backend.app.agent.skills import SkillRegistry
from backend.app.agent.state import build_state_backend
from backend.app.agent.tools import AgentTools
from backend.app.config import Settings, get_settings
from backend.app.execution.client import ArtifactStore, ExecutionQueueClient
from backend.app.ingestion.uploads import withdrawn_document_ids
from backend.app.observability import TurnObservation, build_observability, callbacks_config
from backend.app.providers.chat import get_chat_model
from backend.app.providers.embeddings import VertexEmbeddingProvider
from backend.app.retrieval.qdrant_store import QdrantStore
from backend.app.retrieval.service import HybridRetrievalService
from backend.app.retrieval.sparse import BM25SparseEncoder

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], Awaitable[None]]
ResearchEmitter = Callable[[str, dict[str, object]], Awaitable[None]]

MAX_RESEARCH_SECTIONS = 5
RESEARCH_SECTION_CONCURRENCY = 3


class ResearchPlanner(Protocol):
    async def plan_research(self, topic: str) -> ResearchPlan: ...


class UnknownSessionError(ValueError):
    pass


class UnknownRunError(LookupError):
    pass


class AgentChatError(RuntimeError):
    def __init__(self, error: AgentOperationalError, session_id: str, run_id: str) -> None:
        super().__init__(error.message)
        self.error = error
        self.session_id = session_id
        self.run_id = run_id


class AgentService:
    def __init__(
        self,
        settings: Settings,
        model: AgentModel,
        tools: AgentTools,
    ) -> None:
        self.settings = settings
        self.model = model
        self.state = build_state_backend(settings)
        self.sessions = self.state.sessions
        self.observability = build_observability(settings)
        self.traces = TraceStore(settings.workspace_root)
        self.execution_queue = ExecutionQueueClient(settings.execution_queue_root)
        self.artifacts = ArtifactStore(settings.workspace_root, settings.execution_queue_root)
        self.graph_factory = AgentGraph(
            model,
            tools,
            max_tool_calls=settings.agent_max_tool_calls,
            memory_turn_threshold=settings.agent_memory_turn_threshold,
            assessment_max_characters=settings.agent_assessment_max_characters,
            assessment_max_chunks=settings.agent_assessment_max_chunks,
            execution_queue=self.execution_queue,
            artifact_store=self.artifacts,
            execution_timeout_seconds=settings.artifact_execution_timeout_seconds,
            artifact_max_files=settings.artifact_max_files,
            artifact_max_total_bytes=settings.artifact_max_total_bytes,
        )

    @classmethod
    def live(cls, settings: Settings | None = None) -> "AgentService":
        resolved = settings or get_settings()
        client = QdrantClient(url=resolved.qdrant_url)
        store = QdrantStore(
            client,
            resolved.qdrant_collection,
            dense_dimensions=resolved.gemini_embedding_dimension,
            dense_model=resolved.gemini_embedding_model,
            sparse_model=resolved.sparse_embedding_model,
        )
        retrieval = HybridRetrievalService(
            client=client,
            collection_name=resolved.qdrant_collection,
            dense_provider=VertexEmbeddingProvider(resolved),
            sparse_encoder=BM25SparseEncoder(
                resolved.sparse_embedding_model,
                cache_dir=resolved.data_root / "processed" / "fastembed-cache",
            ),
            excluded_document_ids=partial(withdrawn_document_ids, resolved.data_root),
        )
        tools = AgentTools(
            retrieval,
            store,
            SkillRegistry(resolved.skills_dir),
            resolved.data_root,
            timeout_seconds=resolved.agent_provider_timeout_seconds,
        )
        model = GeminiAgentModel(
            get_chat_model(resolved),
            timeout_seconds=resolved.agent_provider_timeout_seconds,
            max_retries=resolved.agent_provider_max_retries,
            catalog=tools.list_documents,
        )
        return cls(resolved, model, tools)

    async def initialize(self) -> None:
        await self.state.initialize()

    async def close(self) -> None:
        await self.state.close()
        self.observability.shutdown()

    async def create_session(self) -> SessionRecord:
        await self.initialize()
        return await self.sessions.create()

    async def get_session(self, session_id: str) -> SessionRecord | None:
        await self.initialize()
        return await self.sessions.get(session_id)

    async def list_sessions(self, limit: int = 100) -> list[SessionSummary]:
        await self.initialize()
        return await self.sessions.list_recent(limit)

    async def get_transcript(self, session_id: str) -> SessionTranscript:
        await self.initialize()
        session = await self.sessions.get(session_id)
        if session is None:
            raise UnknownSessionError("Unknown session")
        return SessionTranscript(session=session, messages=await self.sessions.messages(session_id))

    async def rename_session(self, session_id: str, title: str) -> SessionRecord:
        await self.initialize()
        if await self.sessions.get(session_id) is None:
            raise UnknownSessionError("Unknown session")
        await self.sessions.set_title(session_id, title)
        session = await self.sessions.get(session_id)
        assert session is not None
        return session

    async def delete_session(self, session_id: str) -> None:
        """Remove a session's transcript, checkpoints, traces, and artifacts.

        Waits for any in-flight turn on the same session so a running graph never loses its
        checkpoint underneath it.
        """
        validate_session_id(session_id)
        await self.initialize()
        if await self.sessions.get(session_id) is None:
            raise UnknownSessionError("Unknown session")
        async with self.state.session_lock(session_id):
            # Research sections live in hidden child sessions; they go with their chat.
            for owned in [*await self.sessions.children(session_id), session_id]:
                async with self.state.checkpointer() as saver:
                    await saver.adelete_thread(owned)
                await self.sessions.delete(owned)
                sessions_root = (self.settings.workspace_root / "sessions").resolve()
                session_dir = (sessions_root / owned).resolve()
                if session_dir.parent == sessions_root and session_dir.is_dir():
                    await asyncio.to_thread(shutil.rmtree, session_dir)

    async def research(
        self,
        session_id: str,
        topic: str,
        *,
        emit: ResearchEmitter,
        planner: ResearchPlanner | None = None,
    ) -> ResearchReport:
        """Plan a brief, answer each section with a full validated agent turn, and save it.

        Each section runs in its own hidden child session so sections cannot contaminate each
        other's validated memory, and every factual sentence in the brief is a claim that
        passed the same citation validation as an ordinary chat answer.
        """
        validate_session_id(session_id)
        topic = " ".join(topic.split())
        if not topic:
            raise ValueError("Research topic must be non-empty")
        if planner is None and hasattr(self.model, "plan_research"):
            planner = cast(ResearchPlanner, self.model)
        if planner is None:
            raise ValueError("Research planning is unavailable")
        resolved_planner = planner
        async with self.state.session_lock(session_id):
            await self.initialize()
            if await self.sessions.get(session_id) is None:
                raise UnknownSessionError("Unknown session")
            await self.sessions.append_user_message(session_id, topic, mode="research")
            started = time.monotonic()
            try:
                plan = await resolved_planner.plan_research(topic)
            except Exception as error:
                logger.exception("Research planning failed")
                failure = AgentErrorResponse(
                    error_code="RESEARCH_PLANNING_FAILED",
                    message="The research plan could not be created. Please retry.",
                    session_id=session_id,
                    trace_id="",
                    retryable=True,
                )
                await self._record_assistant(session_id, error=failure)
                raise AgentChatError(
                    AgentOperationalError(
                        code=failure.error_code,
                        node="plan_research",
                        message=failure.message,
                        retryable=True,
                        elapsed_seconds=time.monotonic() - started,
                        configured_timeout_seconds=self.settings.agent_provider_timeout_seconds,
                        retry_count=0,
                    ),
                    session_id,
                    "",
                ) from error
            sections = _normalized_sections(plan)
            await emit(
                "plan",
                {
                    "title": plan.title,
                    "in_scope": plan.in_scope and bool(sections),
                    "reason": plan.reason,
                    "sections": [section.model_dump() for section in sections],
                },
            )
            gate = asyncio.Semaphore(RESEARCH_SECTION_CONCURRENCY)

            async def run_section(index: int) -> ResearchSection:
                plan_section = sections[index]
                async with gate:
                    child = await self.sessions.create(parent_session_id=session_id)
                    await emit("section_started", {"index": index})

                    async def progress(node: str) -> None:
                        await emit("progress", {"index": index, "node": node})

                    try:
                        response = await self._isolated_turn(
                            child.session_id, plan_section.question, progress
                        )
                        section = ResearchSection(
                            heading=plan_section.heading,
                            question=plan_section.question,
                            status="declined" if response.refusal else "answered",
                            session_id=child.session_id,
                            response=response,
                        )
                    except Exception as unexpected:
                        # One section must never sink the whole brief.
                        if isinstance(unexpected, AgentChatError):
                            error = unexpected
                        else:
                            logger.exception("Research section failed unexpectedly")
                            error = AgentChatError(
                                AgentOperationalError(
                                    code="INTERNAL_ERROR",
                                    node="research_section",
                                    message="This section could not be completed.",
                                    retryable=True,
                                    elapsed_seconds=None,
                                    configured_timeout_seconds=None,
                                    retry_count=0,
                                ),
                                child.session_id,
                                "",
                            )
                        section = ResearchSection(
                            heading=plan_section.heading,
                            question=plan_section.question,
                            status="failed",
                            session_id=child.session_id,
                            error=AgentErrorResponse(
                                error_code=error.error.code,
                                message=error.error.message,
                                session_id=child.session_id,
                                trace_id=error.run_id,
                                retryable=error.error.retryable,
                            ),
                        )
                    await emit(
                        "section_done",
                        {"index": index, "section": section.model_dump(mode="json", by_alias=True)},
                    )
                    return section

            results = (
                await asyncio.gather(*(run_section(index) for index in range(len(sections))))
                if sections
                else []
            )
            answered = [
                item.response for item in results if item.status == "answered" and item.response
            ]
            report = ResearchReport(
                topic=topic,
                title=plan.title,
                in_scope=plan.in_scope and bool(sections),
                reason=plan.reason
                or (
                    None if sections else "No answerable questions could be planned for this topic."
                ),
                sections=list(results),
                verified_claim_count=sum(len(response.claims) for response in answered),
                citation_count=len(
                    {
                        (citation.document_id, citation.page_number, citation.chunk_id)
                        for response in answered
                        for citation in response.citations
                    }
                ),
                duration_seconds=round(time.monotonic() - started, 1),
            )
            try:
                await self.sessions.append_assistant_message(session_id, report=report)
            except Exception:
                logger.exception("Could not record research report")
            return report

    async def _isolated_turn(
        self, session_id: str, message: str, on_progress: ProgressCallback
    ) -> AgentResponse:
        async with self.state.session_lock(session_id):
            return await self._chat_turn(session_id, message, on_progress)

    async def get_context_status(self, session_id: str) -> SessionContextStatus:
        validate_session_id(session_id)
        await self.initialize()
        if await self.sessions.get(session_id) is None:
            raise UnknownSessionError("Unknown session")
        history = []
        async with self.state.checkpointer() as saver:
            checkpoint_id = await self.sessions.get_checkpoint_id(session_id)
            if checkpoint_id is not None:
                state = await self._checkpoint_state(saver, session_id, checkpoint_id)
                history = state.get("validated_claim_history", [])
        if not history:
            history = self.traces.recover_validated_claim_history(session_id)
        return SessionContextStatus(
            session_id=session_id,
            validated_turn_count=len(history),
            has_validated_comparison=any(turn.task_type == "comparison" for turn in history),
        )

    async def chat(
        self,
        session_id: str,
        message: str,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> AgentResponse:
        validate_session_id(session_id)
        # Hold the session lock through checkpoint and trace persistence. With Postgres state it
        # is an advisory lock, so a turn on another replica waits for this one.
        async with self.state.session_lock(session_id):
            if not message.strip():
                raise ValueError("Message must be non-empty")
            await self.initialize()
            if await self.sessions.get(session_id) is None:
                raise UnknownSessionError("Unknown session")
            await self.sessions.append_user_message(session_id, message.strip())
            try:
                response = await self._chat_turn(session_id, message, on_progress)
            except AgentChatError as chat_error:
                await self._record_assistant(
                    session_id,
                    error=AgentErrorResponse(
                        error_code=chat_error.error.code,
                        message=chat_error.error.message,
                        session_id=session_id,
                        trace_id=chat_error.run_id,
                        retryable=chat_error.error.retryable,
                    ),
                )
                raise
            except Exception:
                await self._record_assistant(
                    session_id,
                    error=AgentErrorResponse(
                        error_code="INTERNAL_ERROR",
                        message="The request could not be completed. Please retry.",
                        session_id=session_id,
                        trace_id="",
                        retryable=True,
                    ),
                )
                raise
            await self._record_assistant(session_id, response=response)
            return response

    async def _record_assistant(
        self,
        session_id: str,
        *,
        response: AgentResponse | None = None,
        error: AgentErrorResponse | None = None,
    ) -> None:
        # The answer (or error) has already been produced and checkpointed; a failure to store
        # its display copy must not turn a finished turn into an error for the caller.
        try:
            await self.sessions.append_assistant_message(session_id, response=response, error=error)
        except Exception:
            logger.exception("Could not record assistant transcript entry")

    async def _chat_turn(
        self,
        session_id: str,
        message: str,
        on_progress: ProgressCallback | None = None,
    ) -> AgentResponse:
        validate_session_id(session_id)
        if not message.strip():
            raise ValueError("Message must be non-empty")
        await self.initialize()
        if await self.sessions.get(session_id) is None:
            raise UnknownSessionError("Unknown session")
        run_id = str(uuid4())
        with self.observability.turn(
            session_id=session_id, run_id=run_id, question=message.strip()
        ) as observation:
            try:
                response, refusal_reason = await self._observed_turn(
                    session_id, message, run_id, on_progress, observation
                )
            except AgentChatError as chat_error:
                observation.failed(chat_error.error.code)
                raise
            except Exception:
                observation.failed("INTERNAL_ERROR")
                raise
            observation.answered(response, refusal_reason=refusal_reason)
            return response

    async def _observed_turn(
        self,
        session_id: str,
        message: str,
        run_id: str,
        on_progress: ProgressCallback | None,
        observation: TurnObservation,
    ) -> tuple[AgentResponse, str | None]:
        request_started = time.monotonic()
        initial: AgentState = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "messages": [HumanMessage(content=message.strip())],
            "session_id": session_id,
            "thread_id": session_id,
            "run_id": run_id,
            "user_query": message.strip(),
            "resolved_query": "",
            "artifact_requirement": None,
            "selected_skill": None,
            "skill_instructions": None,
            "retrieved_evidence": [],
            "selected_evidence": [],
            "evidence_assessment": None,
            "evidence_sufficient": False,
            "validation_error_codes": [],
            "support_assessment": None,
            "draft_answer": None,
            "answer_claims": [],
            "citations": [],
            "limitations": [],
            "available_limitations": [],
            "artifacts": [],
            "tool_calls": [],
            "errors": [],
            "retry_count": 0,
            "final_response": None,
            "trace_events": [],
            "calculations": [],
            "artifact_dataset": None,
            "table_selection": None,
            "generated_code": None,
            "execution_request": None,
            "execution_result": None,
            "artifact_descriptors": [],
            "artifact_attempt": 0,
            "artifact_errors": [],
        }
        async with self.state.checkpointer() as saver:
            graph = self.graph_factory.build(saver)
            checkpoint_id = await self.sessions.get_checkpoint_id(session_id)
            if checkpoint_id is None:
                checkpoint_id = await self._latest_successful_checkpoint(saver, session_id)
            configurable = {"thread_id": session_id}
            if checkpoint_id is not None:
                configurable["checkpoint_id"] = checkpoint_id
                prior = await self._checkpoint_state(saver, session_id, checkpoint_id)
                history = prior.get("validated_claim_history", [])
                initial["validated_claim_history"] = checkpoint_safe(
                    history if history else self.traces.recover_validated_claim_history(session_id)
                )
            config = {
                "configurable": configurable,
                "recursion_limit": self.settings.agent_max_steps,
                **callbacks_config(observation),
            }
            token = set_request_deadline(
                request_started + self.settings.agent_request_timeout_seconds
            )
            try:
                async with asyncio.timeout(self.settings.agent_request_timeout_seconds):
                    result = await self._run_graph(graph, initial, config, on_progress)
            except AgentOperationalError as operational_error:
                await self._persist_failed_trace(saver, session_id, run_id, operational_error)
                raise AgentChatError(operational_error, session_id, run_id) from operational_error
            except GraphRecursionError as cause:
                step_error = AgentOperationalError(
                    code="AGENT_STEP_LIMIT",
                    node="agent_request",
                    message="The request needed more reasoning steps than allowed. Please retry.",
                    retryable=True,
                    elapsed_seconds=time.monotonic() - request_started,
                    configured_timeout_seconds=self.settings.agent_request_timeout_seconds,
                    retry_count=0,
                    diagnostics={"max_steps": self.settings.agent_max_steps},
                )
                await self._persist_failed_trace(saver, session_id, run_id, step_error)
                raise AgentChatError(step_error, session_id, run_id) from cause
            except TimeoutError as cause:
                request_error = AgentOperationalError(
                    code="AGENT_REQUEST_TIMEOUT",
                    node="agent_request",
                    message="The agent request exceeded its processing deadline.",
                    retryable=True,
                    elapsed_seconds=time.monotonic() - request_started,
                    configured_timeout_seconds=self.settings.agent_request_timeout_seconds,
                    retry_count=0,
                )
                await self._persist_failed_trace(saver, session_id, run_id, request_error)
                raise AgentChatError(request_error, session_id, run_id) from cause
            finally:
                reset_request_deadline(token)
            successful = await self._checkpoint_for_run(saver, session_id, run_id)
            if successful is not None:
                await self.sessions.set_checkpoint_id(session_id, successful)
        state = hydrate_agent_state(result)
        response = state.get("final_response")
        if response is None:
            raise RuntimeError("Agent graph completed without a response")
        trace = RunTrace(
            session_id=session_id,
            run_id=run_id,
            events=state.get("trace_events", []),
            tool_calls=state.get("tool_calls", []),
            errors=state.get("errors", []),
            run_status="completed",
            answer_status="refused" if response.refusal else "answered",
            refusal_reason=(
                "citation_validation_failed"
                if response.refusal and state.get("validation_error_codes")
                else "artifact_generation_failed"
                if response.refusal and state.get("artifact_errors")
                else "insufficient_evidence"
                if response.refusal
                else None
            ),
        )
        await self.sessions.touch(session_id)
        self.traces.write(trace)
        return response, trace.refusal_reason

    @staticmethod
    async def _run_graph(
        graph: Any,
        initial: AgentState,
        config: dict[str, Any],
        on_progress: ProgressCallback | None,
    ) -> dict[str, Any]:
        """Equivalent to `ainvoke` (its result is the final `values` chunk), additionally
        reporting each node as it starts so clients can show live progress."""
        if on_progress is None:
            return cast(dict[str, Any], await graph.ainvoke(initial, config=config))
        result: dict[str, Any] | None = None
        async for mode, chunk in graph.astream(
            initial, config=config, stream_mode=["tasks", "values"]
        ):
            if mode == "values":
                result = chunk
            elif mode == "tasks" and "result" not in chunk and isinstance(chunk.get("name"), str):
                try:
                    await on_progress(chunk["name"])
                except Exception:
                    logger.exception("Progress callback failed")
        if result is None:
            raise RuntimeError("Agent graph produced no state")
        return result

    async def _latest_successful_checkpoint(self, saver: Any, session_id: str) -> str | None:
        agen = saver.alist({"configurable": {"thread_id": session_id}})
        try:
            async for item in agen:
                values = item.checkpoint.get("channel_values", {})
                run_id = values.get("run_id")
                if values.get("final_response") is not None and isinstance(run_id, str):
                    trace = self.traces.read(run_id)
                    if trace is not None and trace.status == "success":
                        return cast(str, item.config["configurable"]["checkpoint_id"])
        finally:
            await agen.aclose()
        return None

    async def _checkpoint_state(
        self, saver: Any, session_id: str, checkpoint_id: str
    ) -> AgentState:
        agen = saver.alist({"configurable": {"thread_id": session_id}})
        try:
            async for item in agen:
                if item.config["configurable"]["checkpoint_id"] == checkpoint_id:
                    return hydrate_agent_state(item.checkpoint.get("channel_values", {}))
        finally:
            await agen.aclose()
        return {}

    async def _checkpoint_for_run(self, saver: Any, session_id: str, run_id: str) -> str | None:
        agen = saver.alist({"configurable": {"thread_id": session_id}})
        try:
            async for item in agen:
                values = item.checkpoint.get("channel_values", {})
                if values.get("run_id") == run_id and values.get("final_response") is not None:
                    return cast(str, item.config["configurable"]["checkpoint_id"])
        finally:
            await agen.aclose()
        return None

    async def _persist_failed_trace(
        self,
        saver: Any,
        session_id: str,
        run_id: str,
        error: AgentOperationalError,
    ) -> None:
        state: AgentState = {}
        agen = saver.alist({"configurable": {"thread_id": session_id}})
        try:
            async for item in agen:
                values = hydrate_agent_state(item.checkpoint.get("channel_values", {}))
                if values.get("run_id") == run_id:
                    state = values
                    break
        finally:
            await agen.aclose()
        raw_details: dict[str, object] = {
            "error_code": error.code,
            "retry_count": error.retry_count,
            "terminal_status": "failed",
            **error.diagnostics,
        }
        if error.elapsed_seconds is not None:
            raw_details["elapsed_seconds"] = round(error.elapsed_seconds, 3)
        if error.configured_timeout_seconds is not None:
            raw_details["configured_timeout_seconds"] = error.configured_timeout_seconds
        # `error.diagnostics` is assembled at each of many different raise sites across the graph
        # (see graph.py); safe_trace_details' forbidden-substring/size-cap filter was previously
        # defined but never actually applied to a real trace, only exercised by its own unit test.
        # Applying it here means a future diagnostics field named/shaped like a secret is dropped
        # before ever reaching a persisted trace file, not just in cases someone remembered to
        # filter by hand.
        details = safe_trace_details(**raw_details)
        terminal = self.graph_factory._event("run_failed", error.node, **details)
        self.traces.write(
            RunTrace(
                session_id=session_id,
                run_id=run_id,
                events=[*state.get("trace_events", []), terminal],
                tool_calls=state.get("tool_calls", []),
                errors=[error.code],
                status="failed",
                error_code=error.code,
                run_status="failed",
                answer_status="refused",
                refusal_reason="operational_error",
            )
        )

    def get_trace(self, run_id: str) -> RunTrace | None:
        return self.traces.read(run_id)

    async def record_feedback(self, run_id: str, *, positive: bool) -> None:
        trace = self.traces.read(run_id)
        if trace is None:
            raise UnknownRunError("Unknown run")
        await asyncio.to_thread(
            self.traces.write_feedback, trace.session_id, run_id, positive=positive
        )
        try:
            self.observability.record_feedback(run_id, positive=positive)
        except Exception:
            logger.exception("Could not forward feedback to Langfuse")


def _normalized_sections(plan: ResearchPlan) -> list[Any]:
    if not plan.in_scope:
        return []
    seen: set[str] = set()
    sections = []
    for section in plan.sections:
        key = " ".join(section.question.casefold().split())
        if key and key not in seen:
            seen.add(key)
            sections.append(section)
    return sections[:MAX_RESEARCH_SECTIONS]


@lru_cache
def get_agent_service() -> AgentService:
    return AgentService.live()
