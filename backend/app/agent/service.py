import asyncio
import logging
import shutil
import time
from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import Any, cast
from uuid import uuid4
from weakref import WeakValueDictionary

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
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
    RunTrace,
    SessionContextStatus,
    SessionRecord,
    SessionSummary,
    SessionTranscript,
)
from backend.app.agent.persistence import (
    SessionStore,
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
from backend.app.agent.tools import AgentTools
from backend.app.config import Settings, get_settings
from backend.app.execution.client import ArtifactStore, ExecutionQueueClient
from backend.app.providers.chat import get_chat_model
from backend.app.providers.embeddings import VertexEmbeddingProvider
from backend.app.retrieval.qdrant_store import QdrantStore
from backend.app.retrieval.service import HybridRetrievalService
from backend.app.retrieval.sparse import BM25SparseEncoder

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], Awaitable[None]]


class UnknownSessionError(ValueError):
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
        self._session_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
        self.sessions = SessionStore(settings.workspace_root)
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
        await self.sessions.initialize()

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
        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            async with AsyncSqliteSaver.from_conn_string(str(self.sessions.database)) as saver:
                await saver.adelete_thread(session_id)
            await self.sessions.delete(session_id)
            sessions_root = (self.settings.workspace_root / "sessions").resolve()
            session_dir = (sessions_root / session_id).resolve()
            if session_dir.parent == sessions_root and session_dir.is_dir():
                await asyncio.to_thread(shutil.rmtree, session_dir)

    async def get_context_status(self, session_id: str) -> SessionContextStatus:
        validate_session_id(session_id)
        await self.initialize()
        if await self.sessions.get(session_id) is None:
            raise UnknownSessionError("Unknown session")
        history = []
        async with AsyncSqliteSaver.from_conn_string(str(self.sessions.database)) as saver:
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
        # Compose runs one API process. Hold the lock through checkpoint and trace persistence;
        # weak references release idle session locks without an ever-growing session registry.
        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
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
            "generated_code": None,
            "execution_request": None,
            "execution_result": None,
            "artifact_descriptors": [],
            "artifact_attempt": 0,
            "artifact_errors": [],
        }
        async with AsyncSqliteSaver.from_conn_string(str(self.sessions.database)) as saver:
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
        return response

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


@lru_cache
def get_agent_service() -> AgentService:
    return AgentService.live()
