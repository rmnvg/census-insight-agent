import asyncio
import json
import logging
import mimetypes
from collections.abc import AsyncIterator
from functools import lru_cache, partial
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from backend.app.agent.models import (
    AgentErrorResponse,
    AgentResponse,
    RunTrace,
    SessionContextStatus,
    SessionRecord,
    SessionSummary,
    SessionTitleUpdate,
    SessionTranscript,
)
from backend.app.agent.service import (
    AgentChatError,
    UnknownRunError,
    UnknownSessionError,
    get_agent_service,
)
from backend.app.config import get_settings
from backend.app.execution.client import EXECUTOR_HEARTBEAT_HEALTHY_SECONDS
from backend.app.execution.contracts import ArtifactDescriptor, ArtifactListing, ExecutorHealth
from backend.app.ingestion.discovery import ensure_within
from backend.app.ingestion.models import (
    CoverageLimitation,
    DocumentCoverageReport,
    DocumentCoverageSummary,
    DocumentManifest,
    DocumentPublicSummary,
    IngestionReport,
)
from backend.app.ingestion.page_images import page_count, render_page_png
from backend.app.ingestion.service import IngestionService
from backend.app.ingestion.uploads import (
    UploadError,
    UploadJob,
    UploadService,
    withdrawn_document_ids,
)
from backend.app.ingestion.worker import enqueue_upload
from backend.app.providers.embeddings import VertexEmbeddingProvider
from backend.app.retrieval.models import (
    DocumentSummary,
    QdrantHealthResponse,
    RetrievalSearchRequest,
    RetrievalSearchResponse,
)
from backend.app.retrieval.qdrant_store import QdrantStore
from backend.app.retrieval.service import HybridRetrievalService
from backend.app.retrieval.sparse import BM25SparseEncoder
from backend.app.trust import Scorecard

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str
    # No prior upper bound existed at all: a message of any size reached classify/resolve/
    # synthesize model calls (and could be echoed back into every later turn's context) at full
    # cost. 4000 characters comfortably covers a detailed question or multi-part comparison
    # request while bounding the worst case; `AgentService._chat_turn`'s own non-empty check
    # stays as the guard for direct programmatic callers that bypass this HTTP boundary (e.g.
    # tests constructing `AgentService` directly).
    message: str = Field(min_length=1, max_length=4000)


SESSION_EXAMPLE = {
    "session_id": "6e594f6e52754ef5932df652638e3f0d",
    "created_at": "2026-01-01T12:00:00Z",
    "updated_at": "2026-01-01T12:00:00Z",
}

CHAT_SUCCESS_EXAMPLE = {
    "answer": "Karnataka's literacy rate in 2011 was 75.36 percent.",
    "claims": [
        {
            "claim_id": "claim-1",
            "text": "Karnataka's literacy rate in 2011 was 75.36 percent.",
            "citation_ids": ["citation-1"],
            "document_derived": True,
            "metric": "literacy rate",
            "region": "Karnataka",
            "year": 2011,
            "population_scope": "Persons",
            "residence_scope": "Total",
            "value": 75.36,
            "unit": "percent",
            "derivation": None,
        }
    ],
    "citations": [
        {
            "citation_id": "citation-1",
            "document_id": "census-2011-karnataka-pca-highlights",
            "document_title": "Primary Census Abstract Data Highlights: Karnataka",
            "page_number": 50,
            "snippet": "Karnataka  Persons  Total  75.36",
            "chunk_id": "f725734b-65b8-58e2-9e50-ad0d4cc41e27",
            "section_path": ["Literacy rate"],
            "evidence_span": {
                "evidence_id": "f725734b-65b8-58e2-9e50-ad0d4cc41e27",
                "start_offset": 100,
                "end_offset": 133,
            },
        }
    ],
    "artifacts": [],
    "limitations": [],
    "refusal": False,
    "trace_id": "bd52f034-88f2-4c4d-8a5f-938ebc0d0ed8",
}

CHAT_REFUSAL_EXAMPLE = {
    "answer": "I can only answer from the supplied Census documents.",
    "claims": [],
    "citations": [],
    "artifacts": [],
    "limitations": ["The requested topic is outside the supplied document collection."],
    "refusal": True,
    "trace_id": "dd147a1f-9119-4753-845b-c23de579e31c",
}

ERROR_EXAMPLE = {
    "error_code": "EVIDENCE_ASSESSMENT_TIMEOUT",
    "message": "Evidence assessment timed out. Please retry the request.",
    "session_id": SESSION_EXAMPLE["session_id"],
    "trace_id": "e8bf0d0e-990d-44c2-9796-eeb13c59ec18",
    "retryable": True,
}


@router.post(
    "/sessions",
    response_model=SessionRecord,
    responses={200: {"content": {"application/json": {"example": SESSION_EXAMPLE}}}},
    tags=["agent"],
)
async def create_session() -> SessionRecord:
    return await get_agent_service().create_session()


@router.get("/sessions/{session_id}", response_model=SessionRecord, tags=["agent"])
async def get_session(session_id: str) -> SessionRecord:
    try:
        session = await get_agent_service().get_session(session_id)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session")
    return session


@router.get("/sessions/{session_id}/context", response_model=SessionContextStatus, tags=["agent"])
async def get_session_context(session_id: str) -> SessionContextStatus:
    try:
        return await get_agent_service().get_context_status(session_id)
    except UnknownSessionError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/sessions", response_model=list[SessionSummary], tags=["agent"])
async def list_sessions(
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> list[SessionSummary]:
    """Sessions with at least one message, most recently active first."""
    return await get_agent_service().list_sessions(limit)


@router.get("/sessions/{session_id}/messages", response_model=SessionTranscript, tags=["agent"])
async def get_session_messages(session_id: str) -> SessionTranscript:
    try:
        return await get_agent_service().get_transcript(session_id)
    except UnknownSessionError as error:
        raise HTTPException(status_code=404, detail="Unknown session") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.patch("/sessions/{session_id}", response_model=SessionRecord, tags=["agent"])
async def rename_session(session_id: str, update: SessionTitleUpdate) -> SessionRecord:
    try:
        return await get_agent_service().rename_session(session_id, update.title)
    except UnknownSessionError as error:
        raise HTTPException(status_code=404, detail="Unknown session") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.delete("/sessions/{session_id}", status_code=204, tags=["agent"])
async def delete_session(session_id: str) -> Response:
    try:
        await get_agent_service().delete_session(session_id)
    except UnknownSessionError as error:
        raise HTTPException(status_code=404, detail="Unknown session") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return Response(status_code=204)


@router.get(
    "/sessions/{session_id}/artifacts",
    response_model=ArtifactListing,
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "artifacts": [
                            {
                                "artifact_id": "9027752e-7018-4591-b29e-6f63c3f9abaf",
                                "artifact_type": "chart",
                                "title": "Karnataka and Odisha literacy rates",
                                "filename": "chart.png",
                                "media_type": "image/png",
                                "byte_size": 48123,
                                "sha256": "0" * 64,
                                "session_id": SESSION_EXAMPLE["session_id"],
                                "run_id": "bd52f034-88f2-4c4d-8a5f-938ebc0d0ed8",
                                "source_manifest_path": "source-manifest.json",
                                "download_url": (
                                    "/sessions/example/artifacts/example/files/chart.png"
                                ),
                            }
                        ]
                    }
                }
            }
        }
    },
    tags=["artifacts"],
)
async def list_artifacts(session_id: str) -> ArtifactListing:
    service = get_agent_service()
    try:
        if await service.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Not found")
        return service.artifacts.list(session_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="Not found") from error


@router.get(
    "/sessions/{session_id}/artifacts/{artifact_id}",
    response_model=ArtifactDescriptor,
    tags=["artifacts"],
)
async def get_artifact(session_id: str, artifact_id: str) -> ArtifactDescriptor:
    service = get_agent_service()
    try:
        if await service.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Not found")
        artifact = service.artifacts.get(session_id, artifact_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="Not found") from error
    if artifact is None:
        raise HTTPException(status_code=404, detail="Not found")
    return artifact


@router.get(
    "/sessions/{session_id}/artifacts/{artifact_id}/files/{filename}",
    responses={
        200: {
            "description": "Allowlisted validated artifact file",
            "content": {
                "image/png": {"schema": {"type": "string", "format": "binary"}},
                "text/csv": {"example": "Region,Literacy rate\nKarnataka,75.36\n"},
                "text/markdown": {"example": "| Region | Literacy rate |\n|---|---:|"},
                "application/json": {
                    "example": {"artifact_id": "9027752e-7018-4591-b29e-6f63c3f9abaf"}
                },
            },
        }
    },
    tags=["artifacts"],
)
async def download_artifact_file(session_id: str, artifact_id: str, filename: str) -> FileResponse:
    service = get_agent_service()
    try:
        if await service.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Not found")
        path = service.artifacts.public_file(session_id, artifact_id, filename)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Not found") from error
    if path is None:
        raise HTTPException(status_code=404, detail="Not found")
    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=filename)


@router.get("/evaluation/scorecard", response_model=Scorecard, tags=["evaluation"])
def trust_scorecard() -> Scorecard:
    """The latest trust benchmark run: a locally generated one, else the one shipped in evals/."""
    candidates = [
        get_settings().data_root / "processed" / "trust-scorecard.json",
        Path("evals") / "trust-scorecard.json",
    ]
    for path in candidates:
        if path.is_file():
            try:
                return Scorecard.model_validate_json(path.read_text(encoding="utf-8"))
            except ValueError as error:
                raise HTTPException(status_code=503, detail="Scorecard is unreadable") from error
    raise HTTPException(status_code=404, detail="No trust scorecard has been generated yet")


@router.get("/health/executor", response_model=ExecutorHealth, tags=["health"])
def executor_health() -> ExecutorHealth:
    age = get_agent_service().execution_queue.heartbeat_age_seconds()
    healthy = age is not None and age < EXECUTOR_HEARTBEAT_HEALTHY_SECONDS
    return ExecutorHealth(
        status="ok" if healthy else "error",
        detail="Executor heartbeat is current" if healthy else "Executor heartbeat is unavailable",
        heartbeat_age_seconds=round(age, 3) if age is not None else None,
    )


_ERROR_CODE_STATUS: dict[str, int] = {
    "INTERNAL_PROVENANCE_INVALID": 500,
    "MODEL_TIMEOUT": 504,
    "EVIDENCE_ASSESSMENT_TIMEOUT": 504,
    "AGENT_REQUEST_TIMEOUT": 504,
    "MODEL_RATE_LIMITED": 503,
    "MODEL_UNAVAILABLE": 503,
    "EXECUTOR_UNAVAILABLE": 503,
}
_DEFAULT_ERROR_STATUS = 502


@router.post(
    "/chat",
    response_model=AgentResponse,
    responses={
        200: {
            "content": {
                "application/json": {
                    "examples": {
                        "success": {"value": CHAT_SUCCESS_EXAMPLE},
                        "refusal": {"value": CHAT_REFUSAL_EXAMPLE},
                    }
                }
            }
        },
        500: {
            "model": AgentErrorResponse,
            "content": {"application/json": {"example": ERROR_EXAMPLE}},
        },
        502: {
            "model": AgentErrorResponse,
            "content": {"application/json": {"example": ERROR_EXAMPLE}},
        },
        503: {
            "model": AgentErrorResponse,
            "content": {"application/json": {"example": ERROR_EXAMPLE}},
        },
        504: {
            "model": AgentErrorResponse,
            "content": {"application/json": {"example": ERROR_EXAMPLE}},
        },
    },
    tags=["agent"],
)
async def chat(request: ChatRequest) -> AgentResponse | JSONResponse:
    try:
        return await get_agent_service().chat(request.session_id, request.message)
    except UnknownSessionError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except AgentChatError as error:
        payload = AgentErrorResponse(
            error_code=error.error.code,
            message=error.error.message,
            session_id=error.session_id,
            trace_id=error.run_id,
            retryable=error.error.retryable,
        )
        status_code = _ERROR_CODE_STATUS.get(error.error.code, _DEFAULT_ERROR_STATUS)
        return JSONResponse(status_code=status_code, content=payload.model_dump())


# Human-readable progress labels for LangGraph nodes, shown while a turn is running.
PROGRESS_LABELS: dict[str, str] = {
    "load_memory": "Loading validated conversation memory",
    "classify_task": "Understanding the question",
    "resolve_query": "Resolving follow-up references",
    "plan": "Planning retrieval",
    "load_skill": "Loading task skill",
    "call_tools": "Searching the Census reports",
    "assess_evidence": "Assessing evidence relevance",
    "synthesize": "Drafting a cited answer",
    "prepare_artifact": "Hydrating verified data for the artifact",
    "generate_artifact_code": "Writing chart/table code",
    "execute_artifact": "Running code in the isolated executor",
    "inspect_artifact": "Inspecting generated artifact",
    "repair_artifact": "Repairing artifact code",
    "validate_citations": "Validating every citation against the source",
    "repair": "Repairing an unsupported claim",
    "graceful_response": "Preparing a safe response",
    "persist_result": "Saving validated memory",
}
_SSE_KEEPALIVE_SECONDS = 15.0
# Turns keep running after a client disconnects so their result still lands in the transcript.
_background_turns: set[asyncio.Task[None]] = set()


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post(
    "/chat/stream",
    tags=["agent"],
    responses={
        200: {
            "description": (
                "Server-sent events: `progress` ({node, label}) while the graph runs, then "
                "exactly one `result` (the /chat response body) or `error` (AgentErrorResponse)."
            ),
            "content": {"text/event-stream": {}},
        }
    },
)
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    service = get_agent_service()
    try:
        if await service.get_session(request.session_id) is None:
            raise HTTPException(status_code=404, detail="Unknown session")
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def progress(node: str) -> None:
        queue.put_nowait(_sse("progress", {"node": node, "label": PROGRESS_LABELS.get(node, node)}))

    async def run_turn() -> None:
        try:
            response = await service.chat(request.session_id, request.message, on_progress=progress)
            queue.put_nowait(_sse("result", response.model_dump(mode="json", by_alias=True)))
        except AgentChatError as error:
            payload = AgentErrorResponse(
                error_code=error.error.code,
                message=error.error.message,
                session_id=error.session_id,
                trace_id=error.run_id,
                retryable=error.error.retryable,
            )
            queue.put_nowait(_sse("error", payload.model_dump()))
        except ValueError as error:
            queue.put_nowait(
                _sse("error", {"error_code": "INVALID_REQUEST", "message": str(error)})
            )
        except Exception:
            logger.exception("Streaming chat turn failed")
            queue.put_nowait(
                _sse(
                    "error",
                    {
                        "error_code": "INTERNAL_ERROR",
                        "message": "The request could not be completed. Please retry.",
                        "retryable": True,
                    },
                )
            )
        finally:
            queue.put_nowait(None)

    task = asyncio.create_task(run_turn())
    _background_turns.add(task)
    task.add_done_callback(_background_turns.discard)

    return StreamingResponse(
        _drain(queue, {"session_id": request.session_id}),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class ResearchRequest(BaseModel):
    session_id: str
    topic: str = Field(min_length=3, max_length=500)


@router.post(
    "/research/stream",
    tags=["agent"],
    responses={
        200: {
            "description": (
                "Server-sent events: `plan`, then per section `section_started`, `progress` "
                "({index, node, label}) and `section_done`, then one `result` (ResearchReport) "
                "or `error`."
            ),
            "content": {"text/event-stream": {}},
        }
    },
)
async def research_stream(request: ResearchRequest) -> StreamingResponse:
    """Deep research: plan questions, answer each with a validated agent turn, stream it all."""
    service = get_agent_service()
    try:
        if await service.get_session(request.session_id) is None:
            raise HTTPException(status_code=404, detail="Unknown session")
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def emit(event: str, data: dict[str, object]) -> None:
        if event == "progress":
            node = str(data["node"])
            data = {**data, "label": PROGRESS_LABELS.get(node, node)}
        queue.put_nowait(_sse(event, data))

    async def run() -> None:
        try:
            report = await service.research(request.session_id, request.topic, emit=emit)
            queue.put_nowait(_sse("result", report.model_dump(mode="json", by_alias=True)))
        except AgentChatError as error:
            queue.put_nowait(
                _sse(
                    "error",
                    {
                        "error_code": error.error.code,
                        "message": error.error.message,
                        "retryable": error.error.retryable,
                    },
                )
            )
        except ValueError as error:
            queue.put_nowait(
                _sse("error", {"error_code": "INVALID_REQUEST", "message": str(error)})
            )
        except Exception:
            logger.exception("Research run failed")
            queue.put_nowait(
                _sse(
                    "error",
                    {
                        "error_code": "INTERNAL_ERROR",
                        "message": "The research brief could not be completed. Please retry.",
                        "retryable": True,
                    },
                )
            )
        finally:
            queue.put_nowait(None)

    task = asyncio.create_task(run())
    _background_turns.add(task)
    task.add_done_callback(_background_turns.discard)
    return StreamingResponse(
        _drain(queue, {"session_id": request.session_id}),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _drain(queue: asyncio.Queue[str | None], started: dict[str, Any]) -> AsyncIterator[str]:
    yield _sse("started", started)
    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=_SSE_KEEPALIVE_SECONDS)
        except TimeoutError:
            yield ": keepalive\n\n"
            continue
        if item is None:
            return
        yield item


@router.get(
    "/runs/{run_id}/trace",
    response_model=RunTrace,
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "session_id": SESSION_EXAMPLE["session_id"],
                        "run_id": "bd52f034-88f2-4c4d-8a5f-938ebc0d0ed8",
                        "events": [
                            {
                                "timestamp": "2026-01-01T12:00:01Z",
                                "event": "retrieval_completed",
                                "node": "retrieve",
                                "details": {"candidate_count": 10},
                                "latency_ms": 84.2,
                            }
                        ],
                        "tool_calls": [],
                        "errors": [],
                        "status": "success",
                        "error_code": None,
                        "run_status": "completed",
                        "answer_status": "answered",
                        "refusal_reason": None,
                    }
                }
            }
        }
    },
    tags=["agent"],
)
def run_trace(run_id: str) -> RunTrace:
    try:
        trace = get_agent_service().get_trace(run_id)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if trace is None:
        raise HTTPException(status_code=404, detail="Unknown run")
    return trace


class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"]


@router.post("/runs/{run_id}/feedback", status_code=204, tags=["agent"])
async def run_feedback(run_id: str, feedback: FeedbackRequest) -> Response:
    """Rate an answer. Stored beside the run's trace and, when enabled, sent to Langfuse."""
    try:
        await get_agent_service().record_feedback(run_id, positive=feedback.rating == "up")
    except ValueError as error:
        raise HTTPException(status_code=422, detail="Invalid run ID") from error
    except UnknownRunError as error:
        raise HTTPException(status_code=404, detail="Unknown run") from error
    return Response(status_code=204)


class IngestRequest(BaseModel):
    source_dir: Path | None = None
    document_id: str | None = None
    dry_run: bool = False
    rebuild: bool = False
    review_override: bool = False
    report_output: Path | None = None


def _store() -> QdrantStore:
    settings = get_settings()
    return QdrantStore(
        QdrantClient(url=settings.qdrant_url),
        settings.qdrant_collection,
        dense_dimensions=settings.gemini_embedding_dimension,
        dense_model=settings.gemini_embedding_model,
        sparse_model=settings.sparse_embedding_model,
    )


def _public_documents() -> list[DocumentPublicSummary]:
    """Read portable public metadata without scrolling every vector payload."""
    generated = get_settings().data_root / "manifests" / "generated"
    documents: list[DocumentPublicSummary] = []
    try:
        paths = sorted(generated.glob("*.json"))
        for path in paths:
            manifest = DocumentManifest.model_validate_json(path.read_text(encoding="utf-8"))
            documents.append(
                DocumentPublicSummary(
                    document_id=manifest.document_id,
                    title=manifest.title,
                    region=manifest.region,
                    source_checksum=manifest.source_checksum,
                )
            )
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail="Document catalog unavailable") from error
    return documents


@router.get("/health/qdrant", response_model=QdrantHealthResponse, tags=["health"])
def qdrant_health() -> QdrantHealthResponse:
    settings = get_settings()
    try:
        store = _store()
        exists = store.client.collection_exists(settings.qdrant_collection)
        if exists:
            store.validate_schema()
        detail = "Collection is available" if exists else "Qdrant is reachable; collection absent"
        return QdrantHealthResponse(
            status="ok", collection=settings.qdrant_collection, detail=detail
        )
    except Exception as error:
        return QdrantHealthResponse(
            status="error",
            collection=settings.qdrant_collection,
            detail=f"Qdrant unavailable: {type(error).__name__}",
        )


@router.get(
    "/documents",
    response_model=list[DocumentSummary],
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": [
                        {
                            "document_id": "census-2011-karnataka-pca-highlights",
                            "title": "Primary Census Abstract Data Highlights: Karnataka",
                            "region": "Karnataka",
                            "source_checksum": "0" * 64,
                        }
                    ]
                }
            }
        }
    },
    tags=["documents"],
)
def documents() -> list[DocumentSummary]:
    return [DocumentSummary.model_validate(item.model_dump()) for item in _public_documents()]


@router.get(
    "/documents/{document_id}/coverage",
    response_model=DocumentCoverageSummary,
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "document": {
                            "document_id": "census-2011-karnataka-pca-highlights",
                            "title": "Primary Census Abstract Data Highlights: Karnataka",
                            "region": "Karnataka",
                            "source_checksum": "0" * 64,
                        },
                        "coverage": {
                            "document_id": "census-2011-karnataka-pca-highlights",
                            "pdf_page_count": 82,
                            "indexed_pages": 69,
                            "blank_decorative_pages": 1,
                            "excluded_visual_pages": 12,
                            "failed_mappings": 0,
                            "percentage_pages_indexed": 84.15,
                            "page_lists_by_status": {"excluded_unverified_visual": [13]},
                            "pages": [
                                {
                                    "page_number": 13,
                                    "status": "excluded_unverified_visual",
                                    "reason": "Reviewed visual page",
                                }
                            ],
                        },
                        "limitations": [
                            {
                                "document_id": "census-2011-karnataka-pca-highlights",
                                "excluded_pages": [13],
                                "statuses": ["excluded_unverified_visual"],
                                "message": (
                                    "Some visual pages were excluded from automated answering."
                                ),
                            }
                        ],
                    }
                }
            }
        }
    },
    tags=["documents"],
)
def document_coverage(document_id: str) -> DocumentCoverageSummary:
    settings = get_settings()
    document = next(
        (item for item in _public_documents() if item.document_id == document_id),
        None,
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Unknown document")
    report_path = settings.data_root / "processed" / "dry-run-report.json"
    if not report_path.is_file():
        report_path = settings.data_root / "processed" / "idempotency-report.json"
    upload_report = UploadService.report_path_for(settings.data_root, document_id)
    if upload_report is not None:
        report_path = upload_report
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        record = next(
            item for item in report.get("documents", []) if item.get("document_id") == document_id
        )
        coverage = DocumentCoverageReport.model_validate(record["coverage"])
        limitations = [
            CoverageLimitation.model_validate(item) for item in record.get("limitations", [])
        ]
    except (OSError, ValueError, KeyError, StopIteration, json.JSONDecodeError) as error:
        raise HTTPException(status_code=503, detail="Coverage report unavailable") from error
    return DocumentCoverageSummary(
        document=DocumentPublicSummary(
            document_id=document_id,
            title=document.title,
            region=document.region,
            source_checksum=document.source_checksum,
        ),
        coverage=coverage,
        limitations=limitations,
    )


@lru_cache
def get_upload_service() -> UploadService:
    settings = get_settings()
    return UploadService(
        settings,
        point_remover=lambda document_id: _store().delete_document(document_id),
        # With a durable worker, queued jobs belong to the queue, not to this process.
        recover_interrupted=not settings.ingestion_broker_url,
    )


def _require_uploads_enabled() -> UploadService:
    if not get_settings().document_upload_enabled:
        raise HTTPException(status_code=403, detail="Document upload is disabled")
    return get_upload_service()


@router.post("/documents/upload", response_model=UploadJob, status_code=202, tags=["documents"])
async def upload_document(
    background: BackgroundTasks,
    file: Annotated[UploadFile, File(description="A text-layer PDF")],
    title: Annotated[str, Form(min_length=1, max_length=200)],
    region: Annotated[str, Form(min_length=1, max_length=80)],
) -> UploadJob:
    """Validate and stage a PDF, then index it in the background (poll the returned job)."""
    uploads = _require_uploads_enabled()
    limit = get_settings().document_upload_max_bytes
    content = await file.read(limit + 1)
    try:
        job = await asyncio.to_thread(
            uploads.stage,
            filename=file.filename or "document.pdf",
            content=content,
            title=title,
            region=region,
        )
    except UploadError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
    if not get_settings().ingestion_broker_url:
        background.add_task(uploads.process, job.job_id)
        return job
    try:
        await asyncio.to_thread(enqueue_upload, job.job_id)
    except Exception as error:
        await asyncio.to_thread(uploads.fail, job.job_id, "The indexing queue was unavailable.")
        raise HTTPException(
            status_code=503, detail="The indexing queue is unavailable; please retry."
        ) from error
    return job


@router.get("/documents/uploads", response_model=list[UploadJob], tags=["documents"])
def list_uploads() -> list[UploadJob]:
    return get_upload_service().list_jobs()


@router.get("/documents/uploads/{job_id}", response_model=UploadJob, tags=["documents"])
def get_upload(job_id: str) -> UploadJob:
    job = get_upload_service().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown upload job")
    return job


@router.delete("/documents/{document_id}", status_code=204, tags=["documents"])
async def delete_document(document_id: str) -> Response:
    uploads = _require_uploads_enabled()
    try:
        await uploads.delete_document(document_id)
    except UploadError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
    return Response(status_code=204)


def _document_pdf(document_id: str) -> Path:
    settings = get_settings()
    path = settings.data_root / "manifests" / "generated" / f"{document_id}.json"
    generated = (settings.data_root / "manifests" / "generated").resolve()
    try:
        manifest_path = ensure_within(path, generated)
        manifest = DocumentManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        pdf = ensure_within(settings.data_root / manifest.pdf_path, settings.data_root / "source")
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Unknown document") from error
    if not pdf.is_file():
        raise HTTPException(status_code=404, detail="Source PDF unavailable")
    return pdf


@router.get(
    "/documents/{document_id}/pages/{page_number}",
    tags=["documents"],
    responses={200: {"content": {"image/png": {}}, "description": "Rendered PDF page"}},
)
async def document_page_image(
    document_id: str,
    page_number: int,
    highlight: Annotated[str | None, Query(max_length=2000)] = None,
) -> Response:
    """Render a physical PDF page so a citation can be checked against the original.

    `highlight` is the citation snippet; matching text is highlighted when it locates
    unambiguously. Headers: `X-Page-Count` (document page count) and `X-Highlight`
    (`matched`, `unmatched`, `no_text_layer`, or `none`).
    """
    pdf = _document_pdf(document_id)
    try:
        (content, status), pages = await asyncio.to_thread(
            lambda: (render_page_png(pdf, page_number, highlight), page_count(pdf))
        )
    except IndexError as error:
        raise HTTPException(status_code=404, detail="Page out of range") from error
    return Response(
        content,
        media_type="image/png",
        headers={
            "Cache-Control": "private, max-age=3600",
            "X-Page-Count": str(pages),
            "X-Highlight": status,
        },
    )


@router.post("/admin/ingest", response_model=IngestionReport, tags=["admin"])
def ingest(request: IngestRequest) -> IngestionReport:
    settings = get_settings()
    if not settings.admin_ingestion_enabled:
        raise HTTPException(status_code=403, detail="HTTP ingestion is disabled; use the CLI")
    service = IngestionService(settings) if request.dry_run else IngestionService.live(settings)
    try:
        return service.ingest(
            source_dir=request.source_dir or settings.data_root / "source",
            document_id=request.document_id,
            dry_run=request.dry_run,
            rebuild=request.rebuild,
            review_override=request.review_override,
            report_output=request.report_output,
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@lru_cache
def _retrieval_service() -> HybridRetrievalService:
    """Cached singleton, matching `get_agent_service()`'s pattern below.

    Every prior call built a fresh Qdrant client, Vertex embedder, and — most notably — a fresh
    `BM25SparseEncoder`, which loads a local FastEmbed ONNX model, on every single request to this
    endpoint.
    """
    settings = get_settings()
    return HybridRetrievalService(
        client=QdrantClient(url=settings.qdrant_url),
        collection_name=settings.qdrant_collection,
        dense_provider=VertexEmbeddingProvider(settings),
        sparse_encoder=BM25SparseEncoder(
            settings.sparse_embedding_model,
            cache_dir=settings.data_root / "processed" / "fastembed-cache",
        ),
        excluded_document_ids=partial(withdrawn_document_ids, settings.data_root),
    )


@router.post(
    "/retrieval/search",
    response_model=RetrievalSearchResponse,
    tags=["retrieval"],
)
def retrieval_search(request: RetrievalSearchRequest) -> RetrievalSearchResponse:
    service = _retrieval_service()
    try:
        return service.search_response(
            query=request.query,
            document_ids=request.document_ids,
            regions=request.regions,
            top_k=request.top_k,
            debug=request.debug,
        )
    except ValueError as error:
        # The only ValueError `search_response` raises itself ("Retrieval query must be
        # non-empty") is a safe, application-level validation message.
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        # Anything else (Qdrant connectivity, a malformed stored payload via
        # RetrievalProvenanceError, ...) is an internal failure; report only its type, the same
        # sanitization `qdrant_health` already uses, rather than the raw exception text — which
        # for a connectivity failure could include the internal Qdrant URL/port.
        raise HTTPException(
            status_code=502, detail=f"Retrieval unavailable: {type(error).__name__}"
        ) from error
