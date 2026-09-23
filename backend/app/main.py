import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from backend.app.api import router
from backend.app.config import get_settings
from backend.app.retrieval.sparse import BM25SparseEncoder


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: Literal["backend"]


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    del app
    # FastEmbed's BM25 sparse model load (a local ONNX model, no network call once cached) was
    # previously paid entirely by whichever request first needed retrieval — `/chat`'s first
    # `search_documents` call, or `/retrieval/search`'s first call now that it's cached (see
    # `_retrieval_service` in api.py) — shifting real user-facing latency onto that one request.
    # Warming it at process startup instead moves that cost to container boot. Best-effort and
    # non-fatal: get_settings() alone requires Vertex/GCP configuration this warm-up has nothing
    # to do with, and any failure here must never prevent the process from serving `/health`.
    with suppress(Exception):
        settings = get_settings()
        await asyncio.to_thread(
            BM25SparseEncoder,
            settings.sparse_embedding_model,
            cache_dir=settings.data_root / "processed" / "fastembed-cache",
        )
    yield


app = FastAPI(title="Census Insight Agent API", version="0.1.0", lifespan=_lifespan)
app.include_router(router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Report process health without contacting external dependencies."""
    return HealthResponse(status="ok", service="backend")
