import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.types import ASGIApp, Receive, Scope, Send

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


_UPLOAD_PATH = "/documents/upload"
# Multipart framing and form fields on top of the PDF itself.
_UPLOAD_OVERHEAD_BYTES = 1024 * 1024


class UploadSizeLimit:
    """Reject oversized uploads from the Content-Length header before any body is read.

    Starlette spools a multipart file to disk while parsing the form, which happens before the
    endpoint can check the size; without this, an arbitrarily large body would be written first.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == _UPLOAD_PATH:
            headers = dict(scope["headers"])
            declared = headers.get(b"content-length")
            status, detail = None, ""
            if declared is None or not declared.isdigit():
                status, detail = 411, "Content-Length is required for uploads"
            else:
                limit = _upload_limit_bytes()
                if int(declared) > limit + _UPLOAD_OVERHEAD_BYTES:
                    status = 413
                    detail = f"PDF exceeds the {limit // (1024 * 1024)} MB upload limit."
            if status is not None:
                body = json.dumps({"detail": detail}).encode()
                await send(
                    {
                        "type": "http.response.start",
                        "status": status,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


def _upload_limit_bytes() -> int:
    try:
        return get_settings().document_upload_max_bytes
    except Exception:
        return 50 * 1024 * 1024


app = FastAPI(title="Census Insight Agent API", version="0.1.0", lifespan=_lifespan)
app.add_middleware(UploadSizeLimit)
app.include_router(router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Report process health without contacting external dependencies."""
    return HealthResponse(status="ok", service="backend")
