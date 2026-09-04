from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient

from backend.app.config import get_settings
from backend.app.ingestion.models import IngestionReport
from backend.app.ingestion.service import IngestionService
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

router = APIRouter()


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


@router.get("/documents", response_model=list[DocumentSummary], tags=["documents"])
def documents() -> list[DocumentSummary]:
    try:
        payloads = _store().list_documents()
        return [
            DocumentSummary(
                document_id=str(payload["document_id"]),
                title=str(payload["document_title"]),
                region=str(payload["region"]),
                source_checksum=str(payload["source_checksum"]),
            )
            for payload in payloads
        ]
    except Exception as error:
        raise HTTPException(
            status_code=503, detail=f"Qdrant error: {type(error).__name__}"
        ) from error


@router.post("/admin/ingest", response_model=IngestionReport, tags=["admin"])
def ingest(request: IngestRequest) -> IngestionReport:
    settings = get_settings()
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


@router.post(
    "/retrieval/search",
    response_model=RetrievalSearchResponse,
    tags=["retrieval"],
)
def retrieval_search(request: RetrievalSearchRequest) -> RetrievalSearchResponse:
    settings = get_settings()
    service = HybridRetrievalService(
        client=QdrantClient(url=settings.qdrant_url),
        collection_name=settings.qdrant_collection,
        dense_provider=VertexEmbeddingProvider(settings),
        sparse_encoder=BM25SparseEncoder(
            settings.sparse_embedding_model,
            cache_dir=settings.data_root / "processed" / "fastembed-cache",
        ),
    )
    try:
        return service.search_response(
            query=request.query,
            document_ids=request.document_ids,
            regions=request.regions,
            top_k=request.top_k,
            debug=request.debug,
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
