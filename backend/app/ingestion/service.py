import hashlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pymupdf
from qdrant_client import QdrantClient

from backend.app.config import Settings
from backend.app.ingestion.chunking import chunk_pages
from backend.app.ingestion.coverage import (
    coverage_limitation,
    failed_coverage_report,
    load_coverage_decisions,
    load_manual_transcriptions,
)
from backend.app.ingestion.discovery import discover_documents, ensure_within, source_checksum
from backend.app.ingestion.embeddings import CachedDenseEmbedder, EmbeddingCache
from backend.app.ingestion.errors import PageMappingError
from backend.app.ingestion.models import (
    CoverageLimitation,
    DocumentCoverageReport,
    DocumentIngestionResult,
    DocumentManifest,
    DocumentPage,
    DocumentPair,
    IndexedExtractionMethod,
    IngestionReport,
    ManualTranscriptionRecord,
    PageCoverageDecision,
)
from backend.app.ingestion.pages import map_document_pages
from backend.app.ingestion.safety import UnsafeExtractionError, validate_chunks_for_indexing
from backend.app.providers.embeddings import VertexEmbeddingProvider
from backend.app.retrieval.qdrant_store import QdrantStore
from backend.app.retrieval.sparse import BM25SparseEncoder


class IngestionService:
    """Orchestrate citation-safe extraction, chunking, embedding, and storage."""

    def __init__(
        self,
        settings: Settings,
        *,
        store: QdrantStore | None = None,
        dense_embedder: CachedDenseEmbedder | None = None,
        sparse_encoder: BM25SparseEncoder | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.dense_embedder = dense_embedder
        self.sparse_encoder = sparse_encoder

    @classmethod
    def live(cls, settings: Settings) -> "IngestionService":
        cache = EmbeddingCache(settings.data_root / "processed" / "embedding-cache.sqlite3")
        dense = CachedDenseEmbedder(VertexEmbeddingProvider(settings), cache)
        sparse = BM25SparseEncoder(
            settings.sparse_embedding_model,
            cache_dir=settings.data_root / "processed" / "fastembed-cache",
        )
        store = QdrantStore(
            QdrantClient(url=settings.qdrant_url),
            settings.qdrant_collection,
            dense_dimensions=settings.gemini_embedding_dimension,
            dense_model=settings.gemini_embedding_model,
            sparse_model=settings.sparse_embedding_model,
        )
        return cls(settings, store=store, dense_embedder=dense, sparse_encoder=sparse)

    def ingest(
        self,
        *,
        source_dir: Path,
        document_id: str | None = None,
        dry_run: bool = False,
        rebuild: bool = False,
        review_override: bool = False,
        report_output: Path | None = None,
    ) -> IngestionReport:
        started = time.monotonic()
        configured_source = (self.settings.data_root / "source").resolve()
        requested_source = source_dir.resolve()
        if any(
            part.casefold() in {"page-review", "ocr", "ocr-annotated"}
            for part in requested_source.parts
        ):
            raise UnsafeExtractionError(
                "Page-review and raw OCR directories are not ingestion sources"
            )
        source_root = ensure_within(source_dir, configured_source)
        manifest_dir = (self.settings.data_root / "manifests").resolve()
        pairs = discover_documents(source_root, manifest_dir)
        if document_id:
            pairs = [pair for pair in pairs if pair.document_id == document_id]

        report = IngestionReport(discovered_documents=len(pairs), dry_run=dry_run)
        if not dry_run:
            store, _, _ = self._require_live_components()
            store.ensure_collection(rebuild=rebuild)

        for pair in pairs:
            result = DocumentIngestionResult(
                document_id=pair.document_id,
                pdf_filename=pair.pdf_path.name,
                markdown_filename=pair.markdown_path.name if pair.markdown_path else None,
                paired=pair.markdown_path is not None,
            )
            decisions: dict[int, PageCoverageDecision] = {}
            transcriptions: dict[int, ManualTranscriptionRecord] = {}
            try:
                checksum = source_checksum(pair.pdf_path)
                decisions = load_coverage_decisions(manifest_dir, pair.document_id, checksum)
                transcriptions = load_manual_transcriptions(
                    manifest_dir, pair.document_id, checksum
                )
                mapping = map_document_pages(
                    document_id=pair.document_id,
                    pdf_path=pair.pdf_path,
                    markdown_path=pair.markdown_path,
                    review_override=review_override,
                    coverage_decisions=decisions,
                    manual_transcriptions=transcriptions,
                )
                chunks = chunk_pages(
                    mapping.pages,
                    document_title=pair.title,
                    region=pair.region,
                    source_checksum=checksum,
                )
                validate_chunks_for_indexing(chunks)
                content_fingerprint = hashlib.sha256(
                    json.dumps(
                        [chunk.model_dump(mode="json") for chunk in chunks],
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                result.pages = len(mapping.pages)
                result.chunks = len(chunks)
                result.provided_markdown_pages = mapping.markdown_pages
                result.fallback_pages = mapping.fallback_pages
                result.unresolved_pages = mapping.unresolved_pages
                result.duplicate_pages = mapping.duplicate_pages
                result.empty_pages = mapping.empty_pages
                result.coverage = mapping.coverage
                result.limitations = self.document_coverage_limitations(mapping.coverage)
                result.excluded_pages = sorted(
                    mapping.coverage.page_lists_by_status["excluded_blank"]
                    + mapping.coverage.page_lists_by_status["excluded_decorative"]
                    + mapping.coverage.page_lists_by_status["excluded_unverified_visual"]
                )
                result.page_count_mismatch = mapping.page_count_mismatch
                result.citation_ready_chunks = sum(
                    1
                    for chunk in chunks
                    if chunk.metadata.document_id
                    and chunk.metadata.page_number > 0
                    and chunk.metadata.citation_snippet
                    and chunk.metadata.citation_snippet in chunk.text
                    and chunk.metadata.source_checksum
                )
                result.chunks_missing_provenance = len(chunks) - result.citation_ready_chunks

                if dry_run:
                    report.estimated_dense_embedding_requests += math.ceil(len(chunks) / 16)
                    report.processed_documents += 1
                else:
                    store, dense_embedder, sparse_encoder = self._require_live_components()
                    state = self._read_state(pair.document_id)
                    if self._is_unchanged(state, checksum, len(chunks), store, content_fingerprint):
                        result.skipped_unchanged = True
                        report.skipped_unchanged_documents += 1
                        report.documents.append(result)
                        continue
                    dense_vectors, request_count = dense_embedder.embed_chunks(chunks)
                    sparse_vectors = sparse_encoder.embed_documents(
                        [chunk.text for chunk in chunks]
                    )
                    store.upsert(chunks, dense_vectors, sparse_vectors)
                    report.qdrant_upsert_operations += 1
                    report.qdrant_upserted_points += len(chunks)
                    stale_ids = set(self._state_point_ids(state)) - {
                        chunk.chunk_id for chunk in chunks
                    }
                    store.delete_points(sorted(stale_ids))
                    self._write_manifest(pair, checksum, mapping.pages)
                    self._write_state(
                        pair.document_id,
                        checksum=checksum,
                        point_ids=[chunk.chunk_id for chunk in chunks],
                        content_fingerprint=content_fingerprint,
                    )
                    result.dense_embeddings = len(dense_vectors)
                    result.sparse_embeddings = len(sparse_vectors)
                    report.estimated_dense_embedding_requests += request_count
                    report.dense_embedding_request_count += request_count
                    report.processed_documents += 1
            except PageMappingError as error:
                result.unresolved_pages = error.unresolved_pages
                result.duplicate_pages = error.duplicate_pages
                result.page_count_mismatch = error.page_count_mismatch
                with pymupdf.open(pair.pdf_path) as pdf:
                    result.pages = pdf.page_count
                result.coverage = failed_coverage_report(
                    document_id=pair.document_id,
                    page_count=result.pages,
                    decisions=decisions,
                    transcriptions=transcriptions,
                    reason=str(error),
                )
                result.limitations = self.document_coverage_limitations(result.coverage)
                result.excluded_pages = sorted(
                    result.coverage.page_lists_by_status["excluded_blank"]
                    + result.coverage.page_lists_by_status["excluded_decorative"]
                    + result.coverage.page_lists_by_status["excluded_unverified_visual"]
                )
                failure = f"{pair.pdf_path.name}: {error}"
                result.failures.append(failure)
                report.failures.append(failure)
            except Exception as error:
                failure = f"{pair.pdf_path.name}: {error}"
                result.failures.append(failure)
                report.failures.append(failure)
            report.documents.append(result)

        report.pages = sum(item.pages for item in report.documents)
        report.chunks = sum(item.chunks for item in report.documents)
        report.dense_embedding_count = sum(item.dense_embeddings for item in report.documents)
        report.sparse_embedding_count = sum(item.sparse_embeddings for item in report.documents)
        report.fallback_pages = sum(len(item.fallback_pages) for item in report.documents)
        report.indexed_pages = sum(
            item.coverage.indexed_pages for item in report.documents if item.coverage
        )
        report.excluded_pages = sum(len(item.excluded_pages) for item in report.documents)
        report.failed_page_mappings = sum(
            item.coverage.failed_mappings for item in report.documents if item.coverage
        )
        report.coverage_limitations = [
            limitation for item in report.documents for limitation in item.limitations
        ]
        report.citation_ready_chunks = sum(item.citation_ready_chunks for item in report.documents)
        report.chunks_missing_provenance = sum(
            item.chunks_missing_provenance for item in report.documents
        )
        report.review_required = bool(
            report.failures
            or report.chunks_missing_provenance
            or any(
                item.unresolved_pages
                or item.duplicate_pages
                or item.page_count_mismatch
                or (item.coverage is not None and item.coverage.failed_mappings > 0)
                for item in report.documents
            )
        )
        report.elapsed_seconds = round(time.monotonic() - started, 3)
        if report_output:
            output = ensure_within(report_output, self.settings.data_root.resolve())
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return report

    @staticmethod
    def document_coverage_limitations(
        coverage: DocumentCoverageReport,
    ) -> list[CoverageLimitation]:
        """Expose excluded visual coverage without asserting absence from the PDF."""
        limitation = coverage_limitation(coverage)
        return [limitation] if limitation else []

    def _require_live_components(
        self,
    ) -> tuple[QdrantStore, CachedDenseEmbedder, BM25SparseEncoder]:
        if not self.store or not self.dense_embedder or not self.sparse_encoder:
            raise RuntimeError("Live ingestion components are not configured")
        return self.store, self.dense_embedder, self.sparse_encoder

    def _state_path(self, document_id: str) -> Path:
        safe_name = document_id.replace("/", "_").replace("\\", "_")
        return self.settings.data_root / "processed" / f"{safe_name}.json"

    def _read_state(self, document_id: str) -> dict[str, Any] | None:
        path = self._state_path(document_id)
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None

    @staticmethod
    def _state_point_ids(state: dict[str, Any] | None) -> list[str]:
        if not state or not isinstance(state.get("point_ids"), list):
            return []
        return [value for value in state["point_ids"] if isinstance(value, str)]

    def _is_unchanged(
        self,
        state: dict[str, Any] | None,
        checksum: str,
        chunk_count: int,
        store: QdrantStore,
        content_fingerprint: str,
    ) -> bool:
        if not state:
            return False
        return (
            state.get("source_checksum") == checksum
            and state.get("content_fingerprint") == content_fingerprint
            and state.get("ingestion_version") == self.settings.ingestion_version
            and len(self._state_point_ids(state)) == chunk_count
            and store.document_count(str(state.get("document_id"))) == chunk_count
        )

    def _write_state(
        self, document_id: str, *, checksum: str, point_ids: list[str], content_fingerprint: str
    ) -> None:
        path = self._state_path(document_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "document_id": document_id,
                    "source_checksum": checksum,
                    "content_fingerprint": content_fingerprint,
                    "ingestion_version": self.settings.ingestion_version,
                    "point_ids": point_ids,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _write_manifest(self, pair: DocumentPair, checksum: str, pages: list[DocumentPage]) -> None:
        data_root = self.settings.data_root.resolve()
        pdf_path = pair.pdf_path.resolve().relative_to(data_root).as_posix()
        markdown_path = (
            pair.markdown_path.resolve().relative_to(data_root).as_posix()
            if pair.markdown_path
            else None
        )
        extraction_methods: set[IndexedExtractionMethod] = set()
        for page in pages:
            if page.extraction_method in {
                "provided_markdown",
                "pymupdf4llm_fallback",
                "approved_manual_transcription",
            }:
                extraction_methods.add(page.extraction_method)
        manifest = DocumentManifest(
            document_id=pair.document_id,
            title=pair.title,
            region=pair.region,
            pdf_path=pdf_path,
            markdown_path=markdown_path,
            source_checksum=checksum,
            page_count=len(pages),
            ingestion_version=self.settings.ingestion_version,
            extraction_method=sorted(extraction_methods),
            created_at=datetime.now(UTC),
        )
        path = self.settings.data_root / "manifests" / "generated" / f"{pair.document_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
