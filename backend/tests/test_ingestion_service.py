import json
from pathlib import Path
from typing import Any, cast

from backend.app.config import Settings
from backend.app.ingestion.coverage import build_coverage_report
from backend.app.ingestion.models import DocumentPage, PageCoverageEntry, PageMappingResult
from backend.app.ingestion.service import IngestionService
from backend.app.retrieval.models import SparseVectorData


class FakeStore:
    def __init__(self) -> None:
        self.count = 0

    def ensure_collection(self, *, rebuild: bool = False) -> None:
        return None

    def document_count(self, document_id: str) -> int:
        return self.count

    def upsert(self, chunks: Any, dense: Any, sparse: Any) -> None:
        self.count = len(chunks)

    def delete_points(self, point_ids: Any) -> None:
        return None


class FakeDense:
    def __init__(self) -> None:
        self.calls = 0

    def embed_chunks(self, chunks: list[Any]) -> tuple[list[list[float]], int]:
        self.calls += 1
        return ([[0.0, 0.0, 0.0] for _ in chunks], 1)


class FakeSparse:
    def embed_documents(self, texts: list[str]) -> list[SparseVectorData]:
        return [SparseVectorData(indices=[1], values=[1.0]) for _ in texts]


def test_unchanged_reingestion_skips_embedding_calls(tmp_path: Path, monkeypatch: Any) -> None:
    data_root = tmp_path / "data"
    source = data_root / "source"
    (source / "pdf").mkdir(parents=True)
    (source / "markdown").mkdir()
    (data_root / "manifests").mkdir()
    (source / "pdf" / "sample.pdf").write_bytes(b"synthetic PDF identity")
    settings = Settings.model_validate(
        {
            "google_cloud_project": "test",
            "data_root": data_root,
            "gemini_embedding_dimension": 3,
        }
    )
    page = DocumentPage(
        document_id="ignored",
        page_number=1,
        text="Stable census evidence",
        extraction_method="pymupdf4llm_fallback",
        coverage_status="indexed_pymupdf4llm_fallback",
    )

    def fake_mapping(**kwargs: Any) -> PageMappingResult:
        page.document_id = str(kwargs["document_id"])
        coverage = build_coverage_report(
            page.document_id,
            1,
            [PageCoverageEntry(page_number=1, status="indexed_pymupdf4llm_fallback")],
        )
        return PageMappingResult(pages=[page], fallback_pages=[1], coverage=coverage)

    monkeypatch.setattr("backend.app.ingestion.service.map_document_pages", fake_mapping)
    store = FakeStore()
    dense = FakeDense()
    service = IngestionService(
        settings,
        store=cast(Any, store),
        dense_embedder=cast(Any, dense),
        sparse_encoder=cast(Any, FakeSparse()),
    )

    first = service.ingest(source_dir=source)
    second = service.ingest(source_dir=source)

    assert first.processed_documents == 1
    assert first.citation_ready_chunks == first.chunks
    assert first.chunks_missing_provenance == 0
    assert second.skipped_unchanged_documents == 1
    assert dense.calls == 1
    assert second.dense_embedding_request_count == 0
    assert second.dense_embedding_count == 0
    assert second.sparse_embedding_count == 0
    assert second.qdrant_upsert_operations == 0
    assert second.qdrant_upserted_points == 0
    document_id = first.documents[0].document_id
    manifest_path = data_root / "manifests" / "generated" / f"{document_id}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["pdf_path"] == "source/pdf/sample.pdf"
    assert str(tmp_path) not in manifest_path.read_text(encoding="utf-8")
