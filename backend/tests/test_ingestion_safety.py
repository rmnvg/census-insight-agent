from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from backend.app.config import Settings
from backend.app.ingestion.chunking import chunk_pages
from backend.app.ingestion.coverage import (
    LIMITATION_MESSAGE,
    build_coverage_report,
    load_manual_transcriptions,
)
from backend.app.ingestion.embeddings import CachedDenseEmbedder
from backend.app.ingestion.errors import PageMappingError
from backend.app.ingestion.models import (
    DocumentChunk,
    DocumentPage,
    ManualTranscriptionRecord,
    PageCoverageEntry,
    PageMappingResult,
)
from backend.app.ingestion.safety import UnsafeExtractionError
from backend.app.ingestion.service import IngestionService
from backend.app.retrieval.models import SparseVectorData
from backend.app.retrieval.qdrant_store import QdrantStore


def safe_chunk() -> DocumentChunk:
    page = DocumentPage(
        document_id="doc",
        page_number=1,
        text="Verbatim evidence",
        extraction_method="provided_markdown",
        coverage_status="indexed_provided_markdown",
    )
    return chunk_pages([page], document_title="Doc", region="Region", source_checksum="a" * 64)[0]


def unsafe_ocr_chunk() -> DocumentChunk:
    chunk = safe_chunk()
    metadata = chunk.metadata.model_copy(
        update={
            "extraction_method": "unverified_ocr",
            "coverage_status": "excluded_unverified_visual",
        }
    )
    return chunk.model_copy(update={"metadata": metadata})


def test_unverified_ocr_cannot_be_chunked() -> None:
    page = DocumentPage(
        document_id="doc",
        page_number=13,
        text="raw tesseract output",
        extraction_method="unverified_ocr",
        coverage_status="excluded_unverified_visual",
    )

    with pytest.raises(UnsafeExtractionError, match="cannot be chunked"):
        chunk_pages([page], document_title="Doc", region="Region", source_checksum="a" * 64)


def test_unverified_ocr_cannot_be_embedded() -> None:
    class Provider:
        settings = Settings.model_validate(
            {"google_cloud_project": "test", "gemini_embedding_dimension": 3}
        )

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("Embedding provider must not be called")

    embedder = CachedDenseEmbedder(cast(Any, Provider()), cast(Any, object()))

    with pytest.raises(UnsafeExtractionError, match="Unverified OCR"):
        embedder.embed_chunks([unsafe_ocr_chunk()])


def test_unverified_ocr_cannot_be_uploaded_to_qdrant() -> None:
    store = QdrantStore(
        cast(Any, object()),
        "collection",
        dense_dimensions=3,
        dense_model="dense",
        sparse_model="sparse",
    )

    with pytest.raises(UnsafeExtractionError, match="Unverified OCR"):
        store.upsert(
            [unsafe_ocr_chunk()],
            [[0.0, 0.0, 0.0]],
            [SparseVectorData(indices=[1], values=[1.0])],
        )


def test_every_pdf_page_has_exactly_one_coverage_status_and_exclusions_are_not_failures() -> None:
    entries = [
        PageCoverageEntry(page_number=1, status="indexed_provided_markdown"),
        PageCoverageEntry(page_number=2, status="excluded_unverified_visual"),
        PageCoverageEntry(page_number=3, status="excluded_blank"),
    ]

    report = build_coverage_report("doc", 3, entries)

    assert len(report.pages) == report.pdf_page_count == 3
    assert report.page_lists_by_status["excluded_unverified_visual"] == [2]
    assert report.page_lists_by_status["excluded_blank"] == [3]
    assert report.failed_mappings == 0
    assert report.percentage_pages_indexed < 100
    with pytest.raises(ValueError, match="exactly one status"):
        build_coverage_report("doc", 3, [*entries, entries[0]])


def test_approved_manual_transcription_requires_human_reviewer_metadata() -> None:
    payload = {
        "document_id": "doc",
        "page_number": 2,
        "transcription": "Reviewed source text",
        "reviewed_at": datetime.now(UTC).isoformat(),
        "source_checksum": "a" * 64,
        "verification_notes": "Checked twice against the PDF",
        "status": "approved",
    }
    with pytest.raises(ValidationError, match="reviewer"):
        ManualTranscriptionRecord.model_validate(payload)
    with pytest.raises(ValidationError, match="human reviewer"):
        ManualTranscriptionRecord.model_validate({**payload, "reviewer": "Tesseract"})


def test_approved_transcription_checksum_must_match_pdf(tmp_path: Path) -> None:
    directory = tmp_path / "manual-transcriptions"
    directory.mkdir()
    record = {
        "document_id": "doc",
        "page_number": 2,
        "transcription": "Reviewed source text",
        "structured_values": None,
        "reviewer": "A. Reviewer",
        "reviewed_at": datetime.now(UTC).isoformat(),
        "source_checksum": "b" * 64,
        "verification_notes": "Checked against the PDF",
        "status": "approved",
    }
    (directory / "doc-page-2.json").write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_manual_transcriptions(tmp_path, "doc", "a" * 64)


def test_limitation_wording_does_not_claim_information_is_absent_from_pdf() -> None:
    normalized = LIMITATION_MESSAGE.casefold()
    assert "absent from the pdf" not in normalized
    assert "not in the pdf" not in normalized
    assert "excluded" in normalized
    assert "indexed text" in normalized


def test_dry_run_makes_no_embedding_or_qdrant_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    source = data_root / "source"
    (source / "pdf").mkdir(parents=True)
    (source / "markdown").mkdir()
    (data_root / "manifests").mkdir()
    (source / "pdf" / "sample.pdf").write_bytes(b"pdf identity")
    coverage = build_coverage_report(
        "placeholder",
        1,
        [PageCoverageEntry(page_number=1, status="indexed_provided_markdown")],
    )

    def fake_mapping(**kwargs: Any) -> PageMappingResult:
        document_id = str(kwargs["document_id"])
        page = DocumentPage(
            document_id=document_id,
            page_number=1,
            text="Citation-safe text",
            extraction_method="provided_markdown",
            coverage_status="indexed_provided_markdown",
        )
        return PageMappingResult(
            pages=[page],
            markdown_pages=[1],
            coverage=coverage.model_copy(update={"document_id": document_id}),
        )

    monkeypatch.setattr("backend.app.ingestion.service.map_document_pages", fake_mapping)

    class NeverStore:
        def ensure_collection(self, *, rebuild: bool = False) -> None:
            raise AssertionError("Qdrant must not be called")

        def upsert(self, *args: object) -> None:
            raise AssertionError("Qdrant must not be called")

    class NeverEmbedder:
        def embed_chunks(self, chunks: list[DocumentChunk]) -> tuple[list[list[float]], int]:
            raise AssertionError("External embedding API must not be called")

    settings = Settings.model_validate({"google_cloud_project": "test", "data_root": data_root})
    service = IngestionService(
        settings,
        store=cast(Any, NeverStore()),
        dense_embedder=cast(Any, NeverEmbedder()),
        sparse_encoder=cast(Any, object()),
    )

    report = service.ingest(source_dir=source, dry_run=True)

    assert report.processed_documents == 1
    assert report.failures == []
    assert report.dense_embedding_count == 0
    assert report.sparse_embedding_count == 0


def test_approved_visual_exclusion_does_not_set_review_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    source = data_root / "source"
    (source / "pdf").mkdir(parents=True)
    (source / "markdown").mkdir()
    (data_root / "manifests").mkdir()
    (source / "pdf" / "sample.pdf").write_bytes(b"pdf identity")
    coverage = build_coverage_report(
        "placeholder",
        2,
        [
            PageCoverageEntry(page_number=1, status="indexed_provided_markdown"),
            PageCoverageEntry(page_number=2, status="excluded_unverified_visual"),
        ],
    )

    def fake_mapping(**kwargs: Any) -> PageMappingResult:
        document_id = str(kwargs["document_id"])
        return PageMappingResult(
            pages=[
                DocumentPage(
                    document_id=document_id,
                    page_number=1,
                    text="Safe evidence",
                    extraction_method="provided_markdown",
                    coverage_status="indexed_provided_markdown",
                ),
                DocumentPage(
                    document_id=document_id,
                    page_number=2,
                    text="",
                    extraction_method=None,
                    coverage_status="excluded_unverified_visual",
                ),
            ],
            coverage=coverage.model_copy(update={"document_id": document_id}),
        )

    monkeypatch.setattr("backend.app.ingestion.service.map_document_pages", fake_mapping)
    settings = Settings.model_validate({"google_cloud_project": "test", "data_root": data_root})
    report = IngestionService(settings).ingest(source_dir=source, dry_run=True)

    assert report.review_required is False
    assert report.excluded_pages == 1
    assert len(report.coverage_limitations) == 1


def test_genuine_unresolved_mapping_sets_review_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pymupdf

    data_root = tmp_path / "data"
    source = data_root / "source"
    (source / "pdf").mkdir(parents=True)
    (source / "markdown").mkdir()
    (data_root / "manifests").mkdir()
    document = pymupdf.open()
    document.new_page()
    document.save(source / "pdf" / "sample.pdf")
    document.close()

    def fail_mapping(**kwargs: Any) -> PageMappingResult:
        raise PageMappingError("unresolved alignment", unresolved_pages=[1])

    monkeypatch.setattr("backend.app.ingestion.service.map_document_pages", fail_mapping)
    settings = Settings.model_validate({"google_cloud_project": "test", "data_root": data_root})
    report = IngestionService(settings).ingest(source_dir=source, dry_run=True)

    assert report.review_required is True
    assert report.failed_page_mappings == 1
    assert report.documents[0].coverage is not None
    assert report.documents[0].coverage.page_lists_by_status["failed_page_mapping"] == [1]


def test_page_review_directory_is_rejected_as_an_ingestion_source(tmp_path: Path) -> None:
    settings = Settings.model_validate(
        {"google_cloud_project": "test", "data_root": tmp_path / "data"}
    )
    service = IngestionService(settings)

    with pytest.raises(UnsafeExtractionError, match="not ingestion sources"):
        service.ingest(
            source_dir=tmp_path / "data" / "processed" / "page-review" / "karnataka" / "ocr",
            dry_run=True,
        )
