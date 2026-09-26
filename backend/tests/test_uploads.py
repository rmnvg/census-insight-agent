import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pymupdf
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.ingestion.discovery import discover_documents
from backend.app.ingestion.models import (
    DocumentIngestionResult,
    DocumentManifest,
    IngestionReport,
)
from backend.app.ingestion.page_images import highlight_fragments, render_page_png
from backend.app.ingestion.service import IngestionService
from backend.app.ingestion.uploads import UploadError, UploadService, withdrawn_document_ids
from backend.app.main import app

PAGE_TEXT = "Kerala literacy rate 2011\nThe literacy rate of Kerala was 94.00 per cent."


def pdf_bytes(pages: int = 2, text: str = PAGE_TEXT, password: str | None = None) -> bytes:
    with pymupdf.open() as pdf:
        for number in range(1, pages + 1):
            page = pdf.new_page()
            page.insert_text((72, 72), f"{text}\nPage {number}", fontsize=11)
        if password:
            return pdf.tobytes(
                encryption=pymupdf.mupdf.PDF_ENCRYPT_AES_256, owner_pw=password, user_pw=password
            )
        return pdf.tobytes()


def upload_settings(tmp_path: Path, **overrides: Any) -> Settings:
    return Settings.model_validate(
        {"google_cloud_project": "test-project", "data_root": tmp_path / "data", **overrides}
    )


class FakeIngestion:
    def __init__(self, *, chunks: int = 4, raises: bool = False) -> None:
        self.chunks = chunks
        self.raises = raises
        self.calls: list[str | None] = []

    def ingest(self, **kwargs: Any) -> IngestionReport:
        self.calls.append(kwargs.get("document_id"))
        if self.raises:
            raise RuntimeError("/app/data/source/pdf/x.pdf: embedding quota exhausted")
        result = DocumentIngestionResult(
            document_id=kwargs["document_id"], pages=2, chunks=self.chunks
        )
        return IngestionReport(documents=[result])


def service(
    tmp_path: Path, ingestion: FakeIngestion | None = None, **overrides: Any
) -> tuple[UploadService, list[str]]:
    removed: list[str] = []
    fake = ingestion or FakeIngestion()
    uploads = UploadService(
        upload_settings(tmp_path, **overrides),
        ingestion_factory=lambda _: fake,  # type: ignore[arg-type,return-value]
        point_remover=removed.append,
    )
    return uploads, removed


def stage(uploads: UploadService, content: bytes | None = None, **fields: str) -> Any:
    return uploads.stage(
        filename=fields.get("filename", "../../etc/Kerala Report.pdf"),
        content=pdf_bytes() if content is None else content,
        title=fields.get("title", "Census 2011 Kerala highlights"),
        region=fields.get("region", "Kerala"),
    )


def test_staged_upload_is_discoverable_with_its_supplied_identity(tmp_path: Path) -> None:
    uploads, _ = service(tmp_path)
    job = stage(uploads)

    assert job.status == "queued" and job.page_count == 2
    assert job.document_id.startswith("upload-census-2011-kerala-highlights-")
    assert job.original_filename == "Kerala Report.pdf"
    pairs = discover_documents(tmp_path / "data" / "source", tmp_path / "data" / "manifests")
    [pair] = pairs
    assert (pair.document_id, pair.title, pair.region) == (
        job.document_id,
        "Census 2011 Kerala highlights",
        "Kerala",
    )
    # The stored filename is derived from the document ID, never from the client.
    assert pair.pdf_path.name == f"{job.document_id}.pdf"


def test_real_extraction_path_indexes_an_uploaded_pdf(tmp_path: Path) -> None:
    uploads, _ = service(tmp_path)
    job = stage(uploads)
    report = IngestionService(upload_settings(tmp_path)).ingest(
        source_dir=tmp_path / "data" / "source", document_id=job.document_id, dry_run=True
    )
    [result] = report.documents
    assert not result.failures
    assert result.fallback_pages == [1, 2]
    assert result.chunks > 0 and result.chunks_missing_provenance == 0


@pytest.mark.parametrize(
    ("content", "fields", "status"),
    [
        (b"hello, not a pdf", {}, 415),
        (b"%PDF-1.7 truncated garbage", {}, 415),
        (None, {"region": "Kerala<script>"}, 422),
        (None, {"title": "   "}, 422),
    ],
)
def test_invalid_uploads_are_rejected_before_anything_is_written(
    tmp_path: Path, content: bytes | None, fields: dict[str, str], status: int
) -> None:
    uploads, _ = service(tmp_path)
    with pytest.raises(UploadError) as caught:
        stage(uploads, content, **fields)
    assert caught.value.status_code == status
    assert not (tmp_path / "data" / "source" / "pdf").exists()


def test_size_page_and_password_limits(tmp_path: Path) -> None:
    uploads, _ = service(tmp_path, document_upload_max_bytes=2048)
    with pytest.raises(UploadError) as too_big:
        stage(uploads, pdf_bytes(pages=3) + b"\0" * 4096)
    assert too_big.value.status_code == 413

    uploads, _ = service(tmp_path, document_upload_max_pages=2)
    with pytest.raises(UploadError) as too_many:
        stage(uploads, pdf_bytes(pages=3))
    assert too_many.value.status_code == 413

    with pytest.raises(UploadError, match="Password"):
        stage(uploads, pdf_bytes(password="secret"))


def test_duplicate_pdf_is_rejected(tmp_path: Path) -> None:
    uploads, _ = service(tmp_path)
    content = pdf_bytes()
    job = stage(uploads, content)
    with pytest.raises(UploadError) as in_flight:
        stage(uploads, content)
    assert in_flight.value.status_code == 409

    generated = tmp_path / "data" / "manifests" / "generated"
    generated.mkdir(parents=True)
    (generated / "existing.json").write_text(
        DocumentManifest(
            document_id="existing",
            title="Existing report",
            region="Kerala",
            pdf_path="source/pdf/existing.pdf",
            source_checksum=job.source_checksum,
            page_count=2,
            ingestion_version="1",
            extraction_method=["pymupdf4llm_fallback"],
            created_at=datetime.now(UTC),
        ).model_dump_json()
    )
    asyncio.run(uploads.process(job.job_id))
    with pytest.raises(UploadError, match="Existing report"):
        stage(uploads, content)


def test_successful_processing_reports_indexed_counts(tmp_path: Path) -> None:
    fake = FakeIngestion(chunks=7)
    uploads, removed = service(tmp_path, fake)
    job = stage(uploads)
    done = asyncio.run(uploads.process(job.job_id))
    assert fake.calls == [job.document_id]
    assert done.status == "succeeded" and done.chunks == 7
    assert removed == []
    assert uploads.get(job.job_id) == done


@pytest.mark.parametrize("ingestion", [FakeIngestion(raises=True), FakeIngestion(chunks=0)])
def test_failed_processing_removes_source_and_points(
    tmp_path: Path, ingestion: FakeIngestion
) -> None:
    uploads, removed = service(tmp_path, ingestion)
    job = stage(uploads)
    done = asyncio.run(uploads.process(job.job_id))
    assert done.status == "failed" and done.detail
    assert "/app/data" not in done.detail
    assert removed == [job.document_id]
    assert not list((tmp_path / "data" / "source" / "pdf").glob("*.pdf"))
    assert not list((tmp_path / "data" / "manifests" / "uploads").glob("*.json"))


class FlakyRemover:
    """A point remover whose Qdrant is unreachable until `available` is set."""

    def __init__(self) -> None:
        self.available = False
        self.removed: list[str] = []

    def __call__(self, document_id: str) -> None:
        if not self.available:
            raise ConnectionError("Qdrant is unreachable")
        self.removed.append(document_id)


def flaky_service(
    tmp_path: Path, remover: FlakyRemover, ingestion: FakeIngestion | None = None
) -> UploadService:
    fake = ingestion or FakeIngestion(raises=True)
    return UploadService(
        upload_settings(tmp_path),
        ingestion_factory=lambda _: fake,  # type: ignore[arg-type,return-value]
        point_remover=remover,
    )


def test_failed_cleanup_keeps_the_source_until_points_are_removed(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    remover = FlakyRemover()
    uploads = flaky_service(tmp_path, remover)
    job = stage(uploads)
    pdf = data_root / "source" / "pdf" / f"{job.document_id}.pdf"
    override = data_root / "manifests" / "uploads" / f"{job.document_id}.override.json"

    done = asyncio.run(uploads.process(job.job_id))

    # Points may still be indexed, so the authoritative PDF must still exist, and retrieval
    # must already exclude the document.
    assert done.status == "failed"
    assert pdf.is_file() and override.is_file()
    assert withdrawn_document_ids(data_root) == [job.document_id]
    remover.available = True
    uploads.finish_withdrawals()
    assert remover.removed == [job.document_id]
    assert not pdf.exists() and not override.exists()
    assert withdrawn_document_ids(data_root) == []


def test_restart_finishes_an_interrupted_withdrawal(tmp_path: Path) -> None:
    remover = FlakyRemover()
    uploads = flaky_service(tmp_path, remover)
    job = stage(uploads)
    asyncio.run(uploads.process(job.job_id))
    assert withdrawn_document_ids(tmp_path / "data") == [job.document_id]

    remover.available = True
    flaky_service(tmp_path, remover)
    assert remover.removed == [job.document_id]
    assert withdrawn_document_ids(tmp_path / "data") == []
    assert not list((tmp_path / "data" / "source" / "pdf").glob("*.pdf"))


def test_reupload_waits_for_the_earlier_attempt_to_be_cleaned_up(tmp_path: Path) -> None:
    remover = FlakyRemover()
    uploads = flaky_service(tmp_path, remover)
    content = pdf_bytes()
    first = stage(uploads, content)
    asyncio.run(uploads.process(first.job_id))

    with pytest.raises(UploadError) as pending:
        stage(uploads, content)
    assert pending.value.status_code == 503

    remover.available = True
    second = stage(uploads, content)
    assert second.document_id == first.document_id and second.status == "queued"
    assert (tmp_path / "data" / "source" / "pdf" / f"{second.document_id}.pdf").is_file()
    assert withdrawn_document_ids(tmp_path / "data") == []


class ManifestThenFailIngestion(FakeIngestion):
    """Fails after the generated manifest and index state were written, as a late error does."""

    def __init__(self, checksum: str) -> None:
        super().__init__()
        self.checksum = checksum

    def ingest(self, **kwargs: Any) -> IngestionReport:
        document_id = kwargs["document_id"]
        data_root = kwargs["source_dir"].parent
        (data_root / "manifests" / "generated").mkdir(parents=True, exist_ok=True)
        (data_root / "manifests" / "generated" / f"{document_id}.json").write_text(
            DocumentManifest(
                document_id=document_id,
                title="Kerala",
                region="Kerala",
                pdf_path=f"source/pdf/{document_id}.pdf",
                source_checksum=self.checksum,
                page_count=2,
                ingestion_version="1",
                extraction_method=["pymupdf4llm_fallback"],
                created_at=datetime.now(UTC),
            ).model_dump_json()
        )
        (data_root / "processed" / f"{document_id}.json").write_text("{}")
        (data_root / "processed" / "tables").mkdir(parents=True, exist_ok=True)
        (data_root / "processed" / "tables" / f"{document_id}.json").write_text("{}")
        result = DocumentIngestionResult(
            document_id=document_id, pages=2, chunks=4, failures=["state write failed"]
        )
        return IngestionReport(documents=[result])


def test_failed_processing_removes_the_generated_manifest(tmp_path: Path) -> None:
    content = pdf_bytes()
    uploads, removed = service(tmp_path)
    checksum = stage(uploads, content).source_checksum
    uploads, removed = service(tmp_path, ManifestThenFailIngestion(checksum))
    # The restart above failed the first job; this is a fresh attempt.
    job = stage(uploads, content)
    done = asyncio.run(uploads.process(job.job_id))

    assert done.status == "failed"
    assert not list((tmp_path / "data" / "manifests" / "generated").glob("*.json"))
    assert not (tmp_path / "data" / "processed" / f"{job.document_id}.json").exists()
    assert not (tmp_path / "data" / "processed" / "tables" / f"{job.document_id}.json").exists()
    # Without its manifest the failed document is neither listed nor a "duplicate".
    assert stage(uploads, content).status == "queued"


def test_interrupted_jobs_are_failed_on_restart(tmp_path: Path) -> None:
    uploads, _ = service(tmp_path)
    job = stage(uploads)
    restarted, removed = service(tmp_path)
    recovered = restarted.get(job.job_id)
    assert recovered is not None and recovered.status == "failed"
    assert removed == [job.document_id]


def test_only_uploaded_documents_can_be_deleted(tmp_path: Path) -> None:
    uploads, removed = service(tmp_path)
    with pytest.raises(UploadError) as corpus:
        asyncio.run(uploads.delete_document("census-2011-karnataka-pca-highlights"))
    assert corpus.value.status_code == 403

    job = stage(uploads)
    asyncio.run(uploads.process(job.job_id))
    asyncio.run(uploads.delete_document(job.document_id))
    assert removed == [job.document_id]
    assert uploads.get(job.job_id) is None
    assert not (tmp_path / "data" / "source" / "pdf" / f"{job.document_id}.pdf").exists()
    with pytest.raises(UploadError) as missing:
        asyncio.run(uploads.delete_document(job.document_id))
    assert missing.value.status_code == 404


def test_page_render_highlights_unambiguous_citation_text(tmp_path: Path) -> None:
    path = tmp_path / "doc.pdf"
    path.write_bytes(pdf_bytes(pages=1))
    plain, plain_status = render_page_png(path, 1)
    highlighted, status = render_page_png(
        path, 1, "The literacy rate of Kerala was 94.00 per cent."
    )
    assert plain.startswith(b"\x89PNG") and highlighted.startswith(b"\x89PNG")
    assert (plain_status, status) == ("none", "matched")
    assert plain != highlighted
    assert render_page_png(path, 1, "Nothing like this appears")[1] == "unmatched"
    with pytest.raises(IndexError):
        render_page_png(path, 2)
    assert highlight_fragments("| **Kerala** | 94.00 |<br>\n|---|---|") == ["Kerala", "94.00"]


def test_http_upload_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    uploads, _ = service(tmp_path)
    test_settings = upload_settings(tmp_path)
    monkeypatch.setattr("backend.app.api.get_settings", lambda: test_settings)
    monkeypatch.setattr("backend.app.api.get_upload_service", lambda: uploads)
    client = TestClient(app)

    response = client.post(
        "/documents/upload",
        files={"file": ("kerala.pdf", pdf_bytes(), "application/pdf")},
        data={"title": "Kerala highlights", "region": "Kerala"},
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    # TestClient runs background tasks before returning.
    assert client.get(f"/documents/uploads/{job_id}").json()["status"] == "succeeded"
    assert [job["job_id"] for job in client.get("/documents/uploads").json()] == [job_id]

    rejected = client.post(
        "/documents/upload",
        files={"file": ("notes.pdf", b"plain text", "application/pdf")},
        data={"title": "Notes", "region": "Kerala"},
    )
    assert rejected.status_code == 415
    assert client.get("/documents/uploads/not-a-job").status_code == 404

    monkeypatch.setattr(
        "backend.app.api.get_settings",
        lambda: upload_settings(tmp_path, document_upload_enabled=False),
    )
    disabled = client.post(
        "/documents/upload",
        files={"file": ("kerala.pdf", pdf_bytes(pages=1), "application/pdf")},
        data={"title": "Kerala", "region": "Kerala"},
    )
    assert disabled.status_code == 403


def test_http_page_image_serves_only_catalogued_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_settings = upload_settings(tmp_path)
    monkeypatch.setattr("backend.app.api.get_settings", lambda: test_settings)
    pdf = tmp_path / "data" / "source" / "pdf" / "doc.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(pdf_bytes(pages=3))
    generated = tmp_path / "data" / "manifests" / "generated"
    generated.mkdir(parents=True)
    (generated / "doc.json").write_text(
        DocumentManifest(
            document_id="doc",
            title="Doc",
            region="Kerala",
            pdf_path="source/pdf/doc.pdf",
            source_checksum="0" * 64,
            page_count=3,
            ingestion_version="1",
            extraction_method=["pymupdf4llm_fallback"],
            created_at=datetime.now(UTC),
        ).model_dump_json()
    )
    client = TestClient(app)
    page = client.get("/documents/doc/pages/2", params={"highlight": "Kerala"})
    assert page.status_code == 200
    assert page.headers["content-type"] == "image/png"
    assert page.headers["x-page-count"] == "3"
    assert page.headers["x-highlight"] == "matched"
    assert client.get("/documents/doc/pages/4").status_code == 404
    assert client.get("/documents/unknown/pages/1").status_code == 404
    assert client.get("/documents/..%2F..%2Fsecret/pages/1").status_code == 404


def test_oversized_upload_is_rejected_before_the_body_is_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploads, _ = service(tmp_path)
    monkeypatch.setattr(
        "backend.app.main.get_settings",
        lambda: upload_settings(tmp_path, document_upload_max_bytes=1024),
    )
    monkeypatch.setattr("backend.app.api.get_upload_service", lambda: uploads)
    client = TestClient(app)
    response = client.post(
        "/documents/upload",
        files={"file": ("big.pdf", b"%PDF-" + b"0" * (2 * 1024 * 1024), "application/pdf")},
        data={"title": "Big", "region": "Kerala"},
    )
    assert response.status_code == 413
    assert not (tmp_path / "data" / "source").exists()


def test_highlight_fragments_strip_inline_html_from_real_table_snippets() -> None:
    # Row copied verbatim from a live citation (Karnataka, physical page 50, Statement 19).
    snippet = (
        "| -                     | <b>KARNATAKA</b> | <b>4,06,47,322</b> | <b>2,26,49,176</b> | "
        "<b>1,79,98,146</b> | <b>66.64</b>  | <b>59.33</b> | <b>80.58</b> | <b>75.36</b> |"
    )
    fragments = highlight_fragments(snippet)
    assert "KARNATAKA" in fragments and "4,06,47,322" in fragments and "75.36" in fragments
    assert not any("<" in fragment or ">" in fragment for fragment in fragments)
    assert highlight_fragments("Literacy<br>rate") == ["Literacy rate"]


def test_vector_text_pages_report_no_text_layer(tmp_path: Path) -> None:
    # The bundled Census PDFs draw glyphs as vector paths; model that with a drawing-only page.
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.draw_rect(pymupdf.Rect(72, 72, 200, 100))
        path = tmp_path / "vector.pdf"
        pdf.save(path)
    assert render_page_png(path, 1, "KARNATAKA 75.36")[1] == "no_text_layer"
