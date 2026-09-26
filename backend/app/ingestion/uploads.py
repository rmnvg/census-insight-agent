"""Add user-supplied PDFs to the document library through the citation-safe ingestion pipeline.

An upload never takes a shortcut around ingestion: the PDF is validated, stored as the
authoritative source file under `data/source/pdf/`, described by an override manifest under
`data/manifests/uploads/`, and then indexed by the exact same `IngestionService.ingest` path the
CLI uses (PyMuPDF4LLM extraction, page-level provenance, checksum-bound chunks, dense + BM25
vectors). Pages without a usable text layer are reported as failed mappings, never guessed.

A failed upload is withdrawn before it is removed. A durable marker under
`data/processed/uploads/withdrawn/` excludes the document from retrieval at once; its source PDF
is deleted only after its Qdrant points are gone, so indexed evidence never outlives its
authoritative source. A cleanup that cannot finish (Qdrant unreachable) is retried at startup and
before each job.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pymupdf
from pydantic import BaseModel, Field

from backend.app.config import Settings
from backend.app.ingestion.models import (
    DocumentIngestionResult,
    DocumentManifest,
    IngestionReport,
    ManifestOverride,
)
from backend.app.ingestion.service import IngestionService

logger = logging.getLogger(__name__)

UPLOAD_PREFIX = "upload-"
_JOB_ID = re.compile(r"^[0-9a-f]{32}$")
_DOCUMENT_ID = re.compile(r"^upload-[a-z0-9-]{1,64}$")
_REGION = re.compile(r"^[\w .,'()&/-]+$", re.UNICODE)

UploadStatus = Literal["queued", "processing", "succeeded", "failed"]


class UploadError(ValueError):
    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


class TransientIngestionError(RuntimeError):
    """Indexing raised; the source is kept so a durable worker can retry the job."""


class UploadJob(BaseModel):
    job_id: str
    document_id: str
    title: str
    region: str
    original_filename: str
    byte_size: int = Field(ge=0)
    page_count: int = Field(ge=0)
    source_checksum: str
    status: UploadStatus
    detail: str | None = None
    indexed_pages: int = 0
    failed_pages: list[int] = Field(default_factory=list)
    chunks: int = 0
    created_at: datetime
    updated_at: datetime


IngestionFactory = Callable[[Settings], IngestionService]
PointRemover = Callable[[str], None]


def _slug(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")[:40] or "document"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _withdrawn_dir(data_root: Path) -> Path:
    return data_root / "processed" / "uploads" / "withdrawn"


def withdrawn_document_ids(data_root: Path) -> list[str]:
    """Uploaded documents whose failed indexing is not fully cleaned up; never retrieve them."""
    directory = _withdrawn_dir(data_root)
    if not directory.is_dir():
        return []
    return sorted(
        path.stem for path in directory.glob("*.json") if _DOCUMENT_ID.fullmatch(path.stem)
    )


class UploadService:
    def __init__(
        self,
        settings: Settings,
        *,
        ingestion_factory: IngestionFactory = IngestionService.live,
        point_remover: PointRemover | None = None,
        recover_interrupted: bool = True,
    ) -> None:
        self.settings = settings
        self.ingestion_factory = ingestion_factory
        self.point_remover = point_remover
        self.data_root = settings.data_root
        self.pdf_dir = self.data_root / "source" / "pdf"
        self.override_dir = self.data_root / "manifests" / "uploads"
        self.generated_dir = self.data_root / "manifests" / "generated"
        self.jobs_dir = self.data_root / "processed" / "uploads" / "jobs"
        self.reports_dir = self.data_root / "processed" / "uploads" / "reports"
        self.withdrawn_dir = _withdrawn_dir(self.data_root)
        # Ingestion shares one embedding cache and one Qdrant collection; one job at a time.
        self._ingestion_lock = asyncio.Lock()
        # Staging runs outside the ingestion lock, and both may finish a withdrawal.
        self._withdrawal_lock = threading.Lock()
        # In-process jobs die with the API process. A durable worker queue owns its jobs and
        # redelivers them after a crash, so the API must not fail them at startup.
        if recover_interrupted:
            self._recover_interrupted_jobs()
        self.finish_withdrawals()

    # ----------------------------------------------------------------------------- staging

    def stage(self, *, filename: str, content: bytes, title: str, region: str) -> UploadJob:
        """Validate an uploaded PDF and store it with its override manifest."""
        clean_title = " ".join(title.split())
        clean_region = " ".join(region.split())
        if not 1 <= len(clean_title) <= 200:
            raise UploadError("Title must be between 1 and 200 characters.")
        if not 1 <= len(clean_region) <= 80 or not _REGION.fullmatch(clean_region):
            raise UploadError("Region must be 1-80 letters, digits, spaces, or basic punctuation.")
        if len(content) > self.settings.document_upload_max_bytes:
            limit = self.settings.document_upload_max_bytes // (1024 * 1024)
            raise UploadError(f"PDF exceeds the {limit} MB upload limit.", 413)
        if not content.lstrip()[:5].startswith(b"%PDF-"):
            raise UploadError("File is not a PDF.", 415)
        try:
            with pymupdf.open(stream=content, filetype="pdf") as pdf:
                if pdf.needs_pass:
                    raise UploadError("Password-protected PDFs are not supported.")
                page_count = pdf.page_count
        except UploadError:
            raise
        except Exception as error:
            raise UploadError("PDF could not be opened; it may be corrupted.", 415) from error
        if page_count < 1:
            raise UploadError("PDF has no pages.")
        if page_count > self.settings.document_upload_max_pages:
            raise UploadError(
                f"PDF has {page_count} pages; the limit is "
                f"{self.settings.document_upload_max_pages}.",
                413,
            )

        checksum = hashlib.sha256(content).hexdigest()
        duplicate = self._document_with_checksum(checksum)
        if duplicate is not None:
            raise UploadError(f"This PDF is already in the library as “{duplicate.title}”.", 409)
        if any(
            job.source_checksum == checksum and job.status in {"queued", "processing"}
            for job in self.list_jobs()
        ):
            raise UploadError("This PDF is already being processed.", 409)

        document_id = f"{UPLOAD_PREFIX}{_slug(clean_title)}-{checksum[:10]}"
        # A failed earlier attempt at this exact document writes to the same paths; its cleanup
        # must finish first or it would delete the new source.
        if not self._finish_withdrawal(document_id):
            raise UploadError(
                "An earlier attempt to index this PDF is still being cleaned up; "
                "please try again shortly.",
                503,
            )
        # The stored filename is derived, never user-controlled.
        pdf_filename = f"{document_id}.pdf"
        _atomic_write(self.pdf_dir / pdf_filename, content)
        override = ManifestOverride(
            document_id=document_id,
            title=clean_title,
            region=clean_region,
            pdf_filename=pdf_filename,
            markdown_filename=None,
        )
        _atomic_write(
            self.override_dir / f"{document_id}.override.json",
            override.model_dump_json(indent=2).encode(),
        )
        now = datetime.now(UTC)
        job = UploadJob(
            job_id=uuid4().hex,
            document_id=document_id,
            title=clean_title,
            region=clean_region,
            original_filename=Path(filename or "document.pdf").name[:200],
            byte_size=len(content),
            page_count=page_count,
            source_checksum=checksum,
            status="queued",
            created_at=now,
            updated_at=now,
        )
        self._save(job)
        return job

    # ----------------------------------------------------------------------------- processing

    async def process(self, job_id: str) -> UploadJob:
        """Index a staged upload inside the API process (the default single-host mode)."""
        async with self._ingestion_lock:
            return await asyncio.to_thread(self.run_job, job_id)

    def run_job(self, job_id: str, *, retry_on_error: bool = False) -> UploadJob:
        """Index a staged upload. Safe to run again for the same job after a crash.

        With `retry_on_error`, an exception keeps the source and raises
        `TransientIngestionError` so the caller can retry; otherwise the job fails and every trace
        of the document is removed. A document that indexes without error but yields no
        citation-safe text fails permanently either way, because retrying cannot change that.
        """
        job = self._require(job_id)
        if job.status in {"succeeded", "failed"}:
            # A redelivered message for a finished job is a no-op, never a second ingestion.
            return job
        self.finish_withdrawals()
        job = self._update(job, status="processing", detail="Extracting and indexing pages")
        try:
            report = self._ingest(job.document_id)
        except Exception as error:
            if retry_on_error:
                self._update(
                    job,
                    status="queued",
                    detail="Indexing hit an error and will be retried automatically.",
                )
                raise TransientIngestionError(self._safe_failure(str(error))) from error
            self._discard_failed(job.document_id)
            return self._update(job, status="failed", detail=self._safe_failure(str(error)))
        result = next(
            (item for item in report.documents if item.document_id == job.document_id), None
        )
        if result is None or result.failures or result.chunks == 0:
            self._discard_failed(job.document_id)
            reason = (
                result.failures[0]
                if result and result.failures
                else "No citation-safe text could be extracted. Scanned PDFs need OCR "
                "review before they can be indexed."
            )
            return self._update(job, status="failed", detail=self._safe_failure(reason))
        return self._finish(job, result)

    def fail(self, job_id: str, detail: str) -> UploadJob:
        """Give up on a staged job (retries exhausted, or it could not be queued)."""
        job = self._require(job_id)
        self._discard_failed(job.document_id)
        return self._update(job, status="failed", detail=self._safe_failure(detail))

    def _ingest(self, document_id: str) -> IngestionReport:
        service = self.ingestion_factory(self.settings)
        return service.ingest(
            source_dir=self.data_root / "source",
            document_id=document_id,
            report_output=self.reports_dir / f"{document_id}.json",
        )

    def _finish(self, job: UploadJob, result: DocumentIngestionResult) -> UploadJob:
        coverage = result.coverage
        failed = coverage.page_lists_by_status.get("failed_page_mapping", []) if coverage else []
        indexed = coverage.indexed_pages if coverage else len(result.fallback_pages)
        detail = f"Indexed {indexed} of {result.pages} pages into {result.chunks} chunks."
        if failed:
            detail += (
                f" {len(failed)} page(s) had no text layer and are excluded from answers "
                "(they remain in the source PDF)."
            )
        return self._update(
            job,
            status="succeeded",
            detail=detail,
            indexed_pages=indexed,
            failed_pages=sorted(failed),
            chunks=result.chunks,
        )

    # ----------------------------------------------------------------------------- queries

    def get(self, job_id: str) -> UploadJob | None:
        if not _JOB_ID.fullmatch(job_id):
            return None
        path = self.jobs_dir / f"{job_id}.json"
        if not path.is_file():
            return None
        return UploadJob.model_validate_json(path.read_text(encoding="utf-8"))

    def list_jobs(self, limit: int = 50) -> list[UploadJob]:
        if not self.jobs_dir.is_dir():
            return []
        jobs: list[UploadJob] = []
        for path in self.jobs_dir.glob("*.json"):
            try:
                jobs.append(UploadJob.model_validate_json(path.read_text(encoding="utf-8")))
            except ValueError:
                continue
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)[:limit]

    @staticmethod
    def report_path_for(data_root: Path, document_id: str) -> Path | None:
        if not _DOCUMENT_ID.fullmatch(document_id):
            return None
        path = data_root / "processed" / "uploads" / "reports" / f"{document_id}.json"
        return path if path.is_file() else None

    @staticmethod
    def is_uploaded(document_id: str) -> bool:
        return _DOCUMENT_ID.fullmatch(document_id) is not None

    # ----------------------------------------------------------------------------- deletion

    async def delete_document(self, document_id: str) -> None:
        """Remove an uploaded document; the curated corpus cannot be deleted over HTTP."""
        if not self.is_uploaded(document_id):
            raise UploadError("Only uploaded documents can be deleted.", 403)
        if not (self.override_dir / f"{document_id}.override.json").is_file():
            raise UploadError("Unknown document.", 404)
        # Indexing may be running in another process (the durable worker), where this
        # process's lock cannot reach; deleting underneath it would strand its points.
        if any(
            job.document_id == document_id and job.status in {"queued", "processing"}
            for job in self.list_jobs(limit=1000)
        ):
            raise UploadError("This document is still being indexed; try again shortly.", 409)
        async with self._ingestion_lock:
            if self.point_remover is not None:
                await asyncio.to_thread(self.point_remover, document_id)
            self._discard_source(document_id)
            for job in self.list_jobs(limit=1000):
                if job.document_id == document_id:
                    (self.jobs_dir / f"{job.job_id}.json").unlink(missing_ok=True)
            for path in (
                self.generated_dir / f"{document_id}.json",
                self.data_root / "processed" / f"{document_id}.json",
                self.data_root / "processed" / "tables" / f"{document_id}.json",
                self.reports_dir / f"{document_id}.json",
            ):
                path.unlink(missing_ok=True)

    # ----------------------------------------------------------------------------- withdrawal

    def finish_withdrawals(self) -> None:
        """Retry every withdrawal whose cleanup could not finish earlier."""
        for document_id in withdrawn_document_ids(self.data_root):
            self._finish_withdrawal(document_id)

    def _discard_failed(self, document_id: str) -> None:
        # Record the withdrawal durably before touching anything: retrieval excludes the document
        # from now on, and a restart finishes the cleanup if this attempt is interrupted.
        _atomic_write(
            self.withdrawn_dir / f"{document_id}.json",
            json.dumps(
                {"document_id": document_id, "withdrawn_at": datetime.now(UTC).isoformat()}
            ).encode(),
        )
        self._finish_withdrawal(document_id)

    def _finish_withdrawal(self, document_id: str) -> bool:
        """Remove a withdrawn document's points, then its source. False while points remain."""
        with self._withdrawal_lock:
            marker = self.withdrawn_dir / f"{document_id}.json"
            if not marker.is_file():
                return True
            # Derived files go first: they list the document in the library and would make a
            # re-upload of the same PDF look like a duplicate.
            for path in (
                self.generated_dir / f"{document_id}.json",
                self.data_root / "processed" / f"{document_id}.json",
                self.data_root / "processed" / "tables" / f"{document_id}.json",
            ):
                path.unlink(missing_ok=True)
            if self.point_remover is not None:
                try:
                    self.point_remover(document_id)
                except Exception:
                    # The authoritative PDF stays until its indexed evidence is gone.
                    logger.warning("Removing points for withdrawn %s failed", document_id)
                    return False
            self._discard_source(document_id)
            marker.unlink(missing_ok=True)
            return True

    # ----------------------------------------------------------------------------- internals

    def _discard_source(self, document_id: str) -> None:
        (self.pdf_dir / f"{document_id}.pdf").unlink(missing_ok=True)
        (self.override_dir / f"{document_id}.override.json").unlink(missing_ok=True)

    def _document_with_checksum(self, checksum: str) -> DocumentManifest | None:
        if not self.generated_dir.is_dir():
            return None
        for path in self.generated_dir.glob("*.json"):
            try:
                manifest = DocumentManifest.model_validate_json(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if manifest.source_checksum == checksum:
                return manifest
        return None

    def _require(self, job_id: str) -> UploadJob:
        job = self.get(job_id)
        if job is None:
            raise UploadError("Unknown upload job.", 404)
        return job

    def _save(self, job: UploadJob) -> None:
        _atomic_write(self.jobs_dir / f"{job.job_id}.json", job.model_dump_json(indent=2).encode())

    def _update(self, job: UploadJob, **changes: object) -> UploadJob:
        updated = job.model_copy(update={**changes, "updated_at": datetime.now(UTC)})
        self._save(updated)
        return updated

    def _recover_interrupted_jobs(self) -> None:
        for job in self.list_jobs(limit=1000):
            if job.status in {"queued", "processing"}:
                self._discard_failed(job.document_id)
                self._update(
                    job,
                    status="failed",
                    detail="Processing was interrupted by a service restart. Please upload again.",
                )

    def _safe_failure(self, message: str) -> str:
        # Ingestion errors can quote absolute container paths; keep only the explanation.
        text = message.replace(str(self.data_root.resolve()), "").replace(str(self.data_root), "")
        text = re.sub(r"(/[\w.-]+)+/", "", text)
        return (text.strip() or "Ingestion failed.")[:400]
