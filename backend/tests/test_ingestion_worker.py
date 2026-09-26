"""Durable upload worker: retry semantics, idempotent redelivery, and the real Celery task path.

The Celery test runs a real worker thread. It uses Redis when TEST_REDIS_URL is set and Celery's
in-memory broker otherwise, so the task wiring is always exercised.
"""

import os
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from celery.contrib.testing.worker import start_worker
from fastapi.testclient import TestClient

from backend.app.ingestion import worker
from backend.app.ingestion.uploads import TransientIngestionError, UploadService
from backend.app.ingestion.worker import run_upload_task
from backend.app.main import app
from backend.tests.test_uploads import FakeIngestion, pdf_bytes, service, stage, upload_settings


class FlakyIngestion(FakeIngestion):
    """Raises on the first `failures` calls, then indexes normally."""

    def __init__(self, failures: int) -> None:
        super().__init__(chunks=3)
        self.failures = failures

    def ingest(self, **kwargs: Any) -> Any:
        if len(self.calls) < self.failures:
            self.calls.append(kwargs.get("document_id"))
            raise RuntimeError("503 UNAVAILABLE from the embedding endpoint")
        return super().ingest(**kwargs)


class FakeTask:
    def __init__(self, retries: int) -> None:
        self.request = SimpleNamespace(retries=retries)
        self.countdowns: list[int] = []

    def retry(self, *, exc: BaseException, countdown: int) -> Exception:
        del exc
        self.countdowns.append(countdown)
        return RuntimeError("retry scheduled")


def source_pdf(tmp_path: Path, document_id: str) -> Path:
    return tmp_path / "data" / "source" / "pdf" / f"{document_id}.pdf"


def test_retryable_failure_keeps_the_source_and_requeues(tmp_path: Path) -> None:
    uploads, removed = service(tmp_path, FakeIngestion(raises=True))
    job = stage(uploads)
    with pytest.raises(TransientIngestionError) as caught:
        uploads.run_job(job.job_id, retry_on_error=True)
    assert "/app/data" not in str(caught.value)
    current = uploads.get(job.job_id)
    assert current is not None and current.status == "queued"
    assert removed == []
    assert source_pdf(tmp_path, job.document_id).is_file()


def test_task_retries_with_backoff_then_fails_cleanly_on_the_last_attempt(tmp_path: Path) -> None:
    uploads, removed = service(tmp_path, FakeIngestion(raises=True))
    job = stage(uploads)
    first = FakeTask(retries=0)
    with pytest.raises(RuntimeError, match="retry scheduled"):
        run_upload_task(first, job.job_id, uploads, max_retries=2)
    assert first.countdowns == [15]
    assert worker.retry_countdown(10) == worker.MAX_BACKOFF_SECONDS

    last = FakeTask(retries=2)
    status = run_upload_task(last, job.job_id, uploads, max_retries=2)
    assert status == "failed" and last.countdowns == []
    assert removed == [job.document_id]
    assert not source_pdf(tmp_path, job.document_id).exists()


def test_redelivered_message_for_a_finished_job_does_not_reindex(tmp_path: Path) -> None:
    fake = FakeIngestion()
    uploads, _ = service(tmp_path, fake)
    job = stage(uploads)
    assert uploads.run_job(job.job_id).status == "succeeded"
    assert uploads.run_job(job.job_id).status == "succeeded"
    assert fake.calls == [job.document_id]


def test_redelivery_after_a_crash_mid_job_reindexes(tmp_path: Path) -> None:
    fake = FakeIngestion()
    uploads, _ = service(tmp_path, fake)
    job = stage(uploads)
    # A worker died after marking the job processing; the queue hands it to another worker.
    uploads._update(job, status="processing")
    assert run_upload_task(FakeTask(0), job.job_id, uploads, max_retries=3) == "succeeded"
    assert fake.calls == [job.document_id]


def test_deleted_job_is_dropped(tmp_path: Path) -> None:
    uploads, _ = service(tmp_path)
    assert run_upload_task(FakeTask(0), "0" * 32, uploads, max_retries=3) == "missing"


def test_worker_mode_does_not_fail_queued_jobs_when_the_api_restarts(tmp_path: Path) -> None:
    uploads, _ = service(tmp_path)
    job = stage(uploads)
    restarted = UploadService(upload_settings(tmp_path), recover_interrupted=False)
    current = restarted.get(job.job_id)
    assert current is not None and current.status == "queued"


def test_document_cannot_be_deleted_while_it_is_being_indexed(tmp_path: Path) -> None:
    import asyncio

    uploads, removed = service(tmp_path)
    job = stage(uploads)
    with pytest.raises(Exception) as caught:
        asyncio.run(uploads.delete_document(job.document_id))
    assert getattr(caught.value, "status_code", None) == 409
    assert removed == []


def test_http_upload_is_enqueued_in_worker_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeIngestion()
    uploads, _ = service(tmp_path, fake)
    configured = upload_settings(tmp_path, ingestion_broker_url="redis://queue:6379/0")
    monkeypatch.setattr("backend.app.api.get_settings", lambda: configured)
    monkeypatch.setattr("backend.app.api.get_upload_service", lambda: uploads)
    enqueued: list[str] = []
    monkeypatch.setattr("backend.app.api.enqueue_upload", enqueued.append)
    client = TestClient(app)

    response = client.post(
        "/documents/upload",
        files={"file": ("kerala.pdf", pdf_bytes(), "application/pdf")},
        data={"title": "Kerala highlights", "region": "Kerala"},
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert enqueued == [job_id]
    # Nothing ran in the API process: the worker owns the job.
    assert fake.calls == []
    assert client.get(f"/documents/uploads/{job_id}").json()["status"] == "queued"


def test_http_upload_fails_cleanly_when_the_queue_is_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploads, removed = service(tmp_path)
    configured = upload_settings(tmp_path, ingestion_broker_url="redis://queue:6379/0")
    monkeypatch.setattr("backend.app.api.get_settings", lambda: configured)
    monkeypatch.setattr("backend.app.api.get_upload_service", lambda: uploads)

    def unavailable(job_id: str) -> None:
        raise ConnectionError(f"cannot reach broker for {job_id}")

    monkeypatch.setattr("backend.app.api.enqueue_upload", unavailable)
    response = TestClient(app).post(
        "/documents/upload",
        files={"file": ("kerala.pdf", pdf_bytes(), "application/pdf")},
        data={"title": "Kerala highlights", "region": "Kerala"},
    )
    assert response.status_code == 503
    [job] = uploads.list_jobs()
    assert job.status == "failed"
    assert removed == [job.document_id]
    assert not source_pdf(tmp_path, job.document_id).exists()


@pytest.fixture
def celery_broker() -> Iterator[str]:
    url = os.environ.get("TEST_REDIS_URL") or "memory://"
    previous = worker.app.conf.broker_url
    worker.app.conf.broker_url = url
    worker.app.conf.broker_transport_options = {"visibility_timeout": 60}
    yield url
    worker.app.conf.broker_url = previous


def test_real_worker_retries_a_transient_failure_then_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, celery_broker: str
) -> None:
    del celery_broker
    flaky = FlakyIngestion(failures=1)
    uploads, removed = service(tmp_path, flaky)
    job = stage(uploads)
    monkeypatch.setattr(worker, "_upload_service", lambda: uploads)
    monkeypatch.setattr(worker, "retry_countdown", lambda retries: 0)
    # The task reads its retry limit from Settings; CI has no .env to supply GOOGLE_CLOUD_PROJECT.
    monkeypatch.setattr(worker, "get_settings", lambda: upload_settings(tmp_path))

    with start_worker(worker.app, perform_ping_check=False, shutdown_timeout=30):
        worker.enqueue_upload(job.job_id)
        deadline = time.monotonic() + 30
        current = uploads.get(job.job_id)
        while current is not None and current.status != "succeeded":
            assert time.monotonic() < deadline, f"job stuck in {current.status}"
            time.sleep(0.1)
            current = uploads.get(job.job_id)

    assert flaky.calls == [job.document_id, job.document_id]
    assert current is not None and current.chunks == 3
    assert removed == []
