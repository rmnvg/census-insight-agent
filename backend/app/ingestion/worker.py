"""Durable upload indexing: a Celery worker, with Redis as the broker, outside the API process.

Enabled by `INGESTION_BROKER_URL`. The API stages and validates the PDF exactly as before, then
enqueues the job instead of indexing it in-process. The worker runs the same
`UploadService.run_job`, so the citation-safe ingestion path is unchanged.

Delivery guarantees:

- Messages are acknowledged only after the job finishes (`acks_late`). A worker that dies
  mid-job leaves the message unacknowledged, and Redis redelivers it after the visibility
  timeout. Deterministic point IDs make the re-run an idempotent upsert.
- Errors while indexing retry with backoff up to `INGESTION_MAX_RETRIES`, keeping the source PDF.
  Only after the last attempt is the job failed and its source and points removed.
- One job runs at a time (`--concurrency=1`): jobs share one embedding cache and one collection.

Run it with `celery -A backend.app.ingestion.worker worker --concurrency=1 -Q ingestion`.
"""

import logging
import os
from functools import lru_cache

from celery import Celery, Task
from qdrant_client import QdrantClient

from backend.app.config import get_settings
from backend.app.ingestion.uploads import TransientIngestionError, UploadError, UploadService
from backend.app.retrieval.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

TASK_NAME = "census.ingest_upload"
QUEUE = "ingestion"
MAX_BACKOFF_SECONDS = 300


def create_app(broker_url: str, *, visibility_timeout_seconds: int = 1800) -> Celery:
    app = Celery("census-ingestion", broker=broker_url)
    app.conf.update(
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_default_queue=QUEUE,
        task_ignore_result=True,
        task_serializer="json",
        accept_content=["json"],
        worker_prefetch_multiplier=1,
        broker_connection_retry_on_startup=True,
        # Must exceed the longest job and retry backoff, or Redis redelivers a live job.
        broker_transport_options={"visibility_timeout": visibility_timeout_seconds},
    )
    return app


# Read from the environment rather than Settings so that importing this module (the API does, to
# enqueue) never requires the full Vertex configuration. The variable names match Settings.
app = create_app(
    os.environ.get("INGESTION_BROKER_URL") or "memory://",
    visibility_timeout_seconds=int(os.environ.get("INGESTION_VISIBILITY_TIMEOUT_SECONDS", "1800")),
)


@lru_cache
def _upload_service() -> UploadService:
    settings = get_settings()
    store = QdrantStore(
        QdrantClient(url=settings.qdrant_url),
        settings.qdrant_collection,
        dense_dimensions=settings.gemini_embedding_dimension,
        dense_model=settings.gemini_embedding_model,
        sparse_model=settings.sparse_embedding_model,
    )
    return UploadService(settings, point_remover=store.delete_document, recover_interrupted=False)


def retry_countdown(retries: int) -> int:
    return int(min(MAX_BACKOFF_SECONDS, 15 * 2**retries))


def run_upload_task(task: Task, job_id: str, uploads: UploadService, *, max_retries: int) -> str:
    """Body of the Celery task, separated so it can be tested without a broker."""
    final_attempt = task.request.retries >= max_retries
    try:
        job = uploads.run_job(job_id, retry_on_error=not final_attempt)
    except TransientIngestionError as error:
        logger.warning("Upload %s failed (attempt %s); retrying", job_id, task.request.retries + 1)
        raise task.retry(exc=error, countdown=retry_countdown(task.request.retries)) from error
    except UploadError:
        # The job was deleted while queued; there is nothing left to index.
        logger.info("Upload %s no longer exists; dropping it", job_id)
        return "missing"
    return job.status


@app.task(name=TASK_NAME, bind=True, max_retries=None)
def ingest_upload(self: Task, job_id: str) -> str:
    return run_upload_task(
        self, job_id, _upload_service(), max_retries=get_settings().ingestion_max_retries
    )


def enqueue_upload(job_id: str) -> None:
    app.send_task(TASK_NAME, args=[job_id], queue=QUEUE)
