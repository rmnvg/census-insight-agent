"""The executor's single consumer of the filesystem job queue.

A job's state is the directory holding its request or result:

- `inbox/<job_id>.json`: submitted by the backend, not yet claimed.
- `processing/<job_id>.json`: claimed by a running worker; its outputs are staged under
  `jobs/<job_id>/output/` and are partial until a result exists.
- `results/<job_id>.json`: finished, succeeded or failed. The backend reads and removes it, then
  accepts or discards `jobs/<job_id>/`.

Exactly one worker consumes the queue, so at startup anything in `processing/` was interrupted.
"""

import argparse
import json
import logging
import os
import shutil
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread
from uuid import UUID

from backend.app.execution.contracts import ExecutionRequest, ExecutionResult
from executor.runner import execute_request

logger = logging.getLogger(__name__)


def atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class Worker:
    def __init__(self, root: Path, *, poll_seconds: float = 0.2) -> None:
        self.root = root
        self.poll_seconds = poll_seconds
        self.inbox = root / "inbox"
        self.processing = root / "processing"
        self.results = root / "results"
        self.jobs = root / "jobs"
        for path in (self.inbox, self.processing, self.results, self.jobs):
            path.mkdir(parents=True, exist_ok=True)

    def heartbeat(self) -> None:
        atomic_json(
            self.root / "heartbeat.json",
            {"status": "ok", "updated_at": datetime.now(UTC).isoformat()},
        )

    def recover_interrupted(self) -> int:
        """Fail every job a previous worker claimed but never finished.

        Such a job is failed, not run again: it may be what killed the worker (an OOM kill takes
        the whole container), and re-running it would crash-loop the only consumer. Its partial
        outputs were never validated, so they are removed rather than returned.
        """
        recovered = 0
        for claimed in sorted(self.processing.glob("*.json")):
            job_id = self._job_id(claimed)
            if job_id is not None:
                partial = self.jobs / job_id
                if partial.is_symlink():
                    partial.unlink()
                elif partial.is_dir():
                    shutil.rmtree(partial)
                now = datetime.now(UTC)
                result = ExecutionResult(
                    job_id=job_id,
                    status="failed",
                    started_at=now,
                    completed_at=now,
                    duration_ms=0,
                    error_code="EXECUTION_INTERRUPTED",
                    stderr="The executor restarted before this job finished.",
                )
                atomic_json(self.results / f"{job_id}.json", result.model_dump(mode="json"))
                recovered += 1
            claimed.unlink(missing_ok=True)
        if recovered:
            logger.warning("Failed %s job(s) interrupted by an executor restart", recovered)
        return recovered

    @staticmethod
    def _job_id(claimed: Path) -> str | None:
        """The claimed request's job ID, or None when it has no valid one to report under."""
        job_id = claimed.stem
        with suppress(Exception):
            request = ExecutionRequest.model_validate_json(claimed.read_text(encoding="utf-8"))
            job_id = request.job_id
        try:
            UUID(job_id)
        except ValueError:
            return None
        return job_id

    def run_once(self) -> bool:
        self.heartbeat()
        for request_path in sorted(self.inbox.glob("*.json")):
            claimed = self.processing / request_path.name
            try:
                request_path.replace(claimed)
            except FileNotFoundError:
                continue
            try:
                request = ExecutionRequest.model_validate_json(claimed.read_text(encoding="utf-8"))
                stopped = Event()

                def keep_alive(stopped: Event = stopped) -> None:
                    while not stopped.wait(2):
                        self.heartbeat()

                heartbeat_thread = Thread(target=keep_alive, daemon=True)
                heartbeat_thread.start()
                try:
                    result = execute_request(request, self.jobs / request.job_id)
                finally:
                    stopped.set()
                    heartbeat_thread.join()
            except Exception as error:
                now = datetime.now(UTC)
                job_id = self._job_id(claimed)
                if job_id is None:
                    claimed.unlink(missing_ok=True)
                    self.heartbeat()
                    return True
                result = ExecutionResult(
                    job_id=job_id,
                    status="failed",
                    started_at=now,
                    completed_at=now,
                    duration_ms=0,
                    error_code="JOB_PROTOCOL_ERROR",
                    stderr=f"Worker protocol error: {type(error).__name__}",
                )
            atomic_json(self.results / f"{result.job_id}.json", result.model_dump(mode="json"))
            claimed.unlink(missing_ok=True)
            self.heartbeat()
            return True
        return False

    def serve(self) -> None:
        self.recover_interrupted()
        while True:
            if not self.run_once():
                time.sleep(self.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated filesystem-queue executor")
    parser.add_argument("--root", type=Path, default=Path("/execution"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    worker = Worker(args.root)
    if args.once:
        worker.run_once()
    else:
        worker.serve()


if __name__ == "__main__":
    main()
