import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread
from uuid import UUID

from backend.app.execution.contracts import ExecutionRequest, ExecutionResult
from executor.runner import execute_request


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
                job_id = claimed.stem
                try:
                    request = ExecutionRequest.model_validate_json(
                        claimed.read_text(encoding="utf-8")
                    )
                    job_id = request.job_id
                except Exception:
                    pass
                try:
                    UUID(job_id)
                except ValueError:
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
