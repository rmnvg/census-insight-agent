import asyncio
import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from backend.app.execution.contracts import (
    ArtifactDescriptor,
    ArtifactListing,
    ExecutionRequest,
    ExecutionResult,
    ExpectedArtifact,
)


class ExecutorUnavailableError(RuntimeError):
    pass


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class ExecutionQueueClient:
    def __init__(self, root: Path, *, poll_seconds: float = 0.1) -> None:
        self.root = root
        self.poll_seconds = poll_seconds

    def submit(self, request: ExecutionRequest) -> Path:
        inbox = self.root / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        path = inbox / f"{request.job_id}.json"
        if path.exists():
            raise ValueError("Duplicate execution job ID")
        atomic_json(path, request.model_dump(mode="json"))
        return path

    async def wait(self, job_id: str, timeout_seconds: float) -> ExecutionResult:
        UUID(job_id)
        result_path = self.root / "results" / f"{job_id}.json"
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            if result_path.is_file():
                result = ExecutionResult.model_validate_json(
                    await asyncio.to_thread(result_path.read_text, encoding="utf-8")
                )
                await asyncio.to_thread(result_path.unlink)
                return result
            await asyncio.sleep(self.poll_seconds)
        raise ExecutorUnavailableError("Executor did not return a result before the deadline")

    def heartbeat_age_seconds(self) -> float | None:
        path = self.root / "heartbeat.json"
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))["updated_at"]
            updated = datetime.fromisoformat(value)
            return max(0.0, (datetime.now(UTC) - updated).total_seconds())
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None


class ArtifactStore:
    def __init__(self, workspace_root: Path, queue_root: Path) -> None:
        self.workspace_root = workspace_root
        self.queue_root = queue_root

    def accept(
        self,
        request: ExecutionRequest,
        result: ExecutionResult,
        *,
        artifact_type: Literal["chart", "table"],
        title: str,
        primary_filename: str,
    ) -> ArtifactDescriptor:
        if result.status != "succeeded":
            raise ValueError("Cannot accept artifacts from a failed execution")
        artifact_id = str(uuid4())
        source = self.queue_root / "jobs" / request.job_id
        destination = (
            self.workspace_root / "sessions" / request.session_id / "artifacts" / artifact_id
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not source.is_dir() or destination.exists():
            raise ValueError("Execution artifact staging directory is unavailable")
        os.replace(source, destination)
        output = destination / "output"
        for path in output.iterdir():
            os.replace(path, destination / path.name)
        output.rmdir()
        atomic_json(destination / "execution-result.json", result.model_dump(mode="json"))
        produced = next(item for item in result.artifacts if item.filename == primary_filename)
        descriptor = ArtifactDescriptor(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            title=title,
            filename=primary_filename,
            media_type=produced.media_type,
            byte_size=produced.byte_size,
            sha256=produced.sha256,
            session_id=request.session_id,
            run_id=request.run_id,
            source_manifest_path="source-manifest.json",
            download_url=(
                f"/sessions/{request.session_id}/artifacts/{artifact_id}/files/{primary_filename}"
            ),
        )
        atomic_json(destination / "artifact.json", descriptor.model_dump(mode="json"))
        return descriptor

    def list(self, session_id: str) -> ArtifactListing:
        UUID(session_id)
        root = self.workspace_root / "sessions" / session_id / "artifacts"
        descriptors = []
        for path in sorted(root.glob("*/artifact.json")) if root.is_dir() else []:
            try:
                descriptor = ArtifactDescriptor.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                continue
            if descriptor.session_id == session_id:
                descriptors.append(descriptor)
        return ArtifactListing(artifacts=descriptors)

    def get(self, session_id: str, artifact_id: str) -> ArtifactDescriptor | None:
        UUID(session_id)
        UUID(artifact_id)
        path = (
            self.workspace_root
            / "sessions"
            / session_id
            / "artifacts"
            / artifact_id
            / "artifact.json"
        )
        if not path.is_file():
            return None
        descriptor = ArtifactDescriptor.model_validate_json(path.read_text(encoding="utf-8"))
        return descriptor if descriptor.session_id == session_id else None

    def public_file(self, session_id: str, artifact_id: str, filename: str) -> Path | None:
        descriptor = self.get(session_id, artifact_id)
        if descriptor is None or filename.startswith(".") or Path(filename).name != filename:
            return None
        root = self.workspace_root / "sessions" / session_id / "artifacts" / artifact_id
        result = ExecutionResult.model_validate_json(
            (root / "execution-result.json").read_text(encoding="utf-8")
        )
        allowed = {item.filename for item in result.artifacts}
        if filename not in allowed or filename in {"generated.py", "input.json"}:
            return None
        path = root / filename
        if not path.is_file() or path.is_symlink():
            return None
        return path

    def discard(self, job_id: str) -> None:
        UUID(job_id)
        path = self.queue_root / "jobs" / job_id
        if path.is_dir():
            shutil.rmtree(path)


def new_request(
    session_id: str,
    run_id: str,
    code: str,
    input_data: dict[str, object],
    expected_artifacts: Sequence[ExpectedArtifact],
    timeout_seconds: float,
) -> ExecutionRequest:
    return ExecutionRequest(
        job_id=str(uuid4()),
        session_id=session_id,
        run_id=run_id,
        code=code,
        code_sha256=hashlib.sha256(code.encode()).hexdigest(),
        input_data=input_data,
        expected_artifacts=list(expected_artifacts),
        timeout_seconds=timeout_seconds,
    )
