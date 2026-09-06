import hashlib
import json
import os
import resource
import selectors
import signal
import subprocess
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from backend.app.execution.contracts import ExecutionRequest, ExecutionResult
from executor.artifacts import ArtifactValidationError, validate_outputs
from executor.policy import validate_code

STREAM_LIMIT = 64 * 1024


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    # macOS does not support this Linux address-space limit reliably. Production runs in
    # Linux Docker with both RLIMIT_AS and the container memory limit; host runs are dev-only.
    if sys.platform == "linux":
        resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (24 * 1024 * 1024, 24 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    os.setsid()


def _capture(process: subprocess.Popen[bytes], timeout: float) -> tuple[bytes, bytes, bool, bool]:
    """Drain both pipes with bounded buffers, killing the process group on limit/deadline."""
    streams = [bytearray(), bytearray()]
    deadline = time.monotonic() + timeout
    timed_out = exceeded = False
    with selectors.DefaultSelector() as selector:
        assert process.stdout is not None and process.stderr is not None
        selector.register(process.stdout, selectors.EVENT_READ, 0)
        selector.register(process.stderr, selectors.EVENT_READ, 1)
        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                for key, _ in selector.select(min(remaining, 0.1)):
                    block = os.read(key.fd, 8192)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    buffer = streams[key.data]
                    available = STREAM_LIMIT - len(buffer)
                    buffer.extend(block[:available])
                    if len(block) > available:
                        exceeded = True
                        break
                if exceeded:
                    break
            if not timed_out and not exceeded:
                try:
                    process.wait(timeout=max(0.001, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    timed_out = True
        finally:
            if timed_out or exceeded or process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            process.stdout.close()
            process.stderr.close()
    return bytes(streams[0]), bytes(streams[1]), timed_out, exceeded


def _bounded(value: bytes) -> tuple[str, bool]:
    truncated = len(value) > STREAM_LIMIT
    return value[:STREAM_LIMIT].decode("utf-8", errors="replace"), truncated


def execute_request(request: ExecutionRequest, job_dir: Path) -> ExecutionResult:
    started_at = datetime.now(UTC)
    started = time.monotonic()
    try:
        if request.code_sha256 != hashlib.sha256(request.code.encode()).hexdigest():
            return _failure(
                request, started_at, started, "JOB_PROTOCOL_ERROR", "Code hash mismatch"
            )
        policy = validate_code(request.code)
        if not policy.valid:
            return _failure(
                request,
                started_at,
                started,
                "CODE_POLICY_VIOLATION",
                "; ".join(policy.errors),
            )
        job_dir.mkdir(parents=True, exist_ok=False)
        output_dir = job_dir / "output"
        output_dir.mkdir()
        (job_dir / "generated.py").write_text(request.code, encoding="utf-8")
        (job_dir / "input.json").write_text(
            json.dumps(request.input_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        environment = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "PYTHONPATH": "",
            "PYTHONHASHSEED": "0",
            "MPLBACKEND": "Agg",
            "HOME": str(job_dir),
            "TMPDIR": "/tmp",
        }
        process = subprocess.Popen(
            [sys.executable, "-I", "generated.py"],
            cwd=job_dir,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=_limits,
        )
        stdout_bytes, stderr_bytes, timed_out, exceeded = _capture(process, request.timeout_seconds)
        stdout, stdout_truncated = _bounded(stdout_bytes)
        stderr, stderr_truncated = _bounded(stderr_bytes)
        if timed_out:
            return ExecutionResult(
                job_id=request.job_id,
                status="failed",
                exit_code=None,
                stdout=stdout,
                stderr=stderr,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                duration_ms=(time.monotonic() - started) * 1000,
                timed_out=True,
                error_code="EXECUTION_TIMEOUT",
                output_truncated=stdout_truncated or stderr_truncated,
            )
        if exceeded or stdout_truncated or stderr_truncated:
            return ExecutionResult(
                job_id=request.job_id,
                status="failed",
                exit_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                duration_ms=(time.monotonic() - started) * 1000,
                error_code="OUTPUT_LIMIT_EXCEEDED",
                output_truncated=True,
            )
        if process.returncode != 0:
            return ExecutionResult(
                job_id=request.job_id,
                status="failed",
                exit_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                duration_ms=(time.monotonic() - started) * 1000,
                error_code="EXECUTION_FAILED",
            )
        try:
            artifacts = validate_outputs(output_dir, request.expected_artifacts)
        except ArtifactValidationError as error:
            code = (
                "OUTPUT_LIMIT_EXCEEDED"
                if error.output_limit
                else "ARTIFACT_NOT_CREATED"
                if error.missing
                else "INVALID_ARTIFACT"
            )
            return _failure(request, started_at, started, code, str(error), exit_code=0)
        return ExecutionResult(
            job_id=request.job_id,
            status="succeeded",
            exit_code=0,
            stdout=stdout,
            stderr=stderr,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            duration_ms=(time.monotonic() - started) * 1000,
            artifacts=artifacts,
        )
    except Exception as error:
        return _failure(
            request,
            started_at,
            started,
            "JOB_PROTOCOL_ERROR",
            f"Worker error: {type(error).__name__}",
        )


def _failure(
    request: ExecutionRequest,
    started_at: datetime,
    started: float,
    code: str,
    stderr: str,
    *,
    exit_code: int | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        job_id=request.job_id,
        status="failed",
        exit_code=exit_code,
        stderr=stderr[:STREAM_LIMIT],
        started_at=started_at,
        completed_at=datetime.now(UTC),
        duration_ms=(time.monotonic() - started) * 1000,
        error_code=code,  # type: ignore[arg-type]
    )
