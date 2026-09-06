import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from backend.app.execution.contracts import ExecutionRequest, ExecutionResult
from executor.artifacts import ArtifactValidationError, validate_outputs
from executor.policy import validate_code

STREAM_LIMIT = 64 * 1024


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (24 * 1024 * 1024, 24 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    os.setsid()


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
        try:
            stdout_bytes, stderr_bytes = process.communicate(timeout=request.timeout_seconds)
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGKILL)
            final_stdout, final_stderr = process.communicate()
            stdout, stdout_truncated = _bounded((error.stdout or b"") + final_stdout)
            stderr, stderr_truncated = _bounded((error.stderr or b"") + final_stderr)
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
        stdout, stdout_truncated = _bounded(stdout_bytes)
        stderr, stderr_truncated = _bounded(stderr_bytes)
        if stdout_truncated or stderr_truncated:
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
