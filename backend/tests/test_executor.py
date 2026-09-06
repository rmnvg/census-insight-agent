import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from backend.app.execution.client import (
    ArtifactStore,
    ExecutionQueueClient,
    ExecutorUnavailableError,
)
from backend.app.execution.contracts import (
    ArtifactDataset,
    ArtifactDescriptor,
    ExecutionRequest,
    ExecutionResult,
    ExpectedArtifact,
    ProducedArtifact,
    SourceManifest,
    SourceRecord,
)
from backend.app.execution.lineage import (
    DatasetValidationError,
    validate_artifact_lineage,
    validate_dataset,
)
from backend.app.main import app
from backend.app.retrieval.models import RetrievedEvidence
from executor.artifacts import ArtifactValidationError, validate_outputs
from executor.policy import ALLOWED_IMPORTS, validate_code
from executor.runner import STREAM_LIMIT, execute_request
from executor.worker import Worker


def source_record() -> SourceRecord:
    return SourceRecord(
        source_record_id="source-1",
        row_id="karnataka",
        field="value",
        raw_value="75.36",
        normalized_numeric_value=75.36,
        unit="percent",
        metric="literacy rate",
        region="Karnataka",
        year=2011,
        population_scope="persons",
        residence_scope="total",
        document_title="Karnataka report",
        document_id="doc-karnataka",
        page_number=50,
        chunk_id="chunk-karnataka",
        exact_supporting_quote="Karnataka literacy was 75.36 percent in 2011.",
        source_checksum="a" * 64,
    )


def dataset() -> ArtifactDataset:
    return ArtifactDataset(
        title="Literacy",
        task_type="artifact_table",
        rows=[{"row_id": "karnataka", "value": 75.36}],
        columns=["row_id", "value"],
        units={"value": "percent"},
        source_records=[source_record()],
        requested_output="table",
    )


def evidence() -> RetrievedEvidence:
    source = source_record()
    return RetrievedEvidence(
        chunk_id=source.chunk_id,
        text=source.exact_supporting_quote,
        document_title="Karnataka report",
        document_id=source.document_id,
        region="Karnataka",
        page_number=source.page_number,
        citation_snippet=source.exact_supporting_quote,
        section_path=["Literacy"],
        extraction_method="provided_markdown",
        coverage_status="indexed_provided_markdown",
        source_checksum=source.source_checksum,
        retrieval_score=1,
    )


def expected_table() -> list[ExpectedArtifact]:
    return [
        ExpectedArtifact(
            artifact_type="data",
            title="Data",
            filename="table.csv",
            media_type="text/csv",
            expected_columns=["row_id", "value"],
        ),
        ExpectedArtifact(
            artifact_type="table",
            title="Table",
            filename="table.md",
            media_type="text/markdown",
        ),
        ExpectedArtifact(
            artifact_type="manifest",
            title="Sources",
            filename="source-manifest.json",
            media_type="application/json",
        ),
    ]


def request(code: str, expected: list[ExpectedArtifact] | None = None) -> ExecutionRequest:
    source_manifest = SourceManifest(dataset_title="Literacy", source_records=[source_record()])
    return ExecutionRequest(
        job_id=str(uuid4()),
        session_id=str(uuid4()),
        run_id=str(uuid4()),
        code=code,
        code_sha256=hashlib.sha256(code.encode()).hexdigest(),
        input_data={
            "dataset": dataset().model_dump(mode="json"),
            "source_manifest": source_manifest.model_dump(mode="json"),
        },
        expected_artifacts=expected or expected_table(),
        timeout_seconds=1,
    )


TABLE_CODE = """
import json
from pathlib import Path
payload = json.loads(Path("input.json").read_text(encoding="utf-8"))
Path("output/table.csv").write_text("row_id,value\\nkarnataka,75.36\\n", encoding="utf-8")
Path("output/table.md").write_text(
    "| row_id | value |\\n|---|---:|\\n| karnataka | 75.36 |\\n", encoding="utf-8"
)
Path("output/source-manifest.json").write_text(
    json.dumps(payload["source_manifest"]), encoding="utf-8"
)
print("stdout-ok")
""".strip()


def test_execution_contracts_reject_unknown_fields_paths_and_invalid_ids() -> None:
    value = request(TABLE_CODE).model_dump(mode="json")
    value["unexpected"] = True
    with pytest.raises(ValidationError):
        ExecutionRequest.model_validate(value)
    value.pop("unexpected")
    value["job_id"] = "not-a-uuid"
    with pytest.raises(ValidationError):
        ExecutionRequest.model_validate(value)
    with pytest.raises(ValidationError):
        ExpectedArtifact(
            artifact_type="data",
            title="bad",
            filename="../escape.csv",
            media_type="text/csv",
        )


@pytest.mark.parametrize("module", sorted(ALLOWED_IMPORTS))
def test_ast_allows_artifact_imports(module: str) -> None:
    assert validate_code(f"import {module}").valid


@pytest.mark.parametrize(
    "module",
    [
        "os",
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "http",
        "multiprocessing",
        "threading",
        "ctypes",
        "importlib",
        "pickle",
    ],
)
def test_ast_rejects_every_forbidden_import(module: str) -> None:
    result = validate_code(f"import {module}")
    assert not result.valid
    assert "Forbidden import" in result.errors[0]


@pytest.mark.parametrize("call", ["eval('1')", "exec('x=1')", "compile('x', 'x', 'exec')"])
def test_ast_rejects_dangerous_builtins(call: str) -> None:
    assert not validate_code(call).valid


@pytest.mark.parametrize("code", ["value.__class__", "Path('../escape')", "Path('/etc/passwd')"])
def test_ast_rejects_dunder_traversal_and_absolute_paths(code: str) -> None:
    assert not validate_code(code).valid


@pytest.mark.parametrize(
    "code",
    [
        'Path("input.json").write_text("tamper")',
        'frame.to_csv("outside.csv")',
        'Path("output") / ".." / "other-job"',
    ],
)
def test_ast_rejects_writes_outside_declared_output(code: str) -> None:
    assert not validate_code(code).valid


def test_ast_allows_statically_known_output_path_aliases() -> None:
    code = """
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

chart_path = Path("output/chart.png")
csv_path = Path("output/plotted-data.csv")
manifest_path = Path("output/source-manifest.json")
chart_path.parent.mkdir(parents=True, exist_ok=True)
frame = pd.DataFrame([{"value": 1}])
plt.savefig(chart_path)
frame.to_csv(csv_path, index=False)
Path(manifest_path).write_text("{}")
"""
    assert validate_code(code).valid


def test_ast_allows_open_only_for_static_job_input_and_outputs() -> None:
    code = """
from pathlib import Path
import json

input_path = Path("input.json")
manifest_path = Path("output/source-manifest.json")
with open(input_path, "r") as source:
    payload = json.load(source)
with open(manifest_path, "w") as destination:
    json.dump(payload["source_manifest"], destination)
"""
    assert validate_code(code).valid


@pytest.mark.parametrize(
    "code",
    [
        'open("/etc/passwd", "r")',
        'open("other-input.json", "r")',
        'open("input.json", "w")',
        'open("output/result.json", "r")',
        'open("output/result.json", "a")',
        'open("output/result.json", "r+")',
        'open(dynamic_path, "w")',
        'open("output/result.json", dynamic_mode)',
    ],
)
def test_ast_rejects_arbitrary_open_paths_and_modes(code: str) -> None:
    assert not validate_code(code).valid


@pytest.mark.parametrize(
    "code",
    [
        'path = Path("output/chart.png")\npath = user_path\nfigure.savefig(path)',
        'path = Path("output/table.csv")\ndef write(path):\n    frame.to_csv(path)',
        'path = Path("output/table.csv")\n'
        "if condition:\n"
        "    path = dynamic\n"
        "else:\n"
        '    path = Path("output/table.csv")\n'
        "frame.to_csv(path)",
        'path = Path("outside.csv")\nframe.to_csv(path)',
        'path = Path("output/table.csv")\npath /= "../escape.csv"\nframe.to_csv(path)',
        'Path("output/subdirectory").mkdir()',
    ],
)
def test_ast_rejects_unproven_rebound_or_non_output_path_aliases(code: str) -> None:
    assert not validate_code(code).valid


def test_atomic_submission_and_worker_round_trip(tmp_path: Path) -> None:
    value = request(TABLE_CODE)
    client = ExecutionQueueClient(tmp_path)
    path = client.submit(value)
    assert path.is_file()
    assert not list(path.parent.glob("*.tmp"))
    assert Worker(tmp_path).run_once()
    result = asyncio.run(client.wait(value.job_id, 1))
    assert result.status == "succeeded"
    assert result.stdout.strip() == "stdout-ok"
    assert {item.filename for item in result.artifacts} == {
        "table.csv",
        "table.md",
        "source-manifest.json",
    }


def test_missing_executor_result_is_bounded_and_typed(tmp_path: Path) -> None:
    client = ExecutionQueueClient(tmp_path, poll_seconds=0.001)
    with pytest.raises(ExecutorUnavailableError):
        asyncio.run(client.wait(str(uuid4()), 0.01))
    assert client.heartbeat_age_seconds() is None


def test_runtime_error_stderr_nonzero_and_timeout(tmp_path: Path) -> None:
    failed = execute_request(request("raise RuntimeError('boom')"), tmp_path / "failed")
    assert failed.status == "failed"
    assert failed.exit_code != 0
    assert failed.error_code == "EXECUTION_FAILED"
    assert "RuntimeError: boom" in failed.stderr
    timeout_request = request("while True:\n    pass").model_copy(update={"timeout_seconds": 0.1})
    timed_out = execute_request(timeout_request, tmp_path / "timeout")
    assert timed_out.error_code == "EXECUTION_TIMEOUT"
    assert timed_out.timed_out


def test_stdout_and_stderr_limits_are_enforced(tmp_path: Path) -> None:
    code = f"print('x' * {STREAM_LIMIT + 1000})"
    result = execute_request(request(code), tmp_path / "large")
    assert result.error_code == "OUTPUT_LIMIT_EXCEEDED"
    assert result.output_truncated
    assert len(result.stdout.encode()) == STREAM_LIMIT


def test_output_allowlist_symlink_and_png_validation(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "unexpected.exe").write_bytes(b"x")
    with pytest.raises(ArtifactValidationError):
        validate_outputs(output, [])
    (output / "unexpected.exe").unlink()
    (output / "table.csv").symlink_to(tmp_path / "target")
    with pytest.raises(ArtifactValidationError, match="Non-regular"):
        validate_outputs(output, [expected_table()[0]])
    (output / "table.csv").unlink()
    (output / "chart.png").write_bytes(b"fake")
    png = ExpectedArtifact(
        artifact_type="chart", title="Chart", filename="chart.png", media_type="image/png"
    )
    with pytest.raises(ArtifactValidationError, match="PNG"):
        validate_outputs(output, [png])
    Image.new("RGB", (100, 80), "white").save(output / "chart.png")
    assert validate_outputs(output, [png])[0].media_type == "image/png"


def test_csv_schema_and_manifest_validation(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "table.csv").write_text("wrong,value\nkarnataka,75.36\n", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="schema"):
        validate_outputs(output, [expected_table()[0]])
    (output / "table.csv").unlink()
    (output / "source-manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="manifest"):
        validate_outputs(output, [expected_table()[-1]])


def test_dataset_and_companion_csv_lineage(tmp_path: Path) -> None:
    value = dataset()
    validate_dataset(value, [evidence()])
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "table.csv").write_text("row_id,value\nkarnataka,75.36\n", encoding="utf-8")
    (root / "source-manifest.json").write_text(
        SourceManifest(
            dataset_title=value.title, source_records=value.source_records
        ).model_dump_json(),
        encoding="utf-8",
    )
    validate_artifact_lineage(value, root)
    (root / "table.csv").write_text("row_id,value\nkarnataka,99\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="values differ"):
        validate_artifact_lineage(value, root)
    bad = value.model_copy(
        update={
            "source_records": [
                value.source_records[0].model_copy(update={"source_checksum": "b" * 64})
            ]
        }
    )
    with pytest.raises(DatasetValidationError, match="provenance"):
        validate_dataset(bad, [evidence()])


def test_artifact_store_enforces_session_isolation_and_hides_code(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    queue = workspace / "execution-queue"
    value = request(TABLE_CODE)
    staged = queue / "jobs" / value.job_id
    output = staged / "output"
    output.mkdir(parents=True)
    (staged / "generated.py").write_text(TABLE_CODE, encoding="utf-8")
    (staged / "input.json").write_text("{}", encoding="utf-8")
    artifacts = []
    for filename, media in (
        ("table.csv", "text/csv"),
        ("table.md", "text/markdown"),
        ("source-manifest.json", "application/json"),
    ):
        path = output / filename
        path.write_text("content", encoding="utf-8")
        artifacts.append(
            ProducedArtifact(
                filename=filename,
                media_type=media,
                byte_size=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    now = datetime.now(UTC)
    result = ExecutionResult(
        job_id=value.job_id,
        status="succeeded",
        exit_code=0,
        started_at=now,
        completed_at=now,
        duration_ms=1,
        artifacts=artifacts,
    )
    store = ArtifactStore(workspace, queue)
    descriptor = store.accept(
        value,
        result,
        artifact_type="table",
        title="Literacy",
        primary_filename="table.md",
    )
    assert store.get(value.session_id, descriptor.artifact_id) == descriptor
    assert store.get(str(uuid4()), descriptor.artifact_id) is None
    assert store.public_file(value.session_id, descriptor.artifact_id, "table.csv") is not None
    assert store.public_file(value.session_id, descriptor.artifact_id, "generated.py") is None
    assert (
        workspace
        / "sessions"
        / value.session_id
        / "artifacts"
        / descriptor.artifact_id
        / "generated.py"
    ).is_file()


def test_artifact_api_serves_only_manifested_session_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    session_id = str(uuid4())
    artifact_id = str(uuid4())
    run_id = str(uuid4())
    root = workspace / "sessions" / session_id / "artifacts" / artifact_id
    root.mkdir(parents=True)
    (root / "table.csv").write_text("row_id,value\nkarnataka,75.36\n", encoding="utf-8")
    (root / "generated.py").write_text("print('private')", encoding="utf-8")
    produced = ProducedArtifact(
        filename="table.csv",
        media_type="text/csv",
        byte_size=(root / "table.csv").stat().st_size,
        sha256=hashlib.sha256((root / "table.csv").read_bytes()).hexdigest(),
    )
    now = datetime.now(UTC)
    (root / "execution-result.json").write_text(
        ExecutionResult(
            job_id=str(uuid4()),
            status="succeeded",
            exit_code=0,
            started_at=now,
            completed_at=now,
            duration_ms=1,
            artifacts=[produced],
        ).model_dump_json(),
        encoding="utf-8",
    )
    descriptor = ArtifactDescriptor(
        artifact_id=artifact_id,
        artifact_type="table",
        title="Literacy",
        filename="table.csv",
        media_type="text/csv",
        byte_size=produced.byte_size,
        sha256=produced.sha256,
        session_id=session_id,
        run_id=run_id,
        source_manifest_path="source-manifest.json",
        download_url=f"/sessions/{session_id}/artifacts/{artifact_id}/files/table.csv",
    )
    (root / "artifact.json").write_text(descriptor.model_dump_json(), encoding="utf-8")

    class Service:
        artifacts = ArtifactStore(workspace, workspace / "execution-queue")

        async def get_session(self, requested: str) -> object:
            return object()

    monkeypatch.setattr("backend.app.api.get_agent_service", lambda: Service())
    client = TestClient(app)
    assert client.get(f"/sessions/{session_id}/artifacts").json()["artifacts"]
    assert client.get(f"/sessions/{session_id}/artifacts/{artifact_id}").status_code == 200
    download = client.get(descriptor.download_url)
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("text/csv")
    assert (
        client.get(f"/sessions/{session_id}/artifacts/{artifact_id}/files/generated.py").status_code
        == 404
    )
    assert (
        client.get(f"/sessions/{uuid4()}/artifacts/{artifact_id}/files/table.csv").status_code
        == 404
    )
