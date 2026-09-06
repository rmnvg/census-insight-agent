from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

ExecutionErrorCode = Literal[
    "CODE_POLICY_VIOLATION",
    "EXECUTION_TIMEOUT",
    "EXECUTION_FAILED",
    "OUTPUT_LIMIT_EXCEEDED",
    "INVALID_ARTIFACT",
    "ARTIFACT_NOT_CREATED",
    "JOB_PROTOCOL_ERROR",
    "EXECUTOR_UNAVAILABLE",
]
ArtifactType = Literal["chart", "table", "data", "manifest"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _uuid(value: str) -> str:
    UUID(value)
    return value


def _filename(value: str) -> str:
    if not value or value.startswith(".") or "/" in value or "\\" in value or ".." in value:
        raise ValueError("Filename must be a plain non-hidden basename")
    return value


class ExpectedArtifact(StrictModel):
    artifact_type: ArtifactType
    title: str = Field(min_length=1, max_length=200)
    filename: str
    media_type: str
    expected_columns: list[str] = Field(default_factory=list)

    _valid_filename = field_validator("filename")(_filename)


class ExecutionRequest(StrictModel):
    job_id: str
    session_id: str
    run_id: str
    code: str = Field(min_length=1, max_length=100_000)
    code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_data: dict[str, object]
    expected_artifacts: list[ExpectedArtifact] = Field(min_length=1, max_length=8)
    timeout_seconds: float = Field(gt=0, le=60)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    _valid_job = field_validator("job_id")(_uuid)
    _valid_session = field_validator("session_id")(_uuid)
    _valid_run = field_validator("run_id")(_uuid)


class ProducedArtifact(StrictModel):
    filename: str
    media_type: str
    byte_size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    _valid_filename = field_validator("filename")(_filename)


class ExecutionResult(StrictModel):
    job_id: str
    status: Literal["succeeded", "failed"]
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    started_at: datetime
    completed_at: datetime
    duration_ms: float = Field(ge=0)
    timed_out: bool = False
    artifacts: list[ProducedArtifact] = Field(default_factory=list)
    error_code: ExecutionErrorCode | None = None
    output_truncated: bool = False

    _valid_job = field_validator("job_id")(_uuid)


class SourceRecord(StrictModel):
    source_record_id: str = Field(min_length=1)
    row_id: str = Field(min_length=1)
    field: str = Field(min_length=1)
    raw_value: str
    normalized_numeric_value: float | None = None
    unit: str | None = None
    metric: str | None = None
    region: str | None = None
    year: int | None = None
    population_scope: str | None = None
    residence_scope: str | None = None
    document_title: str | None = None
    document_id: str
    page_number: int = Field(gt=0)
    chunk_id: str
    exact_supporting_quote: str = Field(min_length=1)
    source_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


class ArtifactRowProposal(BaseModel):
    """Small semantic row proposed by the model; it contains no trusted provenance."""

    label: str
    series: str | None = None
    value: float
    unit: str
    evidence_id: str
    year: int | None = None
    population_scope: str | None = None
    residence_scope: str | None = None


class ArtifactDatasetProposal(BaseModel):
    """Conservative LLM-facing schema, deliberately separate from ArtifactDataset."""

    title: str
    chart_kind: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    rows: list[ArtifactRowProposal]


class ComputedValue(StrictModel):
    field: str
    operation: Literal["sum", "difference", "percentage_difference"]
    operands: list[float] = Field(min_length=2)
    result: float
    input_source_record_ids: list[str] = Field(min_length=1)


class ArtifactDataset(StrictModel):
    title: str = Field(min_length=1, max_length=200)
    task_type: Literal["artifact_chart", "artifact_table"]
    rows: list[dict[str, object]] = Field(min_length=1, max_length=10_000)
    columns: list[str] = Field(min_length=1)
    units: dict[str, str]
    source_records: list[SourceRecord] = Field(min_length=1)
    computed_values: list[ComputedValue] = Field(default_factory=list)
    requested_output: str = Field(min_length=1)


class GeneratedProgram(StrictModel):
    code: str = Field(min_length=1, max_length=100_000)


class SourceManifest(StrictModel):
    dataset_title: str
    source_records: list[SourceRecord] = Field(min_length=1)
    computed_values: list[ComputedValue] = Field(default_factory=list)


class ArtifactDescriptor(StrictModel):
    artifact_id: str
    artifact_type: Literal["chart", "table"]
    title: str
    filename: str
    media_type: str
    byte_size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    session_id: str
    run_id: str
    source_manifest_path: str
    download_url: str

    _valid_artifact = field_validator("artifact_id")(_uuid)
    _valid_session = field_validator("session_id")(_uuid)
    _valid_run = field_validator("run_id")(_uuid)
    _valid_filename = field_validator("filename")(_filename)


class ArtifactListing(StrictModel):
    artifacts: list[ArtifactDescriptor]


class ExecutorHealth(StrictModel):
    status: Literal["ok", "error"]
    detail: str
    heartbeat_age_seconds: float | None = None
