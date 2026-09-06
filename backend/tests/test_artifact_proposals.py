import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.agent.checkpoint import checkpoint_safe, hydrate_agent_state
from backend.app.agent.errors import AgentOperationalError
from backend.app.agent.evidence import pack_targeted_assessment_evidence
from backend.app.agent.graph import AgentGraph
from backend.app.agent.models import (
    ArtifactDataRequirement,
    ResolvedQuery,
    TaskClassification,
)
from backend.app.agent.provider import (
    GeminiAgentModel,
    ProviderCallTimeout,
    ProviderOperationalError,
    map_provider_error,
)
from backend.app.agent.scopes import canonicalize_artifact_requirement
from backend.app.agent.service import AgentService
from backend.app.agent.skills import SkillRegistry
from backend.app.execution.contracts import (
    ArtifactDataset,
    ArtifactDatasetProposal,
    ArtifactRowProposal,
    GeneratedProgram,
    SourceRecord,
)
from backend.app.execution.hydration import (
    ProposalHydrationError,
    TrustedEvidenceProvenanceError,
    hydrate_artifact_dataset,
    resolved_chart_kind,
)
from backend.app.execution.lineage import validate_dataset
from backend.app.execution.schema_complexity import assert_proposal_schema_safe, schema_complexity
from backend.app.main import app
from backend.app.retrieval.models import RetrievedEvidence
from backend.tests.test_agent_service import FakeModel, FakeTools, run, settings


def requirement() -> ArtifactDataRequirement:
    return ArtifactDataRequirement(
        artifact_type="chart",
        metric="literacy rate",
        year=2011,
        regions=["North", "South"],
        population_scope="persons",
        residence_scope="total",
        comparison=True,
    )


def table_evidence(region: str, value: str, identifier: str) -> RetrievedEvidence:
    text = (
        "Literacy Rate by residence: 2011 (Persons)\n\n"
        "| Region | Literacy Rate | | | | | |\n"
        "|---|---|---|---|---|---|---|\n"
        "| | 2001 | | | 2011 | | |\n"
        "| | Total | Rural | Urban | Total | Rural | Urban |\n"
        "| 1 | 2 | 3 | 4 | 5 | 6 | 7 |\n"
        f"| {region} | 60.50 | 55.25 | 70.75 | {value} | 61.25 | 81.75 |"
    )
    return RetrievedEvidence(
        chunk_id=identifier,
        text=text,
        document_title=f"Synthetic {region} report",
        document_id=f"document-{region.casefold()}",
        region=region,
        page_number=12,
        citation_snippet=text,
        section_path=["Literacy Rate by residence", "2011", "Persons"],
        extraction_method="provided_markdown",
        coverage_status="indexed_provided_markdown",
        source_checksum=("a" if region == "North" else "b") * 64,
        retrieval_score=1,
    )


def proposal(**updates: object) -> ArtifactDatasetProposal:
    rows = [
        ArtifactRowProposal(
            label="North",
            value=71.5,
            unit="percent",
            evidence_id="north-evidence",
            year=2011,
            population_scope="persons",
            residence_scope="total",
        ),
        ArtifactRowProposal(
            label="South",
            value=72.9,
            unit="percent",
            evidence_id="south-evidence",
            year=2011,
            population_scope="persons",
            residence_scope="total",
        ),
    ]
    values: dict[str, object] = {
        "title": "Synthetic comparison",
        "chart_kind": "bar",
        "x_label": "Region",
        "y_label": "Literacy rate (percent)",
        "rows": rows,
    }
    values.update(updates)
    return ArtifactDatasetProposal.model_validate(values)


def evidence() -> list[RetrievedEvidence]:
    return [
        table_evidence("North", "71.50", "north-evidence"),
        table_evidence("South", "72.9", "south-evidence"),
    ]


def test_llm_proposal_schema_is_small_and_excludes_internal_contracts() -> None:
    compact = assert_proposal_schema_safe(ArtifactDatasetProposal)
    full = schema_complexity(ArtifactDataset)
    serialized = str(ArtifactDatasetProposal.model_json_schema()).casefold()
    assert compact["serialized_characters"] < full["serialized_characters"]
    assert compact["definition_count"] == 1
    assert compact["formats"] == []
    assert compact["patterns"] == []
    assert "artifact_type" not in serialized


class RecordingStructuredModel:
    def __init__(self) -> None:
        self.schemas: list[type[Any]] = []
        self.messages: list[object] = []

    def with_structured_output(self, schema: type[Any]) -> "RecordingStructuredModel":
        self.schemas.append(schema)
        return self

    async def ainvoke(self, messages: object) -> dict[str, object]:
        self.messages.append(messages)
        if self.schemas[-1] is GeneratedProgram:
            return {"code": "print('artifact')"}
        return proposal().model_dump(mode="json")


def test_provider_never_passes_full_artifact_dataset_as_output_schema() -> None:
    chat = RecordingStructuredModel()
    model = GeminiAgentModel(chat, max_retries=0)  # type: ignore[arg-type]
    result = asyncio.run(model.propose_artifact_dataset("chart", "artifact_chart", evidence()))
    assert result == proposal()
    assert chat.schemas == [ArtifactDatasetProposal]
    assert ArtifactDataset not in chat.schemas


def test_artifact_code_prompt_describes_the_real_executor_input_envelope() -> None:
    chat = RecordingStructuredModel()
    model = GeminiAgentModel(chat, max_retries=0)  # type: ignore[arg-type]
    value = hydrate_artifact_dataset(proposal(), requirement(), evidence(), "chart request")

    result = asyncio.run(model.generate_artifact_code(value, "Create a bar chart."))

    assert result.code == "print('artifact')"
    messages = cast(list[Any], chat.messages[-1])
    prompt = "\n".join(str(message.content) for message in messages)
    assert 'payload["dataset"]' in prompt
    assert 'payload["source_manifest"]' in prompt
    assert '"dataset": {' in prompt
    assert '"source_manifest": {' in prompt
    assert "output directory already" in prompt
    assert "do not create directories" in prompt


def test_valid_proposal_hydrates_with_application_owned_provenance() -> None:
    trusted = evidence()
    dataset = hydrate_artifact_dataset(proposal(), requirement(), trusted, "chart request")
    validate_dataset(dataset, trusted, requirement())
    assert len(dataset.rows) == 2
    assert dataset.task_type == "artifact_chart"
    assert [row["value"] for row in dataset.rows] == [71.5, 72.9]
    for source, item in zip(dataset.source_records, trusted, strict=True):
        assert source.document_id == item.document_id
        assert source.document_title == item.document_title
        assert source.page_number == item.page_number
        assert source.chunk_id == item.chunk_id
        assert source.source_checksum == item.source_checksum
        assert source.exact_supporting_quote in item.text


def test_plural_literacy_metric_is_canonicalized_before_hydration() -> None:
    plural = requirement().model_copy(update={"metric": "literacy rates"})
    canonical = canonicalize_artifact_requirement(plural, "Compare literacy rates")
    dataset = hydrate_artifact_dataset(proposal(), canonical, evidence(), "chart request")
    validate_dataset(dataset, evidence(), canonical)
    assert canonical.metric == "literacy rate"


@pytest.mark.parametrize("legacy_value", ["table", "bar_chart", "bar chart"])
def test_legacy_model_artifact_type_cannot_override_chart_requirement(
    legacy_value: str,
) -> None:
    raw = proposal().model_dump(mode="json")
    raw["artifact_type"] = legacy_value
    legacy = ArtifactDatasetProposal.model_validate(raw)
    dataset = hydrate_artifact_dataset(legacy, requirement(), evidence(), "bar chart")
    assert dataset.task_type == "artifact_chart"


def test_table_identity_is_application_controlled_and_chart_kind_is_presentation_only() -> None:
    table_requirement = requirement().model_copy(update={"artifact_type": "table"})
    value = proposal(chart_kind="bar_chart")
    dataset = hydrate_artifact_dataset(value, table_requirement, evidence(), "table")
    assert dataset.task_type == "artifact_table"
    assert resolved_chart_kind(value, table_requirement) is None
    assert resolved_chart_kind(value, requirement()) == "bar"
    assert resolved_chart_kind(proposal(chart_kind="unknown visual"), requirement()) is None


@pytest.mark.parametrize(
    ("row_update", "error_code"),
    [
        ({"evidence_id": "unknown"}, "UNKNOWN_EVIDENCE_ID"),
        ({"value": 99.0}, "UNSUPPORTED_OR_WRONG_TABLE_CELL"),
        ({"value": 61.25}, "UNSUPPORTED_OR_WRONG_TABLE_CELL"),
        ({"residence_scope": "rural"}, "WRONG_RESIDENCE_SCOPE"),
        ({"population_scope": "female"}, "WRONG_POPULATION_SCOPE"),
        ({"year": 2001}, "WRONG_YEAR"),
        ({"label": "Elsewhere"}, "WRONG_REGION_ROW"),
    ],
)
def test_proposal_rejects_unknown_or_wrong_cell_dimensions(
    row_update: dict[str, object], error_code: str
) -> None:
    value = proposal()
    changed = value.rows[0].model_copy(update=row_update)
    value = value.model_copy(update={"rows": [changed, value.rows[1]]})
    with pytest.raises(ProposalHydrationError) as caught:
        hydrate_artifact_dataset(value, requirement(), evidence(), "chart request")
    assert caught.value.code == error_code


def test_hydration_rejects_invalid_trusted_source_checksum() -> None:
    trusted = evidence()
    trusted[0] = trusted[0].model_copy(update={"source_checksum": ""})
    with pytest.raises(TrustedEvidenceProvenanceError, match="INVALID_SOURCE_CHECKSUM"):
        hydrate_artifact_dataset(proposal(), requirement(), trusted, "chart request")


def test_checksum_survives_checkpoint_packing_and_artifact_selection() -> None:
    trusted = evidence()
    raw = checkpoint_safe({"retrieved_evidence": trusted, "selected_evidence": trusted})
    restored = hydrate_agent_state(raw)
    restored_evidence = restored["selected_evidence"]
    packed = pack_targeted_assessment_evidence(
        restored_evidence,
        required_targets=["North", "South"],
        max_characters=20_000,
        max_chunks=4,
    ).included
    dataset = hydrate_artifact_dataset(proposal(), requirement(), packed, "chart request")
    assert [item.source_checksum for item in restored_evidence] == ["a" * 64, "b" * 64]
    assert [item.source_checksum for item in packed] == ["a" * 64, "b" * 64]
    assert [item.source_checksum for item in dataset.source_records] == ["a" * 64, "b" * 64]


def test_proposal_has_no_checksum_field_and_cannot_override_trusted_checksum() -> None:
    assert "source_checksum" not in str(ArtifactDatasetProposal.model_json_schema())
    raw = proposal().model_dump(mode="json")
    raw["source_checksum"] = "f" * 64
    raw["rows"][0]["source_checksum"] = "f" * 64
    dataset = hydrate_artifact_dataset(
        ArtifactDatasetProposal.model_validate(raw), requirement(), evidence(), "chart request"
    )
    assert [item.source_checksum for item in dataset.source_records] == ["a" * 64, "b" * 64]


class CountingProposalModel:
    def __init__(self) -> None:
        self.calls = 0

    async def propose_artifact_dataset(
        self,
        query: str,
        task_type: str,
        values: list[RetrievedEvidence],
        *,
        rank_all: bool = False,
    ) -> ArtifactDatasetProposal:
        del query, task_type, values, rank_all
        self.calls += 1
        return proposal()


def test_invalid_provenance_stops_before_proposal_or_executor_submission() -> None:
    model = CountingProposalModel()
    graph = AgentGraph(model, object())  # type: ignore[arg-type]
    trusted = evidence()
    trusted[0] = trusted[0].model_copy(update={"source_checksum": ""})
    with pytest.raises(AgentOperationalError) as caught:
        run(
            graph.prepare_artifact(
                {
                    "artifact_requirement": requirement(),
                    "selected_evidence": trusted,
                    "resolved_query": "chart request",
                    "task_type": "artifact_chart",
                }
            )
        )
    assert caught.value.code == "INTERNAL_PROVENANCE_INVALID"
    assert caught.value.diagnostics == {
        "failure_stage": "trusted_evidence_validation",
        "reason_code": "INVALID_SOURCE_CHECKSUM",
        "affected_evidence_id": "north-evidence",
        "checksum_present": False,
        "executor_submitted": False,
    }
    assert model.calls == 0


def test_internal_source_record_validation_maps_to_provenance_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValidationError) as validation:
        SourceRecord.model_validate({"source_checksum": ""})

    def fail_hydration(*_: object, **__: object) -> None:
        raise validation.value

    monkeypatch.setattr("backend.app.agent.graph.hydrate_artifact_dataset", fail_hydration)
    graph = AgentGraph(CountingProposalModel(), object())  # type: ignore[arg-type]
    with pytest.raises(AgentOperationalError) as caught:
        run(
            graph.prepare_artifact(
                {
                    "artifact_requirement": requirement(),
                    "selected_evidence": evidence(),
                    "resolved_query": "chart request",
                    "task_type": "artifact_chart",
                }
            )
        )
    assert caught.value.code == "INTERNAL_PROVENANCE_INVALID"
    assert caught.value.diagnostics["failure_stage"] == "internal_contract_validation"
    assert caught.value.diagnostics["executor_submitted"] is False


def test_decimal_formatting_is_compatible_and_duplicates_are_rejected() -> None:
    dataset = hydrate_artifact_dataset(proposal(), requirement(), evidence(), "chart request")
    assert dataset.source_records[0].raw_value == "71.50"
    duplicated = proposal(rows=[proposal().rows[0], proposal().rows[0], proposal().rows[1]])
    with pytest.raises(ProposalHydrationError, match="DUPLICATE_PROPOSAL_ROW"):
        hydrate_artifact_dataset(duplicated, requirement(), evidence(), "chart request")


def test_total_persons_scope_matches_trusted_persons_table_header() -> None:
    scoped = requirement().model_copy(update={"population_scope": "Total Persons"})
    value = proposal()
    value = value.model_copy(
        update={
            "rows": [
                row.model_copy(update={"population_scope": "Total Persons"}) for row in value.rows
            ]
        }
    )
    dataset = hydrate_artifact_dataset(value, scoped, evidence(), "chart request")
    validate_dataset(dataset, evidence(), scoped)
    assert {source.population_scope for source in dataset.source_records} == {"Total Persons"}


def test_ambiguous_matching_table_cells_are_rejected() -> None:
    trusted = evidence()
    changed_text = (
        trusted[0]
        .text.replace(
            "| | Total | Rural | Urban | Total | Rural | Urban |",
            "| | Total | Rural | Urban | Total | Total | Urban |",
        )
        .replace(
            "| North | 60.50 | 55.25 | 70.75 | 71.50 | 61.25 |",
            "| North | 60.50 | 55.25 | 70.75 | 71.50 | 71.50 |",
        )
    )
    trusted[0] = trusted[0].model_copy(
        update={"text": changed_text, "citation_snippet": changed_text}
    )
    with pytest.raises(ProposalHydrationError, match="AMBIGUOUS_TABLE_CELL"):
        hydrate_artifact_dataset(proposal(), requirement(), trusted, "chart request")


class ArtifactProviderFailureModel(FakeModel):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def classify(self, query: str, context: list[Any]) -> TaskClassification:
        del query, context
        return TaskClassification(
            task_type="artifact_chart",
            regions=["Karnataka"],
            artifact_requirement=ArtifactDataRequirement(
                artifact_type="chart",
                metric="literacy rate",
                year=2011,
                regions=["Karnataka"],
                population_scope="persons",
                residence_scope="total",
            ),
            reason="Artifact",
        )

    async def resolve(self, query: str, context: list[Any]) -> ResolvedQuery:
        del context
        return ResolvedQuery(query=query)

    async def propose_artifact_dataset(
        self,
        query: str,
        task_type: str,
        values: list[RetrievedEvidence],
        *,
        rank_all: bool = False,
    ) -> ArtifactDatasetProposal:
        del query, task_type, values, rank_all
        raise self.error


class ArtifactHydrationFailureModel(ArtifactProviderFailureModel):
    def __init__(self) -> None:
        FakeModel.__init__(self)

    async def propose_artifact_dataset(
        self,
        query: str,
        task_type: str,
        values: list[RetrievedEvidence],
        *,
        rank_all: bool = False,
    ) -> ArtifactDatasetProposal:
        del query, task_type, values, rank_all
        value = proposal()
        return value.model_copy(
            update={"rows": [value.rows[0].model_copy(update={"evidence_id": "unknown-evidence"})]}
        )


class InvalidProvenanceTools(FakeTools):
    async def search_documents(self, value: Any) -> list[RetrievedEvidence]:
        values = await super().search_documents(value)
        return [item.model_copy(update={"source_checksum": ""}) for item in values]


class CountingArtifactModel(ArtifactProviderFailureModel):
    def __init__(self) -> None:
        FakeModel.__init__(self)
        self.proposal_calls = 0

    async def propose_artifact_dataset(
        self,
        query: str,
        task_type: str,
        values: list[RetrievedEvidence],
        *,
        rank_all: bool = False,
    ) -> ArtifactDatasetProposal:
        del query, task_type, values, rank_all
        self.proposal_calls += 1
        return proposal()


@pytest.mark.parametrize(
    ("error", "code", "status"),
    [
        (
            ProviderOperationalError(
                "MODEL_SCHEMA_REJECTED",
                retryable=False,
                provider_exception="GoogleInvalidRequestError",
            ),
            "MODEL_SCHEMA_REJECTED",
            502,
        ),
        (
            ProviderCallTimeout(elapsed_seconds=0.1, timeout_seconds=0.1, retry_count=0),
            "MODEL_TIMEOUT",
            504,
        ),
        (
            ProviderOperationalError(
                "MODEL_RATE_LIMITED", retryable=True, provider_exception="ResourceExhausted"
            ),
            "MODEL_RATE_LIMITED",
            503,
        ),
        (
            ProviderOperationalError(
                "MODEL_AUTHENTICATION_FAILED",
                retryable=False,
                provider_exception="Unauthenticated",
            ),
            "MODEL_AUTHENTICATION_FAILED",
            502,
        ),
    ],
)
def test_prepare_provider_failures_return_sanitized_typed_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    code: str,
    status: int,
) -> None:
    service = AgentService(
        settings(tmp_path),
        ArtifactProviderFailureModel(error),
        FakeTools(SkillRegistry(tmp_path / "skills")),  # type: ignore[arg-type]
    )
    session = run(service.create_session())
    monkeypatch.setattr("backend.app.api.get_agent_service", lambda: service)
    response = TestClient(app).post(
        "/chat", json={"session_id": session.session_id, "message": "Create a literacy chart"}
    )
    body = response.json()
    assert response.status_code == status
    assert body["error_code"] == code
    assert "INVALID_ARGUMENT" not in body["message"]
    trace = service.get_trace(body["trace_id"])
    assert trace is not None and trace.run_status == "failed"
    assert trace.events[-1].node == "prepare_artifact"
    assert not list((tmp_path / "workspace/execution-queue/inbox").glob("*.json"))


def test_deterministic_hydration_failure_has_no_fake_timeout_or_executor_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = AgentService(
        settings(tmp_path),
        ArtifactHydrationFailureModel(),
        FakeTools(SkillRegistry(tmp_path / "skills")),  # type: ignore[arg-type]
    )
    session = run(service.create_session())
    monkeypatch.setattr("backend.app.api.get_agent_service", lambda: service)
    response = TestClient(app).post(
        "/chat",
        json={
            "session_id": session.session_id,
            "message": "Create a chart of 2011 total persons literacy",
        },
    )
    body = response.json()
    assert response.status_code == 502
    assert body["error_code"] == "MODEL_OUTPUT_INVALID"
    trace = service.get_trace(body["trace_id"])
    assert trace is not None
    details = trace.events[-1].details
    assert details["failure_stage"] == "deterministic_hydration"
    assert details["executor_submitted"] is False
    assert "elapsed_seconds" not in details
    assert "configured_timeout_seconds" not in details
    assert not list((tmp_path / "workspace/execution-queue/inbox").glob("*.json"))


def test_internal_provenance_failure_is_sanitized_http_500_before_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = CountingArtifactModel()
    service = AgentService(
        settings(tmp_path),
        model,
        InvalidProvenanceTools(SkillRegistry(tmp_path / "skills")),  # type: ignore[arg-type]
    )
    session = run(service.create_session())
    monkeypatch.setattr("backend.app.api.get_agent_service", lambda: service)
    response = TestClient(app).post(
        "/chat",
        json={
            "session_id": session.session_id,
            "message": "Create a chart of 2011 total persons literacy in Karnataka",
        },
    )
    body = response.json()
    assert response.status_code == 500
    assert body == {
        "error_code": "INTERNAL_PROVENANCE_INVALID",
        "message": "Artifact preparation failed because trusted source provenance was invalid.",
        "session_id": session.session_id,
        "trace_id": body["trace_id"],
        "retryable": False,
    }
    assert "validation error" not in response.text.casefold()
    assert "pattern" not in response.text.casefold()
    assert model.proposal_calls == 0
    trace = service.get_trace(body["trace_id"])
    assert trace is not None and trace.run_status == "failed"
    details = trace.events[-1].details
    assert details["failure_stage"] == "retrieval_provenance_validation"
    assert details["checksum_present"] is False
    assert details["executor_submitted"] is False
    assert "source_checksum" not in details
    assert not list((tmp_path / "workspace/execution-queue/inbox").glob("*.json"))


def test_external_chat_body_validation_remains_http_422() -> None:
    response = TestClient(app).post("/chat", json={"session_id": "missing-message"})
    assert response.status_code == 422


class GoogleInvalidRequestError(Exception):
    pass


def test_google_invalid_schema_error_is_classified_without_leaking_text() -> None:
    error = GoogleInvalidRequestError(
        "400 INVALID_ARGUMENT: schema has too many states; secret-provider-detail"
    )
    mapped = map_provider_error(error)
    assert mapped.code == "MODEL_SCHEMA_REJECTED"
    assert "secret-provider-detail" not in str(mapped)
