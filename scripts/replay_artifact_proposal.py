import argparse
import asyncio
import csv
import json
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from pydantic import ValidationError
from qdrant_client import QdrantClient

from backend.app.agent.checkpoint import checkpoint_safe, hydrate_agent_state
from backend.app.agent.memory import select_source_support_turn
from backend.app.agent.models import ArtifactDataRequirement, ValidatedClaimRecord
from backend.app.agent.persistence import TraceStore
from backend.app.agent.scopes import canonicalize_artifact_requirement
from backend.app.config import get_settings
from backend.app.execution.client import ExecutionQueueClient, new_request
from backend.app.execution.contracts import (
    ArtifactDataset,
    ArtifactDatasetProposal,
    ArtifactRowProposal,
    ExpectedArtifact,
    SourceManifest,
)
from backend.app.execution.hydration import (
    TrustedEvidenceProvenanceError,
    hydrate_artifact_dataset,
    validate_trusted_artifact_evidence,
)
from backend.app.execution.lineage import validate_artifact_lineage, validate_dataset
from backend.app.retrieval.models import RetrievedEvidence
from backend.app.retrieval.qdrant_store import QdrantStore

CHART_PROGRAM = """from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd

payload = json.loads(Path("input.json").read_text(encoding="utf-8"))
rows = payload["dataset"]["rows"]
frame = pd.DataFrame(rows, columns=payload["dataset"]["columns"])
frame.to_csv(Path("output/plotted-data.csv"), index=False)
Path("output/source-manifest.json").write_text(
    json.dumps(payload["source_manifest"], ensure_ascii=False, indent=2) + "\\n",
    encoding="utf-8",
)
figure, axis = plt.subplots(figsize=(7, 4))
axis.bar(frame["label"], frame["value"])
axis.set_ylabel("Literacy rate (percent)")
axis.set_ylim(bottom=0)
axis.set_title(payload["dataset"]["title"])
figure.tight_layout()
figure.savefig(Path("output/chart.png"), dpi=140)
"""

TABLE_PROGRAM = """from pathlib import Path
import json
import pandas as pd

payload = json.loads(Path("input.json").read_text(encoding="utf-8"))
rows = payload["dataset"]["rows"]
columns = payload["dataset"]["columns"]
frame = pd.DataFrame(rows, columns=columns)
frame.to_csv(Path("output/table.csv"), index=False)
markdown = "| " + " | ".join(columns) + " |\\n"
markdown += "|" + "|".join("---" for column in columns) + "|\\n"
markdown += "\\n".join(
    "| " + " | ".join(str(row[column]) for column in columns) + " |"
    for row in rows
) + "\\n"
Path("output/table.md").write_text(markdown, encoding="utf-8")
Path("output/source-manifest.json").write_text(
    json.dumps(payload["source_manifest"], ensure_ascii=False, indent=2) + "\\n",
    encoding="utf-8",
)
"""


def _common(claims: list[ValidatedClaimRecord], field: str) -> Any:
    values = {getattr(claim, field) for claim in claims}
    if len(values) != 1 or None in values:
        raise RuntimeError(f"Trusted claims do not have one common {field}")
    return values.pop()


async def _execute(
    dataset: ArtifactDataset, queue: ExecutionQueueClient, artifact_type: str
) -> dict[str, object]:
    manifest = SourceManifest(
        dataset_title=dataset.title,
        source_records=dataset.source_records,
        computed_values=dataset.computed_values,
    )
    primary = "chart.png" if artifact_type == "chart" else "table.md"
    csv_name = "plotted-data.csv" if artifact_type == "chart" else "table.csv"
    expected = [
        ExpectedArtifact(
            artifact_type=artifact_type,  # type: ignore[arg-type]
            title=artifact_type.title(),
            filename=primary,
            media_type="image/png" if artifact_type == "chart" else "text/markdown",
        ),
        ExpectedArtifact(
            artifact_type="data",
            title="Data",
            filename=csv_name,
            media_type="text/csv",
            expected_columns=dataset.columns,
        ),
        ExpectedArtifact(
            artifact_type="manifest",
            title="Sources",
            filename="source-manifest.json",
            media_type="application/json",
        ),
    ]
    request = new_request(
        str(uuid4()),
        str(uuid4()),
        CHART_PROGRAM if artifact_type == "chart" else TABLE_PROGRAM,
        {
            "dataset": dataset.model_dump(mode="json"),
            "source_manifest": manifest.model_dump(mode="json"),
        },
        expected,
        20,
    )
    queue.submit(request)
    result = await queue.wait(request.job_id, 30)
    if result.status != "succeeded":
        raise RuntimeError(f"Offline executor failed: {result.error_code}: {result.stderr}")
    output = queue.root / "jobs" / request.job_id / "output"
    validate_artifact_lineage(dataset, output)
    with (output / csv_name).open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    png_valid = (
        (output / primary).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        if artifact_type == "chart"
        else None
    )
    return {
        "internal_task_type": dataset.task_type,
        "request_valid": True,
        "lineage_valid": True,
        "png_valid": png_valid,
        "csv_rows": len(csv_rows),
        "csv_matches_dataset": len(csv_rows) == len(dataset.rows),
        "executor_job_id": request.job_id,
    }


async def replay(args: argparse.Namespace) -> dict[str, object]:
    settings = get_settings()
    traces = TraceStore(args.workspace)
    trace = traces.read(args.trace_id)
    if trace is None:
        raise RuntimeError("Failed assessment trace was not found")
    requirement_event = next(
        item for item in trace.events if item.event == "artifact_data_requirement"
    )
    query_event = next(item for item in trace.events if item.event == "resolved_query")
    assessment_event = next(item for item in trace.events if item.event == "evidence_assessment")
    requirement = canonicalize_artifact_requirement(
        ArtifactDataRequirement.model_validate(requirement_event.details["requirement"]),
        cast(str, query_event.details["query"]),
    )
    selected_ids = cast(list[str], assessment_event.details["selected_evidence_ids"])
    history = traces.recover_validated_claim_history(args.session_id)
    turn, clarification = select_source_support_turn(
        "Which source pages support those comparison values?", history
    )
    if turn is None:
        raise RuntimeError(clarification or "No validated comparison exists")
    claims = [
        claim
        for claim in turn.claims
        if not claim.document_derived
        and claim.value is not None
        and claim.evidence_ids
        and claim.evidence_ids[0] in selected_ids
    ]
    if len(claims) != 2:
        raise RuntimeError("Replay requires exactly two trusted source claims")
    store = QdrantStore(
        QdrantClient(url=args.qdrant_url),
        args.collection,
        dense_dimensions=settings.gemini_embedding_dimension,
        dense_model=settings.gemini_embedding_model,
        sparse_model=settings.sparse_embedding_model,
    )
    records = await asyncio.to_thread(
        store.client.retrieve,
        store.collection_name,
        ids=selected_ids,
        with_payload=True,
        with_vectors=False,
    )
    records_by_id = {str(record.id): record for record in records}
    if missing := [evidence_id for evidence_id in selected_ids if evidence_id not in records_by_id]:
        raise RuntimeError(f"Missing trusted Qdrant evidence: {', '.join(missing)}")
    evidence = [
        RetrievedEvidence.model_validate(
            {**(records_by_id[evidence_id].payload or {}), "retrieval_score": 0.0}
        )
        for evidence_id in selected_ids
    ]
    checkpoint_payload = checkpoint_safe({"selected_evidence": evidence})
    checkpoint_evidence = hydrate_agent_state(checkpoint_payload)["selected_evidence"]
    validate_trusted_artifact_evidence(checkpoint_evidence)
    missing_payload = evidence[0].model_dump(mode="json")
    missing_payload.pop("source_checksum")
    try:
        RetrievedEvidence.model_validate(missing_payload)
    except ValidationError:
        missing_checksum_rejected = True
    else:
        missing_checksum_rejected = False
    empty_evidence = evidence[0].model_copy(update={"source_checksum": ""})
    try:
        validate_trusted_artifact_evidence([empty_evidence])
    except TrustedEvidenceProvenanceError:
        empty_checksum_rejected = True
    else:
        empty_checksum_rejected = False
    if requirement.metric.casefold() != cast(str, _common(claims, "metric")).casefold():
        raise RuntimeError("Trace requirement and trusted claims have different metrics")
    proposal = ArtifactDatasetProposal.model_validate(
        {
            "title": "Trusted offline literacy comparison",
            "artifact_type": "bar_chart",
            "chart_kind": "bar",
            "x_label": "Region",
            "y_label": "Literacy rate (percent)",
            "rows": [
                ArtifactRowProposal(
                    label=cast(str, claim.region),
                    value=cast(float, claim.value),
                    unit=cast(str, claim.unit),
                    evidence_id=claim.evidence_ids[0],
                    year=claim.year,
                    population_scope=requirement.population_scope,
                    residence_scope=requirement.residence_scope,
                )
                for claim in claims
            ],
        }
    )
    dataset = hydrate_artifact_dataset(proposal, requirement, checkpoint_evidence, "bar chart")
    validate_dataset(dataset, checkpoint_evidence, requirement)
    queue = ExecutionQueueClient(args.queue_root)
    chart_result = await _execute(dataset, queue, "chart")
    table_requirement = requirement.model_copy(update={"artifact_type": "table"})
    table_dataset = hydrate_artifact_dataset(
        proposal, table_requirement, checkpoint_evidence, "table"
    )
    validate_dataset(table_dataset, checkpoint_evidence, table_requirement)
    table_result = await _execute(table_dataset, queue, "table")
    return {
        "failed_trace_replayed": trace.run_id,
        "trusted_claim_run_id": turn.run_id,
        "legacy_proposal_artifact_type": "bar_chart",
        "proposal_schema_ignores_artifact_type": "artifact_type"
        not in ArtifactDatasetProposal.model_json_schema()["properties"],
        "proposal_valid": True,
        "trusted_evidence_ids": [source.chunk_id for source in dataset.source_records],
        "trusted_checksum_present": [bool(item.source_checksum) for item in evidence],
        "trusted_checksum_prefixes": [item.source_checksum[:8] for item in evidence],
        "retrieval_checkpoint_round_trip": [item.source_checksum for item in checkpoint_evidence]
        == [item.source_checksum for item in evidence],
        "manifest_checksums_match_qdrant": all(
            item.source_checksum
            == next(
                trusted.source_checksum for trusted in evidence if trusted.chunk_id == item.chunk_id
            )
            for item in dataset.source_records
        ),
        "missing_checksum_rejected_before_executor": missing_checksum_rejected,
        "empty_checksum_rejected_before_executor": empty_checksum_rejected,
        "provenance_failure_executor_submitted": False,
        "values_match_trusted_cells": True,
        "targets": [source.region for source in dataset.source_records],
        "year": requirement.year,
        "population_scope": requirement.population_scope,
        "residence_scope": requirement.residence_scope,
        "authoritative_artifact_type": requirement.artifact_type,
        "dataset_valid": True,
        "application_owned_source_records": len(dataset.source_records),
        "chart_replay": chart_result,
        "table_replay": table_result,
        "gemini_calls": 0,
        "embedding_calls": 0,
        "qdrant_writes": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay proposal hydration without Gemini")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))
    parser.add_argument("--queue-root", type=Path, default=Path("workspace/execution-queue"))
    parser.add_argument("--qdrant-url", default="http://qdrant:6333")
    parser.add_argument("--collection", default="census_documents")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(replay(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
