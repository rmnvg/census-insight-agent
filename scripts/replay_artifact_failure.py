import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.app.agent.evidence import pack_targeted_assessment_evidence
from backend.app.agent.models import ArtifactDataRequirement
from backend.app.execution.client import new_request
from backend.app.execution.contracts import (
    ArtifactDataset,
    ExpectedArtifact,
    SourceManifest,
    SourceRecord,
)
from backend.app.execution.lineage import validate_artifact_lineage, validate_dataset
from backend.app.retrieval.models import RetrievedEvidence
from executor.runner import execute_request

PROGRAM = """from pathlib import Path
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
axis.bar(frame["region"], frame["literacy_rate"])
axis.set_ylabel("Literacy rate (percent)")
axis.set_ylim(bottom=0)
axis.set_title("Synthetic offline comparison")
figure.tight_layout()
figure.savefig(Path("output/chart.png"), dpi=140)
"""


def evidence(region: str, page: int, identifier: str, kind: str, score: float) -> RetrievedEvidence:
    if kind == "direct":
        raw = "71.00" if region == "Karnataka" else "72.00"
        text = (
            "| Region | Year | Residence | Population | Metric | Value | Unit |\n"
            "|---|---:|---|---|---|---:|---|\n"
            f"| {region} | 2011 | Total | Persons | Literacy rate | {raw} | percent |"
        )
    else:
        text = f"{kind} material for {region} with non-statewide context 1. " + "x" * 1300
    return RetrievedEvidence(
        chunk_id=identifier,
        text=text,
        document_title=f"Synthetic {region} report",
        document_id=f"fixture-{region.casefold().replace(' ', '-')}",
        region=region,
        page_number=page,
        citation_snippet=text[:80],
        section_path=["Literacy"],
        extraction_method="provided_markdown",
        coverage_status="indexed_provided_markdown",
        source_checksum=("a" if region == "Karnataka" else "b") * 64,
        retrieval_score=score,
    )


def load_candidates() -> list[RetrievedEvidence]:
    fixture = Path(__file__).parents[1] / "backend/tests/fixtures/prompt5_failed_candidates.json"
    return [
        evidence(row["region"], row["page"], row["id"], row["kind"], row["score"])
        for row in json.loads(fixture.read_text(encoding="utf-8"))
    ]


def build_dataset(selected: list[RetrievedEvidence]) -> ArtifactDataset:
    direct = [item for item in selected if "| Region |" in item.text]
    records: list[SourceRecord] = []
    rows: list[dict[str, object]] = []
    for index, item in enumerate(direct, 1):
        raw = "71.00" if item.region == "Karnataka" else "72.00"
        records.append(
            SourceRecord(
                source_record_id=f"source-{index}",
                row_id=item.region.casefold(),
                field="literacy_rate",
                raw_value=raw,
                normalized_numeric_value=float(raw),
                unit="percent",
                metric="literacy rate",
                region=item.region,
                year=2011,
                population_scope="persons",
                residence_scope="total",
                document_title=item.document_title,
                document_id=item.document_id,
                page_number=item.page_number,
                chunk_id=item.chunk_id,
                exact_supporting_quote=item.text,
                source_checksum=item.source_checksum,
            )
        )
        rows.append(
            {
                "row_id": item.region.casefold(),
                "region": item.region,
                "literacy_rate": float(raw),
            }
        )
    return ArtifactDataset(
        title="Synthetic offline comparison",
        task_type="artifact_chart",
        rows=rows,
        columns=["row_id", "region", "literacy_rate"],
        units={"literacy_rate": "percent"},
        source_records=records,
        requested_output="bar chart",
    )


def main() -> int:
    candidates = load_candidates()
    packed = pack_targeted_assessment_evidence(
        candidates,
        required_targets=["Karnataka", "Odisha"],
        max_characters=12_000,
        max_chunks=12,
    )
    dataset = build_dataset(packed.included)
    requirement = ArtifactDataRequirement(
        artifact_type="chart",
        metric="literacy rate",
        year=2011,
        regions=["Karnataka", "Odisha"],
        population_scope="persons",
        residence_scope="total",
        comparison=True,
    )
    direct = [item for item in packed.included if "| Region |" in item.text]
    validate_dataset(dataset, direct, requirement)
    manifest = SourceManifest(
        dataset_title=dataset.title,
        source_records=dataset.source_records,
        computed_values=dataset.computed_values,
    )
    request = new_request(
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
        PROGRAM,
        {
            "dataset": dataset.model_dump(mode="json"),
            "source_manifest": manifest.model_dump(mode="json"),
        },
        [
            ExpectedArtifact(
                artifact_type="chart",
                title="Chart",
                filename="chart.png",
                media_type="image/png",
            ),
            ExpectedArtifact(
                artifact_type="data",
                title="Plotted data",
                filename="plotted-data.csv",
                media_type="text/csv",
                expected_columns=dataset.columns,
            ),
            ExpectedArtifact(
                artifact_type="manifest",
                title="Sources",
                filename="source-manifest.json",
                media_type="application/json",
            ),
        ],
        20,
    )
    with TemporaryDirectory(prefix="prompt5-replay-") as temporary:
        job = Path(temporary) / request.job_id
        result = execute_request(request, job)
        if result.status != "succeeded":
            raise RuntimeError(f"Offline executor failed: {result.error_code}: {result.stderr}")
        validate_artifact_lineage(dataset, job / "output")
        if not (job / "output/chart.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("Chart does not have a valid PNG signature")
        with (job / "output/plotted-data.csv").open(encoding="utf-8", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        output = {
            "retrieval_candidates": len(candidates),
            "reserved_by_target": packed.reserved_by_target,
            "assessment_includes_page_82": any(
                item.region == "Odisha" and item.page_number == 82 for item in packed.included
            ),
            "direct_targets": sorted(item.region for item in direct),
            "dataset_valid": True,
            "executor_request_valid": True,
            "png_valid": True,
            "csv_rows": len(csv_rows),
            "csv_matches_dataset": len(csv_rows) == 2,
            "manifest_source_records": len(dataset.source_records),
            "gemini_calls": 0,
            "qdrant_writes": 0,
        }
        print(json.dumps(output, indent=2))
        return (
            0
            if all(
                (
                    output["assessment_includes_page_82"],
                    output["dataset_valid"],
                    output["png_valid"],
                    output["csv_matches_dataset"],
                    output["manifest_source_records"] == 2,
                )
            )
            else 1
        )


if __name__ == "__main__":
    raise SystemExit(main())
