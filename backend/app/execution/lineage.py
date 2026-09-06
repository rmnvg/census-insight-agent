from __future__ import annotations

import csv
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING

from backend.app.execution.contracts import ArtifactDataset, SourceManifest
from backend.app.retrieval.models import RetrievedEvidence

if TYPE_CHECKING:
    from backend.app.agent.models import ArtifactDataRequirement


class DatasetValidationError(ValueError):
    pass


def validate_dataset(
    dataset: ArtifactDataset,
    evidence: list[RetrievedEvidence],
    requirement: ArtifactDataRequirement | None = None,
) -> None:
    by_id = {item.chunk_id: item for item in evidence}
    source_keys: set[tuple[str, str]] = set()
    source_ids: set[str] = set()
    for source in dataset.source_records:
        item = by_id.get(source.chunk_id)
        if item is None:
            raise DatasetValidationError(f"Unknown source evidence: {source.chunk_id}")
        if item.coverage_status.startswith("excluded_"):
            raise DatasetValidationError("Unverified or excluded evidence cannot enter artifacts")
        if (
            source.document_id != item.document_id
            or source.page_number != item.page_number
            or source.source_checksum != item.source_checksum
            or (source.document_title is not None and source.document_title != item.document_title)
            or source.exact_supporting_quote not in item.text
        ):
            raise DatasetValidationError(f"Source provenance mismatch: {source.source_record_id}")
        if source.raw_value not in source.exact_supporting_quote:
            raise DatasetValidationError(
                f"Raw value is not present in quote: {source.source_record_id}"
            )
        if requirement is not None:
            required = {
                "metric": source.metric,
                "region": source.region,
                "year": source.year,
                "population_scope": source.population_scope,
                "residence_scope": source.residence_scope,
                "unit": source.unit,
                "document_title": source.document_title,
            }
            missing = [name for name, value in required.items() if value in {None, ""}]
            if missing:
                raise DatasetValidationError(
                    f"Artifact source metadata missing {', '.join(missing)}: "
                    f"{source.source_record_id}"
                )
            if requirement.year is not None and source.year != requirement.year:
                raise DatasetValidationError(
                    f"Artifact source year does not match requirement: {source.source_record_id}"
                )
            for name in ("population_scope", "residence_scope"):
                expected = getattr(requirement, name)
                actual = getattr(source, name)
                if expected and actual and expected.casefold() != actual.casefold():
                    raise DatasetValidationError(
                        f"Artifact source {name} does not match requirement: "
                        f"{source.source_record_id}"
                    )
        if source.source_record_id in source_ids:
            raise DatasetValidationError("Duplicate source-record ID")
        source_ids.add(source.source_record_id)
        source_keys.add((source.row_id, source.field))
    computed_keys = {("computed", item.field) for item in dataset.computed_values}
    for computed in dataset.computed_values:
        if not set(computed.input_source_record_ids) <= source_ids:
            raise DatasetValidationError(f"Computed value has unknown inputs: {computed.field}")
    if requirement is not None and requirement.regions:
        represented = {
            source.region.casefold() for source in dataset.source_records if source.region
        }
        missing_regions = [
            region for region in requirement.regions if region.casefold() not in represented
        ]
        if missing_regions:
            raise DatasetValidationError(
                f"Artifact dataset lacks required target(s): {', '.join(missing_regions)}"
            )
        units = {source.unit.casefold() for source in dataset.source_records if source.unit}
        if requirement.comparison and len(units) != 1:
            raise DatasetValidationError("Comparison artifact source units are incompatible")
    if len(dataset.columns) != len(set(dataset.columns)):
        raise DatasetValidationError("Dataset columns must be unique")
    for index, row in enumerate(dataset.rows):
        if list(row) != dataset.columns:
            raise DatasetValidationError(f"Row {index} does not match declared columns")
        row_id = str(row.get("row_id", index))
        for field, value in row.items():
            if field == "row_id" or not isinstance(value, int | float):
                continue
            if (row_id, field) not in source_keys and ("computed", field) not in computed_keys:
                raise DatasetValidationError(
                    f"Numeric field lacks source lineage: row={row_id} field={field}"
                )


def _same_cell(expected: object, actual: str) -> bool:
    if expected is None:
        return actual.strip().casefold() in {"", "na", "n/a", "null", "missing"}
    if isinstance(expected, bool):
        return actual.casefold() == str(expected).casefold()
    if isinstance(expected, int | float):
        try:
            return Decimal(str(expected)) == Decimal(actual.strip())
        except InvalidOperation:
            return False
    return str(expected) == actual


def validate_artifact_lineage(dataset: ArtifactDataset, artifact_root: Path) -> None:
    filename = "plotted-data.csv" if dataset.task_type == "artifact_chart" else "table.csv"
    with (artifact_root / filename).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
        fieldnames = list(rows[0]) if rows else []
    if fieldnames != dataset.columns or len(rows) != len(dataset.rows):
        raise DatasetValidationError("Companion CSV shape differs from approved dataset")
    for expected, actual in zip(dataset.rows, rows, strict=True):
        if any(not _same_cell(expected[column], actual[column]) for column in dataset.columns):
            raise DatasetValidationError("Companion CSV values differ from approved dataset")
    manifest = SourceManifest.model_validate(
        json.loads((artifact_root / "source-manifest.json").read_text(encoding="utf-8"))
    )
    expected_manifest = SourceManifest(
        dataset_title=dataset.title,
        source_records=dataset.source_records,
        computed_values=dataset.computed_values,
    )
    if manifest != expected_manifest:
        raise DatasetValidationError("Source manifest differs from approved lineage")
