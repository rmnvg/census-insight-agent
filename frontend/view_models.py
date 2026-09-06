import csv
import io
import json
from collections.abc import Iterable

import pandas as pd  # type: ignore[import-untyped]

from frontend.models import Citation, Claim, TraceEventView, TraceResponse

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
VISUAL_LIMITATION = (
    "Some visual pages were excluded from automated answering because their labels and values "
    "could not be extracted with citation-safe accuracy."
)

_TRACE_EVENTS = {
    "classification",
    "query_resolution",
    "skill_selected",
    "retrieval_completed",
    "evidence_assessment",
    "artifact_data_requirement",
    "artifact_dataset_validation",
    "code_policy_validation",
    "executor_submission",
    "executor_result",
    "artifact_execution",
    "artifact_repair_decision",
    "artifact_validation",
    "citation_validation",
    "run_completed",
}
_TRACE_DETAIL_FIELDS = {
    "task_type",
    "resolved_query",
    "selected_skill",
    "candidate_count",
    "assessment_chunk_count",
    "assessment_characters",
    "selected_evidence_count",
    "sufficient",
    "valid",
    "status",
    "exit_code",
    "timed_out",
    "error_code",
    "attempt",
    "repair",
    "retry_count",
    "generated_filenames",
    "required_targets",
    "represented_targets",
}


def ordered_citations(citations: Iterable[Citation]) -> list[Citation]:
    unique: dict[str, Citation] = {}
    for citation in citations:
        unique.setdefault(citation.citation_id, citation)
    return list(unique.values())


def citations_for_claim(claim: Claim, citations: Iterable[Citation]) -> list[Citation]:
    indexed = {item.citation_id: item for item in ordered_citations(citations)}
    return [indexed[item] for item in claim.citation_ids if item in indexed]


def sanitize_trace(trace: TraceResponse) -> list[TraceEventView]:
    safe: list[TraceEventView] = []
    for event in trace.events:
        if event.event not in _TRACE_EVENTS:
            continue
        details = {
            key: value
            for key, value in event.details.items()
            if key in _TRACE_DETAIL_FIELDS and _safe_trace_value(value)
        }
        safe.append(
            TraceEventView(
                event=event.event,
                node=event.node,
                details=details,
                latency_ms=event.latency_ms,
            )
        )
    return safe


def _safe_trace_value(value: object) -> bool:
    if value is None or isinstance(value, str | int | float | bool):
        return True
    return isinstance(value, list) and all(
        isinstance(item, str | int | float | bool) for item in value
    )


def validate_png(content: bytes) -> bytes:
    if not content.startswith(PNG_SIGNATURE):
        raise ValueError("Invalid PNG signature")
    return content


def parse_csv(content: bytes) -> pd.DataFrame:
    try:
        text = content.decode("utf-8-sig")
        rows = list(csv.reader(io.StringIO(text)))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError("Invalid CSV artifact") from error
    if not rows or not rows[0] or len(rows) < 2:
        raise ValueError("CSV artifact has no data rows")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("CSV artifact has inconsistent columns")
    return pd.DataFrame(rows[1:], columns=rows[0])


def validate_source_manifest(content: bytes) -> dict[str, object]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid source manifest") from error
    if not isinstance(value, dict) or not isinstance(value.get("source_records"), list):
        raise ValueError("Invalid source manifest")
    return value
