import json
from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError

from backend.app.ingestion.models import (
    CoverageLimitation,
    DocumentCoverageReport,
    ManualTranscriptionRecord,
    PageCoverageDecision,
    PageCoverageEntry,
    PageCoverageStatus,
)

LIMITATION_MESSAGE = (
    "The available indexed text does not provide reliable evidence for this question. "
    "Some chart or map pages were excluded because their labels and values could not be "
    "extracted with citation-safe accuracy."
)
INDEXED_STATUSES: frozenset[PageCoverageStatus] = frozenset(
    {
        "indexed_provided_markdown",
        "indexed_pymupdf4llm_fallback",
        "approved_manual_transcription",
    }
)


def load_coverage_decisions(
    manifest_dir: Path, document_id: str, source_checksum: str
) -> dict[int, PageCoverageDecision]:
    """Load reviewed exclusions only when they match the authoritative PDF checksum."""
    path = manifest_dir / "page-coverage-exclusions.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    decisions: dict[int, PageCoverageDecision] = {}
    for value in raw:
        decision = PageCoverageDecision.model_validate(value)
        if decision.document_id != document_id:
            continue
        if decision.source_checksum != source_checksum:
            raise ValueError(
                f"Coverage decision checksum mismatch for {document_id} page {decision.page_number}"
            )
        if decision.page_number in decisions:
            raise ValueError(
                f"Duplicate coverage decision for {document_id} page {decision.page_number}"
            )
        decisions[decision.page_number] = decision
    return decisions


def load_manual_transcriptions(
    manifest_dir: Path, document_id: str, source_checksum: str
) -> dict[int, ManualTranscriptionRecord]:
    """Load only explicitly approved, checksum-matched human transcription records."""
    directory = manifest_dir / "manual-transcriptions"
    records: dict[int, ManualTranscriptionRecord] = {}
    if not directory.is_dir():
        return records
    for path in sorted(directory.glob("*.json")):
        try:
            record = ManualTranscriptionRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as error:
            raise ValueError(f"Invalid manual transcription {path.name}: {error}") from error
        if record.document_id != document_id:
            continue
        if record.source_checksum != source_checksum:
            raise ValueError(
                f"Manual transcription checksum mismatch for {document_id} page "
                f"{record.page_number}"
            )
        if record.page_number in records:
            raise ValueError(
                f"Duplicate manual transcription for {document_id} page {record.page_number}"
            )
        records[record.page_number] = record
    return records


def build_coverage_report(
    document_id: str, page_count: int, entries: list[PageCoverageEntry]
) -> DocumentCoverageReport:
    """Validate exactly one status per PDF page and summarize indexed limitations."""
    numbers = [entry.page_number for entry in entries]
    expected = list(range(1, page_count + 1))
    if sorted(numbers) != expected or len(numbers) != len(set(numbers)):
        raise ValueError(
            f"Coverage must contain exactly one status for every page in {document_id}"
        )
    grouped: defaultdict[PageCoverageStatus, list[int]] = defaultdict(list)
    for entry in sorted(entries, key=lambda item: item.page_number):
        grouped[entry.status].append(entry.page_number)
    all_statuses: tuple[PageCoverageStatus, ...] = (
        "indexed_provided_markdown",
        "indexed_pymupdf4llm_fallback",
        "excluded_blank",
        "excluded_decorative",
        "excluded_unverified_visual",
        "failed_page_mapping",
        "approved_manual_transcription",
    )
    lists = {status: grouped[status] for status in all_statuses}
    indexed = sum(len(lists[status]) for status in INDEXED_STATUSES)
    return DocumentCoverageReport(
        document_id=document_id,
        pdf_page_count=page_count,
        indexed_pages=indexed,
        blank_decorative_pages=len(lists["excluded_blank"]) + len(lists["excluded_decorative"]),
        excluded_visual_pages=len(lists["excluded_unverified_visual"]),
        failed_mappings=len(lists["failed_page_mapping"]),
        percentage_pages_indexed=round(indexed / page_count * 100, 2),
        page_lists_by_status=lists,
        pages=sorted(entries, key=lambda item: item.page_number),
    )


def coverage_limitation(report: DocumentCoverageReport) -> CoverageLimitation | None:
    """Return a non-false-absence limitation for excluded meaningful visual pages."""
    pages = report.page_lists_by_status["excluded_unverified_visual"]
    if not pages:
        return None
    return CoverageLimitation(
        document_id=report.document_id,
        excluded_pages=pages,
        statuses=["excluded_unverified_visual"],
        message=LIMITATION_MESSAGE,
    )


def failed_coverage_report(
    *,
    document_id: str,
    page_count: int,
    decisions: dict[int, PageCoverageDecision],
    transcriptions: dict[int, ManualTranscriptionRecord],
    reason: str,
) -> DocumentCoverageReport:
    """Assign every page a status even when document-level mapping fails."""
    entries: list[PageCoverageEntry] = []
    for page_number in range(1, page_count + 1):
        if page_number in transcriptions:
            status: PageCoverageStatus = "approved_manual_transcription"
            page_reason = None
        elif page_number in decisions:
            status = decisions[page_number].coverage_status
            page_reason = decisions[page_number].reason
        else:
            status = "failed_page_mapping"
            page_reason = reason
        entries.append(
            PageCoverageEntry(page_number=page_number, status=status, reason=page_reason)
        )
    return build_coverage_report(document_id, page_count, entries)
