import re
from decimal import Decimal, InvalidOperation

from backend.app.agent.models import ArtifactDataRequirement
from backend.app.execution.contracts import (
    ArtifactDataset,
    ArtifactDatasetProposal,
    ArtifactRowProposal,
    SourceRecord,
)
from backend.app.retrieval.models import RetrievedEvidence

_HTML = re.compile(r"<[^>]+>")
_MARKDOWN = re.compile(r"[*_`]+")
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?!\w)")
_WORD = re.compile(r"[a-z0-9]+")


class ProposalHydrationError(ValueError):
    """A model proposal could not be bound unambiguously to trusted evidence."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class TrustedEvidenceProvenanceError(ValueError):
    """Current-run evidence is incomplete or unsafe for artifact construction."""

    def __init__(self, evidence_id: str, code: str, *, checksum_present: bool) -> None:
        super().__init__(code)
        self.evidence_id = evidence_id
        self.code = code
        self.checksum_present = checksum_present


_SAFE_PROVENANCE_PAIRS = {
    ("provided_markdown", "indexed_provided_markdown"),
    ("pymupdf4llm_fallback", "indexed_pymupdf4llm_fallback"),
    ("approved_manual_transcription", "approved_manual_transcription"),
}


def validate_trusted_artifact_evidence(evidence: list[RetrievedEvidence]) -> None:
    """Fail before a provider call when artifact evidence lacks trusted provenance."""
    for item in evidence:
        checksum_present = bool(item.source_checksum)
        checks = (
            (bool(item.chunk_id.strip()), "MISSING_CHUNK_ID"),
            (bool(item.text.strip()), "MISSING_TRUSTED_TEXT"),
            (bool(item.document_title.strip()), "MISSING_DOCUMENT_TITLE"),
            (bool(item.document_id.strip()), "MISSING_DOCUMENT_ID"),
            (item.page_number > 0, "INVALID_PHYSICAL_PAGE"),
            (
                bool(item.citation_snippet) and item.citation_snippet in item.text,
                "INVALID_CITATION_PROVENANCE",
            ),
            (
                bool(re.fullmatch(r"[0-9a-f]{64}", item.source_checksum)),
                "INVALID_SOURCE_CHECKSUM",
            ),
            (
                (str(item.extraction_method), str(item.coverage_status)) in _SAFE_PROVENANCE_PAIRS,
                "UNSAFE_COVERAGE_STATUS",
            ),
        )
        for valid, code in checks:
            if not valid:
                raise TrustedEvidenceProvenanceError(
                    item.chunk_id or "unknown", code, checksum_present=checksum_present
                )


def _plain(value: str) -> str:
    return " ".join(_MARKDOWN.sub("", _HTML.sub(" ", value)).split())


def _fold(value: str) -> str:
    return _plain(value).casefold()


def _decimal(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace(",", "").strip())
    except InvalidOperation:
        return None


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(not cell or set(cell) <= {"-", ":"} for cell in cells)


def _is_column_numbers(cells: list[str]) -> bool:
    plain = [_plain(cell) for cell in cells]
    populated = [cell for cell in plain if cell]
    return len(populated) >= 2 and populated == [
        str(index) for index in range(1, len(populated) + 1)
    ]


def _forward_fill(cells: list[str], width: int) -> list[str]:
    values = [_plain(cell) for cell in cells] + [""] * max(0, width - len(cells))
    active = ""
    for index, value in enumerate(values[:width]):
        if value:
            active = value
        else:
            values[index] = active
    return values[:width]


def _words(value: str) -> set[str]:
    return set(_WORD.findall(_fold(value)))


def _same_scope(expected: str | None, proposed: str | None, name: str) -> str:
    if expected is None:
        raise ProposalHydrationError(f"MISSING_AUTHORITATIVE_{name.upper()}")
    expected_value = _fold(expected)
    proposed_value = _fold(proposed) if proposed is not None else None
    if name == "population_scope":
        categories = {"persons", "person", "male", "female"}
        expected_category = _words(expected) & categories
        proposed_category = _words(proposed or "") & categories
        mismatch = bool(proposed is not None and expected_category != proposed_category)
    else:
        mismatch = bool(proposed is not None and proposed_value != expected_value)
    if mismatch:
        raise ProposalHydrationError(f"WRONG_{name.upper()}")
    return expected


def _unit_supported(unit: str, header: str, context: str) -> bool:
    normalized = _fold(unit).replace("percentage", "percent").replace("per cent", "percent")
    trusted = (
        _fold(f"{header} {context}").replace("percentage", "percent").replace("per cent", "percent")
    )
    if normalized in trusted:
        return True
    return normalized == "percent" and "rate" in _words(header)


def _match_row(
    proposal: ArtifactRowProposal,
    evidence: RetrievedEvidence,
    requirement: ArtifactDataRequirement,
) -> tuple[str, str]:
    if _fold(proposal.label) != _fold(evidence.region):
        raise ProposalHydrationError("WRONG_REGION_ROW")
    if requirement.regions and not any(
        _fold(proposal.label) == _fold(region) for region in requirement.regions
    ):
        raise ProposalHydrationError("WRONG_REQUIRED_REGION")
    year = _same_scope(
        str(requirement.year) if requirement.year is not None else None,
        str(proposal.year) if proposal.year is not None else None,
        "year",
    )
    population = _same_scope(
        requirement.population_scope, proposal.population_scope, "population_scope"
    )
    residence = _same_scope(
        requirement.residence_scope, proposal.residence_scope, "residence_scope"
    )
    del year

    lines = evidence.text.splitlines(keepends=True)
    matches: list[tuple[str, str]] = []
    offset = 0
    for line_index, line in enumerate(lines):
        end_offset = offset + len(line)
        offset = end_offset
        if line.count("|") < 2:
            continue
        cells = _table_cells(line)
        plain_cells = [_plain(cell) for cell in cells]
        if not any(_fold(cell) == _fold(proposal.label) for cell in plain_cells):
            continue
        value_columns = [
            index
            for index, cell in enumerate(plain_cells)
            for token in _NUMBER.findall(cell)
            if _decimal(token) == Decimal(str(proposal.value))
        ]
        if not value_columns:
            continue
        table_start = line_index
        while table_start > 0 and lines[table_start - 1].count("|") >= 2:
            table_start -= 1
        header_rows = []
        for header_line in lines[table_start:line_index]:
            header_cells = _table_cells(header_line)
            if _is_separator(header_cells) or _is_column_numbers(header_cells):
                continue
            header_rows.append(header_cells)
        for column in value_columns:
            width = len(cells)
            header = " ".join(
                row[column]
                for row in (_forward_fill(header_row, width) for header_row in header_rows)
                if column < len(row) and row[column]
            )
            header_words = _words(header)
            metric_words = _words(requirement.metric)
            context = " ".join(evidence.section_path) + " " + evidence.text[:end_offset]
            if not metric_words <= _words(f"{header} {context}"):
                continue
            if requirement.year is not None and str(requirement.year) not in header_words:
                continue
            if _fold(residence) not in _fold(header):
                continue
            population_categories = {"persons", "person", "male", "female"}
            category_headers = header_words & population_categories
            required_categories = _words(population) & population_categories
            if category_headers and not required_categories <= category_headers:
                continue
            if required_categories and not required_categories <= _words(context):
                continue
            if not required_categories and _fold(population) not in _fold(context):
                continue
            if proposal.series and _fold(proposal.series) not in _fold(header):
                continue
            if not _unit_supported(proposal.unit, header, context):
                continue
            raw_tokens = [
                token
                for token in _NUMBER.findall(plain_cells[column])
                if _decimal(token) == Decimal(str(proposal.value))
            ]
            if len(raw_tokens) != 1:
                continue
            matches.append((raw_tokens[0], evidence.text[:end_offset].rstrip("\n")))
    if not matches:
        raise ProposalHydrationError("UNSUPPORTED_OR_WRONG_TABLE_CELL")
    if len(matches) != 1:
        raise ProposalHydrationError("AMBIGUOUS_TABLE_CELL")
    return matches[0]


def hydrate_artifact_dataset(
    proposal: ArtifactDatasetProposal,
    requirement: ArtifactDataRequirement,
    evidence: list[RetrievedEvidence],
    requested_output: str,
) -> ArtifactDataset:
    """Create the strict internal dataset using only application-owned provenance."""
    validate_trusted_artifact_evidence(evidence)
    expected_type = requirement.artifact_type
    if not proposal.rows:
        raise ProposalHydrationError("EMPTY_PROPOSAL")
    by_id = {item.chunk_id: item for item in evidence}
    rows: list[dict[str, object]] = []
    sources: list[SourceRecord] = []
    seen: set[tuple[str, str | None, float, str]] = set()
    for index, row in enumerate(proposal.rows, 1):
        item = by_id.get(row.evidence_id)
        if item is None:
            raise ProposalHydrationError("UNKNOWN_EVIDENCE_ID")
        if not re.fullmatch(r"[0-9a-f]{64}", item.source_checksum):
            raise ProposalHydrationError("INVALID_TRUSTED_SOURCE_CHECKSUM")
        identity = (
            _fold(row.label),
            _fold(row.series) if row.series else None,
            row.value,
            row.evidence_id,
        )
        if identity in seen:
            raise ProposalHydrationError("DUPLICATE_PROPOSAL_ROW")
        seen.add(identity)
        raw_value, quote = _match_row(row, item, requirement)
        row_id = f"row-{index}"
        rows.append(
            {
                "row_id": row_id,
                "label": row.label,
                "series": row.series or "",
                "value": row.value,
            }
        )
        sources.append(
            SourceRecord(
                source_record_id=f"source-{index}",
                row_id=row_id,
                field="value",
                raw_value=raw_value,
                normalized_numeric_value=row.value,
                unit=row.unit,
                metric=requirement.metric,
                region=row.label,
                year=requirement.year,
                population_scope=requirement.population_scope,
                residence_scope=requirement.residence_scope,
                document_title=item.document_title,
                document_id=item.document_id,
                page_number=item.page_number,
                chunk_id=item.chunk_id,
                exact_supporting_quote=quote,
                source_checksum=item.source_checksum,
            )
        )
    represented = {_fold(source.region or "") for source in sources}
    if requirement.regions and any(
        _fold(region) not in represented for region in requirement.regions
    ):
        raise ProposalHydrationError("MISSING_REQUIRED_TARGET")
    return ArtifactDataset(
        title=proposal.title,
        task_type="artifact_chart" if expected_type == "chart" else "artifact_table",
        rows=rows,
        columns=["row_id", "label", "series", "value"],
        units={"value": sources[0].unit or ""},
        source_records=sources,
        requested_output=requested_output,
    )


def resolved_chart_kind(
    proposal: ArtifactDatasetProposal, requirement: ArtifactDataRequirement
) -> str | None:
    """Resolve non-authoritative presentation preference without changing artifact identity."""
    if requirement.artifact_type != "chart" or not proposal.chart_kind:
        return None
    normalized = _fold(proposal.chart_kind.replace("_", " ").replace("-", " "))
    aliases = {
        "bar": "bar",
        "bar chart": "bar",
        "line": "line",
        "line chart": "line",
        "scatter": "scatter",
        "scatter plot": "scatter",
    }
    return aliases.get(normalized)
