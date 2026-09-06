import re
from typing import Literal

from pydantic import BaseModel

from backend.app.agent.models import DraftClaim, EvidenceSpan
from backend.app.retrieval.models import RetrievedEvidence

CITATION_QUOTE_MAX_CHARS = 1600
# Every document in this corpus is exclusively a Census 2011 report (see README/DESIGN.md); used
# only as the implicit year for a sentence that states no year at all (see quote_rejection_reason).
_CORPUS_DEFAULT_YEAR = 2011
# "total" is deliberately excluded: it is this corpus's documented default residence scope (see
# DESIGN.md's "Total persons" convention), so source prose routinely omits it for the baseline
# case (e.g. "the number of literates" rather than "the total number of literates") while a
# claim may naturally include the word as ordinary English ("the total number of X") without
# asserting a Census-specific scope at all. Rural/urban/male/female/persons are genuine
# deviations from that default and must still be confirmed present in the trusted context.
CATEGORY_TERMS = {"rural", "urban", "male", "males", "female", "females", "persons"}
METRIC_TERMS = {"literacy", "population", "density", "worker", "workers", "ratio", "growth"}
_HTML_TAG = re.compile(r"<[^>]+>")


class QuoteSelectionDiagnostic(BaseModel):
    evidence_id: str
    match_type: Literal["table", "prose", "none"]
    rejection_reason_code: str | None = None
    selected_span_length: int = 0


def _plain(value: str) -> str:
    return _HTML_TAG.sub("", value).replace("**", "").replace("__", "")


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z]+", _plain(value).casefold()))


def _number_values(value: str) -> set[float]:
    return {
        float(match.replace(",", ""))
        for match in re.findall(r"(?<![\d,])\d[\d,]*(?:\.\d+)?(?!\d)", value)
    }


def _years(value: str) -> set[int]:
    return {int(number) for number in _number_values(value) if 1900 <= number <= 2099}


def _table_cell_supports(claim: DraftClaim, quote: str) -> bool:
    """Bind numeric claims to a row and column, rather than all words in the quoted table.

    Explicit table categories override document context. Blank multi-level headers inherit the
    preceding heading within that header row. Prose remains subject to semantic assessment.
    """
    rows = [
        [_plain(cell).strip() for cell in line.strip().strip("|").split("|")]
        for line in quote.splitlines()
        if line.strip().startswith("|")
    ]
    values = (
        {claim.value}
        if claim.value is not None
        else _number_values(claim.text) - _years(claim.text)
    )
    if not rows or not values:
        return True
    population = _tokens(claim.population_scope or claim.text) & {
        "persons",
        "male",
        "female",
        "males",
        "females",
    }
    residence = _tokens(claim.residence_scope or claim.text) & {"total", "rural", "urban"}
    expected_years = {claim.year} if claim.year else _years(claim.text)
    headers: list[list[str]] = []
    for row in rows:
        if all(re.fullmatch(r"[:\-\s]*", cell) for cell in row):
            continue
        numeric = set().union(*(_number_values(cell) for cell in row))
        # Column ordinal rows (1, 2, 3...) are layout metadata, never evidence values.
        if [cell for cell in row if cell] == [str(i + 1) for i in range(len(row))]:
            continue
        if not numeric or numeric <= _years(" ".join(row)):
            headers.append(row)
            continue
        row_labels = " ".join(cell for cell in row if not _number_values(cell))
        for column, cell in enumerate(row):
            if not values <= _number_values(cell):
                continue
            column_headers = []
            for header in headers:
                inherited = ""
                for part in header[: column + 1]:
                    if part:
                        inherited = part
                column_headers.append(inherited)
            context = " ".join([row_labels, *column_headers])
            context_tokens = _tokens(context)
            context_tokens |= {word.rstrip("s") for word in context_tokens}
            all_tokens = _tokens(quote)
            if (
                population
                and all_tokens & {"persons", "male", "female", "males", "females"}
                and not {word.rstrip("s") for word in population} <= context_tokens
            ):
                continue
            if (
                residence
                and all_tokens & {"total", "rural", "urban"}
                and not residence <= context_tokens
            ):
                continue
            cell_years = _years(" ".join(column_headers))
            # A row's year label is authoritative when years run vertically.
            row_years = _years(" ".join(row[:column]))
            if (
                expected_years
                and (cell_years or row_years)
                and not expected_years <= cell_years | row_years
            ):
                continue
            if (
                claim.region
                and _tokens(claim.region) <= all_tokens
                and not _tokens(claim.region) <= _tokens(row_labels)
            ):
                continue
            return True
    return False


def quote_rejection_reason(
    claim: DraftClaim, quote: str, evidence: RetrievedEvidence
) -> str | None:
    """Return a stable reason code when an exact quote does not support structured facts."""
    if not quote.strip():
        return "EMPTY_QUOTE"
    if quote not in evidence.text:
        return "NON_CONTIGUOUS_QUOTE"
    if not _table_cell_supports(claim, quote):
        return "TABLE_CELL_SCOPE_MISMATCH"

    claim_numbers = {number for number in _number_values(claim.text) if not 1900 <= number <= 2099}
    expected_values = {claim.value} if claim.value is not None else claim_numbers
    if expected_values and not all(
        any(abs(expected - actual) < 1e-9 for actual in _number_values(quote))
        for expected in expected_values
    ):
        return "VALUE_ABSENT"

    expected_years = {claim.year} if claim.year is not None else _years(claim.text)
    quote_years = _years(quote)
    # A sentence stating no year at all (quote_years empty) is treated as this corpus's
    # sole/default report year: every document here is exclusively a Census 2011 report, so a
    # figure's current-period sentence routinely omits the year even when an explicit comparison
    # year like 2001 appears elsewhere in the same passage (see DESIGN.md/hydration.py's
    # identical "2011 total persons" default for artifacts). This is narrower than it looks: it
    # only ever applies when the sentence names no year at all. A sentence naming any year still
    # must include the expected one — quote_years={2001} still correctly rejects an
    # expected_years={2011} claim, so a genuinely wrong-year sentence is never accepted here.
    implicit_corpus_year = not quote_years and expected_years == {_CORPUS_DEFAULT_YEAR}
    if expected_years and not expected_years <= quote_years and not implicit_corpus_year:
        return "YEAR_ABSENT"

    quote_tokens = _tokens(quote)
    trusted_context_tokens = (
        quote_tokens
        | _tokens(evidence.document_title)
        | _tokens(evidence.region)
        | _tokens(" ".join(evidence.section_path))
    )
    metric_tokens = (
        _tokens(claim.metric.replace("_", " ")) & METRIC_TERMS
        if claim.metric
        else _tokens(claim.text) & METRIC_TERMS
    )
    if metric_tokens and not metric_tokens <= trusted_context_tokens:
        return "METRIC_ABSENT"

    if claim.population_scope:
        scope_tokens = _tokens(claim.population_scope)
        if "children" in scope_tokens:
            # "children" is the natural way to phrase the 0-6-years population category, but
            # Census source prose consistently says "child sex ratio" (singular); this is a
            # morphological variant, not a different assertion, so it must not require the
            # source to literally say "children".
            scope_tokens = (scope_tokens - {"children"}) | {"child"}
        population_supported = scope_tokens <= trusted_context_tokens
        if claim.population_scope.casefold() == "persons":
            population_supported = (
                population_supported
                or {
                    "literates",
                    "age",
                }
                <= trusted_context_tokens
            )
        if not population_supported:
            return "POPULATION_SCOPE_ABSENT"
    if claim.residence_scope:
        residence_supported = _tokens(claim.residence_scope) <= trusted_context_tokens
        if claim.residence_scope.casefold() == "total":
            residence_supported = residence_supported or "state" in trusted_context_tokens
        if not residence_supported:
            return "RESIDENCE_SCOPE_ABSENT"
    if not claim.population_scope and not claim.residence_scope:
        categories = _tokens(claim.text) & CATEGORY_TERMS
        if categories and not categories <= trusted_context_tokens:
            return "CATEGORY_ABSENT"

    unit = claim.unit or (
        "percent" if _tokens(claim.text) & {"percent", "percentage"} or "%" in claim.text else None
    )
    if unit in {"percent", "percentage_points"} and not (
        quote_tokens & {"percent", "percentage", "cent", "rate"} or "%" in quote
    ):
        return "UNIT_ABSENT"

    if claim.region and claim.region.casefold() != evidence.region.casefold():
        return "REGION_METADATA_MISMATCH"
    if claim.region and not _tokens(claim.region) <= trusted_context_tokens:
        return "REGION_ABSENT"
    return None


def quote_supports_claim(claim: DraftClaim, quote: str, evidence: RetrievedEvidence) -> bool:
    return quote_rejection_reason(claim, quote, evidence) is None


def _table_span(claim: DraftClaim, evidence: RetrievedEvidence) -> EvidenceSpan | None:
    lines = list(re.finditer(r"(?m)^\|.*(?:\n|$)", evidence.text))
    claim_numbers = (
        {claim.value}
        if claim.value is not None
        else {
            value
            for value in _number_values(claim.text)
            if not (1900 <= value <= 2099 and value.is_integer())
        }
    )
    target = next(
        (
            line
            for line in lines
            if claim_numbers
            and all(
                any(abs(expected - actual) < 1e-9 for actual in _number_values(line.group()))
                for expected in claim_numbers
            )
        ),
        None,
    )
    if target is None:
        return None
    table_lines = [line for line in lines if line.start() <= target.start()]
    start = table_lines[0].start()
    if target.end() - start > CITATION_QUOTE_MAX_CHARS:
        eligible = [
            line for line in table_lines if target.end() - line.start() <= CITATION_QUOTE_MAX_CHARS
        ]
        start = eligible[0].start() if eligible else target.start()
    return EvidenceSpan(evidence_id=evidence.chunk_id, start_offset=start, end_offset=target.end())


def select_evidence_span_with_diagnostic(
    claim: DraftClaim, evidence: RetrievedEvidence
) -> tuple[EvidenceSpan | None, QuoteSelectionDiagnostic]:
    """Select an untouched exact span and return safe matching diagnostics."""
    rejection_codes: list[str] = []
    table = _table_span(claim, evidence)
    if table is not None:
        quote = evidence.text[table.start_offset : table.end_offset]
        reason = quote_rejection_reason(claim, quote, evidence)
        if reason is None:
            return table, QuoteSelectionDiagnostic(
                evidence_id=evidence.chunk_id,
                match_type="table",
                selected_span_length=table.end_offset - table.start_offset,
            )
        rejection_codes.append(reason)

    candidates = list(
        re.finditer(r"(?m)(?:^|(?<=[.!?])\s+)[^\n]+?(?:[.!?](?=\s|$)|$)", evidence.text)
    )
    supported: list[tuple[int, EvidenceSpan]] = []
    claim_tokens = _tokens(claim.text)
    for match in candidates:
        start, end = match.span()
        while start < end and evidence.text[start].isspace():
            start += 1
        if end - start > CITATION_QUOTE_MAX_CHARS:
            rejection_codes.append("QUOTE_TOO_LONG")
            continue
        quote = evidence.text[start:end]
        if quote.lstrip().startswith("|"):
            # A bare row loses its column headers; it cannot bypass table validation.
            continue
        reason = quote_rejection_reason(claim, quote, evidence)
        if reason is None:
            score = len(claim_tokens & _tokens(quote))
            supported.append(
                (
                    score,
                    EvidenceSpan(evidence_id=evidence.chunk_id, start_offset=start, end_offset=end),
                )
            )
        else:
            rejection_codes.append(reason)
    if supported:
        span = max(supported, key=lambda item: (item[0], -item[1].start_offset))[1]
        return span, QuoteSelectionDiagnostic(
            evidence_id=evidence.chunk_id,
            match_type="prose",
            selected_span_length=span.end_offset - span.start_offset,
        )
    reason = next(
        (
            code
            for code in (
                "TABLE_CELL_SCOPE_MISMATCH",
                "VALUE_ABSENT",
                "YEAR_ABSENT",
                "METRIC_ABSENT",
                "POPULATION_SCOPE_ABSENT",
                "RESIDENCE_SCOPE_ABSENT",
                "CATEGORY_ABSENT",
                "UNIT_ABSENT",
                "REGION_METADATA_MISMATCH",
                "REGION_ABSENT",
                "QUOTE_TOO_LONG",
            )
            if code in rejection_codes
        ),
        "NO_CANDIDATE_SPAN",
    )
    return None, QuoteSelectionDiagnostic(
        evidence_id=evidence.chunk_id,
        match_type="none",
        rejection_reason_code=reason,
    )


def select_evidence_span(claim: DraftClaim, evidence: RetrievedEvidence) -> EvidenceSpan | None:
    return select_evidence_span_with_diagnostic(claim, evidence)[0]


def resolve_evidence_span(span: EvidenceSpan, evidence: RetrievedEvidence) -> str:
    if span.evidence_id != evidence.chunk_id:
        raise ValueError("Evidence span references the wrong chunk")
    if span.end_offset > len(evidence.text) or span.start_offset >= span.end_offset:
        raise ValueError("Evidence span offsets are outside the trusted chunk")
    return evidence.text[span.start_offset : span.end_offset]
