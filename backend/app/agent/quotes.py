import re
from typing import Literal

from pydantic import BaseModel

from backend.app.agent.models import DraftClaim, EvidenceSpan
from backend.app.retrieval.models import RetrievedEvidence

CITATION_QUOTE_MAX_CHARS = 1600
CATEGORY_TERMS = {"total", "rural", "urban", "male", "males", "female", "females", "persons"}
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


def quote_rejection_reason(
    claim: DraftClaim, quote: str, evidence: RetrievedEvidence
) -> str | None:
    """Return a stable reason code when an exact quote does not support structured facts."""
    if not quote.strip():
        return "EMPTY_QUOTE"
    if quote not in evidence.text:
        return "NON_CONTIGUOUS_QUOTE"

    claim_numbers = {number for number in _number_values(claim.text) if not 1900 <= number <= 2099}
    expected_values = {claim.value} if claim.value is not None else claim_numbers
    if expected_values and not all(
        any(abs(expected - actual) < 1e-9 for actual in _number_values(quote))
        for expected in expected_values
    ):
        return "VALUE_ABSENT"

    expected_years = {claim.year} if claim.year is not None else _years(claim.text)
    if expected_years and not expected_years <= _years(quote):
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
        population_supported = _tokens(claim.population_scope) <= trusted_context_tokens
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
