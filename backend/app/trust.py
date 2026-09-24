"""Trust scorecard: an independent audit of live agent answers against verified ground truth.

The checks deliberately do not reuse the agent's own validation code. A case passes only when:

- an answerable question is answered with every expected value (and label/artifact);
- every numeric claim's value appears verbatim in at least one of its own cited quotes;
- every derived value recomputes exactly from its operands;
- every claim carries a citation;
- an unanswerable question is refused with no numeric claims at all.
"""

import re
import statistics
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from pydantic import BaseModel, Field

_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w])")
_TOLERANCE = 0.005


class TrustCase(BaseModel):
    case_id: str
    category: Literal["lookup", "comparison", "ranking", "chart", "refusal"]
    question: str
    expect_refusal: bool = False
    expected_values: list[float] = Field(default_factory=list)
    expected_labels: list[str] = Field(default_factory=list)
    expected_artifact: Literal["chart", "table"] | None = None
    source_note: str


class CaseResult(BaseModel):
    case_id: str
    category: str
    question: str
    expected: str
    passed: bool
    outcome: Literal["answered", "refused", "error"]
    answer_excerpt: str = ""
    observed_values: list[float] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    claims_checked: int = 0
    ungrounded_claims: list[str] = Field(default_factory=list)
    arithmetic_errors: list[str] = Field(default_factory=list)
    uncited_claims: int = 0
    citation_count: int = 0
    artifact_types: list[str] = Field(default_factory=list)
    latency_seconds: float = 0
    attempts: int = 1
    error_code: str | None = None
    trace_id: str | None = None
    source_note: str = ""


class Scorecard(BaseModel):
    generated_at: datetime
    model: str
    cases_total: int
    cases_passed: int
    answerable_total: int
    answerable_passed: int
    refusal_total: int
    refusal_passed: int
    claims_checked: int
    ungrounded_claims: int
    uncited_claims: int
    arithmetic_errors: int
    wrong_answers: int
    median_latency_seconds: float
    results: list[CaseResult]


def _numbers(text: str) -> list[Decimal]:
    values: list[Decimal] = []
    for token in _NUMBER.findall(text):
        try:
            values.append(Decimal(token.replace(",", "")))
        except InvalidOperation:
            continue
    return values


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= _TOLERANCE


def _body(answer: str) -> str:
    return answer.split("\n\nSources:\n", 1)[0]


def _expected_text(case: TrustCase) -> str:
    if case.expect_refusal:
        return "Refuse (no numeric claims)"
    parts = [", ".join(f"{value:g}" for value in case.expected_values)]
    if case.expected_labels:
        parts.append(", ".join(case.expected_labels))
    if case.expected_artifact:
        parts.append(f"{case.expected_artifact} artifact")
    return " · ".join(part for part in parts if part)


def _recompute(derivation: dict[str, Any]) -> float | None:
    operands = [float(value) for value in derivation.get("operands", [])]
    operation = derivation.get("operation")
    if len(operands) < 2:
        return None
    if operation == "sum":
        return sum(operands)
    if operation == "difference":
        return operands[0] - operands[1]
    if operation == "percentage_difference" and operands[1]:
        return (operands[0] - operands[1]) / abs(operands[1]) * 100
    return None


def score_case(
    case: TrustCase,
    response: dict[str, Any] | None,
    *,
    latency_seconds: float,
    attempts: int = 1,
    error_code: str | None = None,
    trace_id: str | None = None,
) -> CaseResult:
    base = CaseResult(
        case_id=case.case_id,
        category=case.category,
        question=case.question,
        expected=_expected_text(case),
        passed=False,
        outcome="error",
        latency_seconds=round(latency_seconds, 1),
        attempts=attempts,
        source_note=case.source_note,
        error_code=error_code,
        trace_id=trace_id,
    )
    if response is None:
        return base
    claims: list[dict[str, Any]] = response.get("claims", [])
    citations = {item["citation_id"]: item for item in response.get("citations", [])}
    answer = _body(str(response.get("answer", "")))
    refused = bool(response.get("refusal"))
    ungrounded: list[str] = []
    arithmetic: list[str] = []
    uncited = 0
    observed: list[float] = []
    for claim in claims:
        value = claim.get("value")
        cited = [citations[item] for item in claim.get("citation_ids", []) if item in citations]
        if not cited:
            uncited += 1
        if value is None:
            continue
        observed.append(float(value))
        derivation = claim.get("derivation")
        if derivation:
            recomputed = _recompute(derivation)
            if recomputed is None or abs(recomputed - float(derivation.get("result", 0))) > 1e-6:
                arithmetic.append(claim.get("text", ""))
            if not _close(float(value), float(derivation.get("result", value))):
                arithmetic.append(claim.get("text", ""))
            continue
        target = Decimal(str(value))
        quoted = [number for item in cited for number in _numbers(str(item.get("snippet", "")))]
        if not any(abs(number - target) <= Decimal("0.005") for number in quoted):
            ungrounded.append(claim.get("text", ""))

    artifact_types = [str(item.get("artifact_type")) for item in response.get("artifacts", [])]
    missing: list[str] = []
    if case.expect_refusal:
        passed = refused and not observed
        if not passed:
            missing.append("expected a refusal")
    else:
        text_numbers = [float(number) for number in _numbers(answer)]
        for expected in case.expected_values:
            if not any(_close(expected, value) for value in [*observed, *text_numbers]):
                missing.append(f"{expected:g}")
        for label in case.expected_labels:
            if label.casefold() not in answer.casefold():
                missing.append(label)
        if case.expected_artifact and case.expected_artifact not in artifact_types:
            missing.append(f"{case.expected_artifact} artifact")
        passed = not refused and not missing and not ungrounded and not arithmetic and not uncited
    return base.model_copy(
        update={
            "passed": passed,
            "outcome": "refused" if refused else "answered",
            "answer_excerpt": answer[:400],
            "observed_values": observed[:60],
            "missing": missing,
            "claims_checked": len(claims),
            "ungrounded_claims": ungrounded,
            "arithmetic_errors": arithmetic,
            "uncited_claims": uncited,
            "citation_count": len(citations),
            "artifact_types": artifact_types,
            "error_code": None,
            "trace_id": str(response.get("trace_id") or trace_id or "") or None,
        }
    )


def summarize(results: list[CaseResult], model: str) -> Scorecard:
    answerable = [item for item in results if item.category != "refusal"]
    refusals = [item for item in results if item.category == "refusal"]
    latencies = [item.latency_seconds for item in results if item.outcome != "error"]
    return Scorecard(
        generated_at=datetime.now(UTC),
        model=model,
        cases_total=len(results),
        cases_passed=sum(item.passed for item in results),
        answerable_total=len(answerable),
        answerable_passed=sum(item.passed for item in answerable),
        refusal_total=len(refusals),
        refusal_passed=sum(item.passed for item in refusals),
        claims_checked=sum(item.claims_checked for item in results),
        ungrounded_claims=sum(len(item.ungrounded_claims) for item in results),
        uncited_claims=sum(item.uncited_claims for item in results),
        arithmetic_errors=sum(len(item.arithmetic_errors) for item in results),
        # An answer that asserts numbers but misses the verified value, or answers a question
        # it should refuse: the failure mode a trustworthy agent must never show.
        wrong_answers=sum(
            1
            for item in results
            if item.outcome == "answered"
            and item.observed_values
            and (item.missing or item.category == "refusal")
        ),
        median_latency_seconds=round(statistics.median(latencies), 1) if latencies else 0,
        results=results,
    )
