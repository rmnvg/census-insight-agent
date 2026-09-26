"""Trust scorecard: an independent audit of live agent answers against verified ground truth.

The checks deliberately do not reuse the agent's own validation code. Ground truth is a set of
facts, each a (region, metric, year, residence, population group, value) tuple checked against
the claim that states it, not a bag of numbers that may appear anywhere in the answer.

Every answer gets exactly one verdict:

- `passed`: every expected fact is stated by a claim with the right region, year, and scope; no
  claim contradicts one; every value is independently found in its own cited quote.
- `wrong_answer`: a claim states an expected fact's region, metric, year, and scope with a
  different value (a swapped label, a wrong-year or wrong-group column), a derived value does not
  recompute, or a question that must be refused was answered with numbers.
- `unsupported`: a value is not on its region's row in any cited quote, or the quote's own column
  headers give it a different year, residence, or group than the claim.
- `incomplete`: nothing is wrong, but an expected fact, derived value, or artifact is missing.
- `false_refusal`: an answerable question was declined, or answered with no numbers at all (only
  a clarifying question or caveat). Reported apart from wrong answers: declining is the safe
  failure.
- `error`: an operational failure; no answer to judge.
"""

import re
import statistics
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, model_validator

from backend.app.trust_provenance import normalize_label, support

_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w])")
_TOLERANCE = 0.005


Residence = Literal["total", "rural", "urban"]
Verdict = Literal["passed", "wrong_answer", "unsupported", "incomplete", "false_refusal", "error"]
_GROUP_WORDS = {
    "scheduled castes": ("scheduled caste",),
    "scheduled tribes": ("scheduled tribe",),
    "females": ("female", "women"),
    "males": (" male", "men "),
    "children": ("child", "0-6"),
}


class ExpectedFact(BaseModel):
    """One verified cell: which region, metric, year, and scope it describes, and its value."""

    region: str
    metric: str  # lower-case keyword the claim's metric or text must contain, e.g. "sex ratio"
    value: float
    year: int = 2011
    residence: Residence = "total"
    group: str | None = None  # None: all persons; else a key of _GROUP_WORDS

    def label(self) -> str:
        scope = "" if self.residence == "total" else f" {self.residence}"
        group = f" ({self.group})" if self.group else ""
        return f"{self.region}{scope} {self.metric}{group} {self.year} = {self.value:g}"


class TrustCase(BaseModel):
    case_id: str
    category: Literal["lookup", "comparison", "ranking", "chart", "refusal"]
    question: str
    expect_refusal: bool = False
    expected_facts: list[ExpectedFact] = Field(default_factory=list)
    # Values the application must compute (e.g. a comparison gap), checked on derived claims.
    expected_derived: list[float] = Field(default_factory=list)
    # Legacy loose checks, used only when a case has no expected facts.
    expected_values: list[float] = Field(default_factory=list)
    expected_labels: list[str] = Field(default_factory=list)
    expected_artifact: Literal["chart", "table"] | None = None
    # Earlier turns asked in the same session before `question`; only the last answer is scored.
    setup_turns: list[str] = Field(default_factory=list)
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
    repeat: int = 1
    verdict: Verdict = "error"
    # Claims stating an expected fact's slot with a different value.
    misattributed_claims: list[str] = Field(default_factory=list)
    # Human-readable reasons the case failed, in the order they were found.
    failures: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _legacy_verdict(cls, data: Any) -> Any:
        """Scorecards written before verdicts existed: infer one from what they recorded."""
        if not isinstance(data, dict) or "verdict" in data:
            return data
        if data.get("error_code") or data.get("outcome") == "error":
            verdict = "error"
        elif data.get("passed"):
            verdict = "passed"
        elif data.get("outcome") == "refused" and data.get("category") != "refusal":
            verdict = "false_refusal"
        elif data.get("ungrounded_claims"):
            verdict = "unsupported"
        elif data.get("category") == "refusal" or data.get("arithmetic_errors"):
            verdict = "wrong_answer"
        else:
            verdict = "incomplete"
        return {**data, "verdict": verdict}


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
    # Scorer version 2 checks facts and provenance; older scorecards load with the defaults.
    scorer_version: int = 1
    false_refusals: int = 0
    unsupported_answers: int = 0
    incomplete_answers: int = 0
    misattributed_claims: int = 0
    repeats: int = 1
    # Cases whose runs disagreed (some passed, some did not).
    unstable_cases: list[str] = Field(default_factory=list)


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
    parts = [fact.label() for fact in case.expected_facts]
    if not case.expected_facts and case.expected_values:
        parts.append(", ".join(f"{value:g}" for value in case.expected_values))
    if case.expected_derived:
        parts.append("computed " + ", ".join(f"{value:g}" for value in case.expected_derived))
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


_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_SEX_RATIO_DEFINITION = re.compile(r"(?:females?|women)\s+per\s+[\d,]+\s+(?:males?|men)")
_GROUP_PATTERNS = {
    "scheduled castes": re.compile(r"\bscheduled castes?\b"),
    "scheduled tribes": re.compile(r"\bscheduled tribes?\b"),
    "females": re.compile(r"\bfemales?\b|\bwomen\b"),
    "males": re.compile(r"\bmales?\b|\bmen\b"),
    "children": re.compile(r"\bchild(?:ren)?\b|\b0\s*-\s*6\b"),
}


def _half_unit(value: float) -> Decimal:
    """Half a unit of the last digit a figure is stated to, e.g. 0.05 for 72.9."""
    digits = f"{value:.10f}".rstrip("0").rstrip(".")
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    return Decimal(5) / Decimal(10) ** (decimals + 1)


class _Scope(BaseModel):
    region: str
    metric_text: str
    year: int | None
    residence: Residence
    group: str | None


def _scope(claim: dict[str, Any]) -> _Scope:
    text = _SEX_RATIO_DEFINITION.sub(" ", str(claim.get("text", "")).casefold())
    metric_text = f"{claim.get('metric') or ''} {text}".casefold()
    year = claim.get("year")
    if year is None:
        years = _YEAR.findall(text)
        year = int(years[0]) if len(set(years)) == 1 else None
    residence = str(claim.get("residence_scope") or "").casefold()
    if residence not in ("total", "rural", "urban"):
        residence = "rural" if "rural" in text else "urban" if "urban" in text else "total"
    # Models sometimes put the sex-ratio unit ("females per 1000 males") in population_scope.
    population = _SEX_RATIO_DEFINITION.sub(" ", str(claim.get("population_scope") or "").casefold())
    described = f"{population} {text}"
    group = next(
        (name for name, pattern in _GROUP_PATTERNS.items() if pattern.search(described)), None
    )
    return _Scope(
        region=normalize_label(str(claim.get("region") or "")),
        metric_text=metric_text,
        year=int(year) if year is not None else None,
        residence=cast(Residence, residence),
        group=group,
    )


def _same_slot(scope: _Scope, fact: ExpectedFact) -> bool:
    if not scope.region or scope.region != normalize_label(fact.region):
        return False
    if fact.metric not in scope.metric_text:
        return False
    # "Child sex ratio" is a different metric from "sex ratio".
    if "child" not in fact.metric and "child" in scope.metric_text:
        return False
    if scope.year is not None and scope.year != fact.year:
        return False
    return scope.residence == fact.residence and scope.group == fact.group


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
        verdict="error",
        latency_seconds=round(latency_seconds, 1),
        attempts=attempts,
        source_note=case.source_note,
        error_code=error_code,
        trace_id=trace_id,
        failures=[f"error {error_code}"] if error_code else [],
    )
    if response is None:
        return base
    claims: list[dict[str, Any]] = response.get("claims", [])
    citations = {item["citation_id"]: item for item in response.get("citations", [])}
    answer = _body(str(response.get("answer", "")))
    refused = bool(response.get("refusal"))
    unsupported: list[str] = []
    arithmetic: list[str] = []
    misattributed: list[str] = []
    uncited = 0
    observed: list[float] = []
    derived_values: list[float] = []
    satisfied: set[int] = set()
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
            derived_values.append(float(value))
            recomputed = _recompute(derivation)
            if recomputed is None or abs(recomputed - float(derivation.get("result", 0))) > 1e-6:
                arithmetic.append(claim.get("text", ""))
            if not _close(float(value), float(derivation.get("result", value))):
                arithmetic.append(claim.get("text", ""))
            continue
        scope = _scope(claim)
        for index, fact in enumerate(case.expected_facts):
            if not _same_slot(scope, fact):
                continue
            if abs(Decimal(str(value)) - Decimal(str(fact.value))) <= _half_unit(fact.value):
                satisfied.add(index)
            else:
                misattributed.append(f"{claim.get('text', '')} (verified: {fact.label()})")
        reason = support(
            claim,
            cited,
            year=scope.year,
            residence=scope.residence,
            group=scope.group,
            tolerance=Decimal("0.005"),
        )
        if reason is not None:
            unsupported.append(f"{claim.get('text', '')}: {reason}")

    artifact_types = [str(item.get("artifact_type")) for item in response.get("artifacts", [])]
    missing: list[str] = []
    if not case.expect_refusal:
        missing += [
            fact.label() for index, fact in enumerate(case.expected_facts) if index not in satisfied
        ]
        if not case.expected_facts:
            text_numbers = [float(number) for number in _numbers(answer)]
            for expected in case.expected_values:
                if not any(_close(expected, value) for value in [*observed, *text_numbers]):
                    missing.append(f"{expected:g}")
        for expected in case.expected_derived:
            # A gap is the same fact whichever side is subtracted ("lower by 48" or "higher by
            # 48"); the direction is checked by the arithmetic and the sentence, not here.
            if not any(
                abs(abs(Decimal(str(value))) - abs(Decimal(str(expected)))) <= _half_unit(expected)
                for value in derived_values
            ):
                missing.append(f"computed {expected:g}")
        for label in case.expected_labels:
            if label.casefold() not in answer.casefold():
                missing.append(label)
        if case.expected_artifact and case.expected_artifact not in artifact_types:
            missing.append(f"{case.expected_artifact} artifact")

    verdict: Verdict
    if case.expect_refusal:
        verdict = "passed" if refused and not observed else "wrong_answer"
    elif refused or not observed:
        # Declining, or replying with only a question or caveat, leaves nothing to check.
        verdict = "false_refusal"
    elif misattributed or arithmetic:
        verdict = "wrong_answer"
    elif unsupported or uncited:
        verdict = "unsupported"
    elif missing:
        verdict = "incomplete"
    else:
        verdict = "passed"
    failures = [
        *(
            ["answered a question it must refuse"]
            if case.expect_refusal and verdict != "passed"
            else []
        ),
        *(["declined an answerable question"] if verdict == "false_refusal" else []),
        *(f"misattributed: {item}" for item in misattributed),
        *(f"arithmetic: {item}" for item in arithmetic),
        *(f"unsupported: {item}" for item in unsupported),
        *([f"{uncited} uncited claim(s)"] if uncited else []),
        *(f"missing {item}" for item in missing if verdict != "false_refusal"),
    ]
    return base.model_copy(
        update={
            "passed": verdict == "passed",
            "verdict": verdict,
            "outcome": "refused" if refused else "answered",
            "answer_excerpt": answer[:400],
            "observed_values": observed[:60],
            "missing": missing,
            "claims_checked": len(claims),
            "ungrounded_claims": unsupported,
            "misattributed_claims": misattributed,
            "arithmetic_errors": arithmetic,
            "uncited_claims": uncited,
            "citation_count": len(citations),
            "artifact_types": artifact_types,
            "error_code": None,
            "trace_id": str(response.get("trace_id") or trace_id or "") or None,
            "failures": failures,
        }
    )


def summarize(results: list[CaseResult], model: str) -> Scorecard:
    answerable = [item for item in results if item.category != "refusal"]
    refusals = [item for item in results if item.category == "refusal"]
    latencies = [item.latency_seconds for item in results if item.outcome != "error"]
    outcomes: dict[str, set[bool]] = {}
    for item in results:
        outcomes.setdefault(item.case_id, set()).add(item.passed)

    def count(verdict: str) -> int:
        return sum(item.verdict == verdict for item in results)

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
        # The failure a trustworthy agent must never show: a stated fact that is false.
        wrong_answers=count("wrong_answer"),
        median_latency_seconds=round(statistics.median(latencies), 1) if latencies else 0,
        results=results,
        scorer_version=2,
        false_refusals=count("false_refusal"),
        unsupported_answers=count("unsupported"),
        incomplete_answers=count("incomplete"),
        misattributed_claims=sum(len(item.misattributed_claims) for item in results),
        repeats=max((item.repeat for item in results), default=1),
        unstable_cases=sorted(case for case, seen in outcomes.items() if len(seen) > 1),
    )


def run_metrics(card: Scorecard) -> dict[str, float]:
    """Rates for tracking a scorecard across models and prompt changes.

    Refusal is the positive class: precision is how often a refusal was the right call, recall is
    how many of the questions that must be refused were. Latency percentiles exclude errors.
    """
    refused = [item for item in card.results if item.outcome == "refused"]
    correct_refusals = sum(item.passed for item in refused if item.category == "refusal")
    latencies = sorted(item.latency_seconds for item in card.results if item.outcome != "error")
    p95_index = max(0, round(0.95 * len(latencies)) - 1)
    return {
        "pass_rate": _rate(card.cases_passed, card.cases_total),
        "answer_accuracy": _rate(card.answerable_passed, card.answerable_total),
        "refusal_precision": _rate(correct_refusals, len(refused)),
        "refusal_recall": _rate(card.refusal_passed, card.refusal_total),
        "grounded_claim_rate": _rate(
            card.claims_checked - card.ungrounded_claims, card.claims_checked
        ),
        "wrong_answers": float(card.wrong_answers),
        "false_refusal_rate": _rate(card.false_refusals, card.answerable_total, empty=0.0),
        "misattributed_claims": float(card.misattributed_claims),
        "unstable_cases": float(len(card.unstable_cases)),
        "median_latency_seconds": card.median_latency_seconds,
        "p95_latency_seconds": latencies[p95_index] if latencies else 0.0,
    }


def _rate(numerator: int, denominator: int, *, empty: float = 1.0) -> float:
    return round(numerator / denominator, 4) if denominator else empty
