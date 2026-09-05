import re
from decimal import Decimal

from backend.app.agent.models import (
    CalculationRequest,
    CalculationResult,
    ClaimDerivation,
    DraftAnswer,
    DraftClaim,
)
from backend.app.retrieval.models import RetrievedEvidence

_NUMBER = re.compile(r"(?<![\d,])\d[\d,]*(?:\.\d+)?(?!\d)")
_RELATIVE_PERCENT = re.compile(
    r"\b(?:percent(?:age)?\s+(?:higher|lower|difference)|relative\s+(?:percent(?:age)?\s+)?difference)\b",
    re.IGNORECASE,
)


def rounded_subtraction(left: float, right: float, *, places: int = 10) -> float:
    """Subtract decimal-form inputs without exposing binary float residue."""
    quantum = Decimal(1).scaleb(-places)
    return float((Decimal(str(left)) - Decimal(str(right))).quantize(quantum))


def _numbers(text: str) -> set[float]:
    return {float(value.replace(",", "")) for value in _NUMBER.findall(text)}


def _ordered_numbers(text: str) -> list[float]:
    return [float(value.replace(",", "")) for value in _NUMBER.findall(text)]


def comparison_operation(query: str, *, values_are_rates: bool = True) -> tuple[str, str]:
    """Choose comparison semantics deterministically from explicit user intent."""
    if values_are_rates and _RELATIVE_PERCENT.search(query):
        return "percentage_difference", "percent"
    return "difference", "percentage_points" if values_are_rates else "count"


def validated_calculation(
    query: str,
    request: CalculationRequest,
    evidence: list[RetrievedEvidence],
) -> CalculationResult | None:
    """Validate copied operands and apply application-owned operation semantics."""
    by_id = {item.chunk_id: item for item in evidence}
    selected = [by_id[value] for value in request.evidence_ids if value in by_id]
    if len(selected) != len(set(request.evidence_ids)) or len(request.values) != 2:
        return None
    available_numbers = set().union(*(_numbers(item.text) for item in selected))
    if not all(value in available_numbers for value in request.values):
        return None
    values_are_rates = all(
        "rate" in item.text.casefold() or "percent" in item.text.casefold() or "%" in item.text
        for item in selected
    )
    operation, unit = comparison_operation(query, values_are_rates=values_are_rates)
    left, right = request.values
    result = (
        float(
            (
                (Decimal(str(left)) - Decimal(str(right))) / abs(Decimal(str(right))) * Decimal(100)
            ).quantize(Decimal("0.0000000001"))
        )
        if operation == "percentage_difference" and right != 0
        else rounded_subtraction(left, right)
    )
    return CalculationResult(
        description=request.description,
        operation=operation,  # type: ignore[arg-type]
        values=request.values,
        result=result,
        evidence_ids=list(dict.fromkeys(request.evidence_ids)),
        unit=unit,  # type: ignore[arg-type]
    )


def enrich_source_claims(
    draft: DraftAnswer, evidence: list[RetrievedEvidence], query: str
) -> DraftAnswer:
    """Fill missing structured source fields using claim text and trusted metadata."""
    by_id = {item.chunk_id: item for item in evidence}
    query_years = [int(value) for value in _NUMBER.findall(query) if len(value) == 4]
    enriched: list[DraftClaim] = []
    for claim in draft.claims:
        if claim.document_derived:
            enriched.append(claim)
            continue
        selected = [by_id[value] for value in claim.evidence_ids if value in by_id]
        regions = {item.region for item in selected}
        values = [
            value
            for value in _ordered_numbers(claim.text)
            if not (1900 <= value <= 2099 and value.is_integer())
        ]
        section = " ".join(part for item in selected for part in item.section_path).casefold()
        text = claim.text.casefold()
        enriched.append(
            claim.model_copy(
                update={
                    "metric": claim.metric or ("literacy_rate" if "literacy" in text else None),
                    "region": claim.region or (next(iter(regions)) if len(regions) == 1 else None),
                    "year": claim.year or (query_years[0] if len(set(query_years)) == 1 else None),
                    "population_scope": claim.population_scope
                    or ("persons" if "persons" in text or "persons" in section else None),
                    "residence_scope": claim.residence_scope
                    or next(
                        (scope for scope in ("total", "rural", "urban") if scope in text),
                        None,
                    ),
                    "value": (
                        claim.value if claim.value is not None else values[0] if values else None
                    ),
                    "unit": claim.unit
                    or ("percent" if "%" in claim.text or "percent" in text else None),
                }
            )
        )
    return draft.model_copy(update={"claims": enriched})


def add_deterministic_derived_claims(
    draft: DraftAnswer, calculations: list[CalculationResult]
) -> DraftAnswer:
    """Replace model-authored derived comparisons with validated application output."""
    source_claims = [claim for claim in draft.claims if not claim.document_derived]
    derived: list[DraftClaim] = []
    for index, calculation in enumerate(calculations, start=1):
        inputs: list[DraftClaim] = []
        for operand in calculation.values:
            match = next(
                (
                    claim
                    for claim in source_claims
                    if claim.value is not None
                    and abs(claim.value - operand) < 1e-9
                    and set(claim.evidence_ids) & set(calculation.evidence_ids)
                    and claim not in inputs
                ),
                None,
            )
            if match is None:
                inputs = []
                break
            inputs.append(match)
        if len(inputs) != 2:
            continue
        left, right = inputs
        formatted = f"{abs(calculation.result):.10f}".rstrip("0").rstrip(".")
        relation = "higher" if calculation.result >= 0 else "lower"
        unit_text = "percentage points" if calculation.unit == "percentage_points" else "percent"
        text = (
            f"{left.region or 'The first value'} is {relation} than "
            f"{right.region or 'the second value'} by {formatted} {unit_text}."
        )
        evidence_ids = list(
            dict.fromkeys(value for claim in inputs for value in claim.evidence_ids)
        )
        derived.append(
            DraftClaim(
                claim_id=f"derived_comparison_{index}",
                text=text,
                evidence_ids=evidence_ids,
                document_derived=True,
                metric=left.metric,
                year=left.year,
                population_scope=left.population_scope,
                residence_scope=left.residence_scope,
                value=calculation.result,
                unit=calculation.unit,
                derivation=ClaimDerivation(
                    operation=calculation.operation,
                    operands=calculation.values,
                    result=calculation.result,
                    unit=calculation.unit,
                    input_claim_ids=[claim.claim_id for claim in inputs],
                ),
            )
        )
    return draft.model_copy(update={"claims": [*source_claims, *derived]})
