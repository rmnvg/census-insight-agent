import re

from backend.app.agent.models import (
    AgentResponse,
    CalculationResult,
    DraftAnswer,
    DraftClaim,
    EvidenceReference,
    ValidatedClaimRecord,
    ValidatedClaimTurn,
)
from backend.app.retrieval.models import RetrievedEvidence

_SOURCE_SUPPORT = re.compile(
    r"\b(?:which|what|show|where).{0,40}\b(?:source|citation|report|page)s?\b|"
    r"\bwhere did (?:that|this|the) (?:number|value) come from\b",
    re.IGNORECASE,
)
_DEICTIC = re.compile(r"\b(?:those|that|these|previous|comparison)\b", re.IGNORECASE)


def is_source_support_query(query: str) -> bool:
    """Narrow deterministic safeguard for unmistakable citation follow-ups."""
    return bool(_SOURCE_SUPPORT.search(query))


def select_source_support_turn(
    query: str, history: list[ValidatedClaimTurn]
) -> tuple[ValidatedClaimTurn | None, str | None]:
    comparisons = [turn for turn in history if turn.task_type == "comparison" and turn.claims]
    if _DEICTIC.search(query) and comparisons:
        return comparisons[-1], None
    successful = [turn for turn in history if turn.claims]
    if len(successful) == 1:
        return successful[0], None
    if len(comparisons) == 1:
        return comparisons[0], None
    return None, "Which previous answer or comparison would you like source pages for?"


def source_support_query(turn: ValidatedClaimTurn) -> str:
    source_claims = [claim for claim in turn.claims if not claim.document_derived]
    descriptors = [
        " ".join(
            value
            for value in (
                claim.region,
                claim.metric,
                str(claim.year) if claim.year else None,
                claim.displayed_value,
            )
            if value
        )
        for claim in source_claims
    ]
    return "Return the physical PDF source pages supporting: " + "; ".join(descriptors)


def build_validated_claim_turn(
    run_id: str,
    task_type: str,
    response: AgentResponse,
    evidence: list[RetrievedEvidence],
) -> ValidatedClaimTurn:
    by_id = {item.chunk_id: item for item in evidence}
    citations = {item.citation_id: item for item in response.citations}
    records: list[ValidatedClaimRecord] = []
    for claim in response.claims:
        claim_citations = [
            citations[citation_id] for citation_id in claim.citation_ids if citation_id in citations
        ]
        evidence_ids = list(dict.fromkeys(item.chunk_id for item in claim_citations))
        records.append(
            ValidatedClaimRecord(
                claim_id=claim.claim_id,
                text=claim.text,
                metric=claim.metric,
                region=claim.region,
                year=claim.year,
                population_scope=claim.population_scope,
                residence_scope=claim.residence_scope,
                value=round(claim.value, 10) if claim.value is not None else None,
                displayed_value=(
                    f"{abs(claim.value):.10f}".rstrip("0").rstrip(".")
                    if claim.value is not None
                    else None
                ),
                unit=claim.unit,
                document_derived=claim.document_derived,
                derivation=claim.derivation,
                evidence_ids=evidence_ids,
                citation_ids=[item.citation_id for item in claim_citations],
                document_ids=[item.document_id for item in claim_citations],
                page_numbers=[item.page_number for item in claim_citations],
                source_checksums=[by_id[item.chunk_id].source_checksum for item in claim_citations],
            )
        )
    return ValidatedClaimTurn(run_id=run_id, task_type=task_type, claims=records)  # type: ignore[arg-type]


def evidence_references(claims: list[ValidatedClaimRecord]) -> dict[str, EvidenceReference]:
    references: dict[str, EvidenceReference] = {}
    for claim in claims:
        for index, evidence_id in enumerate(claim.evidence_ids):
            references.setdefault(
                evidence_id,
                EvidenceReference(
                    evidence_id=evidence_id,
                    document_id=claim.document_ids[index],
                    page_number=claim.page_numbers[index],
                    source_checksum=claim.source_checksums[index],
                ),
            )
    return references


def source_support_draft(
    claims: list[ValidatedClaimRecord], evidence: list[RetrievedEvidence]
) -> tuple[DraftAnswer, list[CalculationResult]]:
    by_id = {item.chunk_id: item for item in evidence}
    drafts: list[DraftClaim] = []
    lines: list[str] = []
    calculations: list[CalculationResult] = []
    for claim in claims:
        if claim.document_derived:
            if claim.derivation is None:
                continue
            source_claims = {item.claim_id: item for item in claims if not item.document_derived}
            input_claims = [
                source_claims[item]
                for item in claim.derivation.input_claim_ids
                if item in source_claims
            ]
            evidence_ids = list(
                dict.fromkeys(value for item in input_claims for value in item.evidence_ids)
            )
            derivation = claim.derivation.model_copy(
                update={"result": round(claim.derivation.result, 10)}
            )
            calculations.append(
                CalculationResult(
                    description="Rehydrated validated comparison",
                    operation=derivation.operation,
                    values=derivation.operands,
                    result=derivation.result,
                    evidence_ids=evidence_ids,
                    unit=derivation.unit,
                )
            )
            drafts.append(
                DraftClaim(
                    **claim.model_dump(
                        include={
                            "claim_id",
                            "text",
                            "metric",
                            "region",
                            "year",
                            "population_scope",
                            "residence_scope",
                            "value",
                            "unit",
                        }
                    ),
                    evidence_ids=evidence_ids,
                    document_derived=True,
                    derivation=derivation,
                )
            )
            lines.append(
                f"- Difference {claim.displayed_value} percentage points — derived from "
                "the two cited source values."
            )
            continue
        if not claim.evidence_ids:
            continue
        item = by_id[claim.evidence_ids[0]]
        value = claim.displayed_value or "value"
        suffix = "%" if claim.unit == "percent" else f" {claim.unit or ''}".rstrip()
        text = (
            f"{claim.region or item.region} {value}{suffix} — {item.document_title}, "
            f"physical PDF page {item.page_number}."
        )
        drafts.append(
            DraftClaim(
                claim_id=claim.claim_id,
                text=text,
                evidence_ids=claim.evidence_ids,
                metric=claim.metric,
                region=claim.region,
                year=claim.year,
                population_scope=claim.population_scope,
                residence_scope=claim.residence_scope,
                value=claim.value,
                unit=claim.unit,
            )
        )
        lines.append(f"- {text}")
    return DraftAnswer(answer_markdown="\n".join(lines), claims=drafts), calculations
