from pydantic import BaseModel, Field

from backend.app.agent.models import AnswerClaim, CalculationResult, Citation, DraftAnswer
from backend.app.agent.quotes import (
    QuoteSelectionDiagnostic,
    resolve_evidence_span,
    select_evidence_span_with_diagnostic,
)
from backend.app.agent.scopes import unrequested_subgroups
from backend.app.retrieval.models import RetrievedEvidence


class CitationValidationResult(BaseModel):
    valid: bool
    claims: list[AnswerClaim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    error_codes: list[str] = Field(default_factory=list)
    quote_diagnostics: list[QuoteSelectionDiagnostic] = Field(default_factory=list)
    # (claim_id, evidence_id) pairs dropped because that chunk did not contain the claim's
    # value while another cited chunk did; the claim stays supported by verbatim quotes.
    pruned_citations: list[tuple[str, str]] = Field(default_factory=list)


def validate_and_materialize_citations(
    draft: DraftAnswer,
    evidence: list[RetrievedEvidence],
    calculations: list[CalculationResult] | None = None,
) -> CitationValidationResult:
    """Resolve model-selected chunk IDs into trusted citation metadata."""
    by_id = {item.chunk_id: item for item in evidence}
    citations: list[Citation] = []
    citation_by_claim_chunk: dict[tuple[str, str], str] = {}
    claims: list[AnswerClaim] = []
    errors: list[str] = []
    error_codes: list[str] = []
    quote_diagnostics: list[QuoteSelectionDiagnostic] = []
    pruned: list[tuple[str, str]] = []
    claim_citations: dict[str, list[str]] = {}
    for claim in draft.claims:
        # Prefer the calculation with exactly this claim's operation and operands. Several
        # calculations can cite the same chunks (total, rural, and urban from one table), and
        # pairing by evidence alone would hand each derived claim the first of them.
        matching_calculation = next(
            (
                item
                for item in calculations or []
                if claim.derivation is not None
                and claim.derivation.operation == item.operation
                and claim.derivation.operands == item.values
            ),
            None,
        ) or next(
            (
                item
                for item in calculations or []
                if set(item.evidence_ids) <= set(claim.evidence_ids)
            ),
            None,
        )
        is_derived = claim.document_derived or claim.derivation is not None
        if is_derived:
            derivation = claim.derivation
            if matching_calculation is None or derivation is None:
                errors.append(f"Derived claim {claim.claim_id} lacks a validated calculation")
                error_codes.append("INVALID_DERIVATION")
            elif (
                derivation.operation != matching_calculation.operation
                or derivation.operands != matching_calculation.values
                or abs(derivation.result - matching_calculation.result) > 1e-9
                or derivation.unit != matching_calculation.unit
                or claim.value is None
                or abs(claim.value - matching_calculation.result) > 1e-9
                or claim.unit != matching_calculation.unit
            ):
                errors.append(f"Derived claim {claim.claim_id} altered its calculation")
                error_codes.append("INVALID_DERIVATION")
        if not claim.evidence_ids:
            errors.append(f"Claim {claim.claim_id} has no citation evidence")
            error_codes.append("CLAIM_WITHOUT_EVIDENCE")
        citation_ids: list[str] = []
        unsupported: list[str] = []
        errors_for_subgroup: list[str] = []
        if is_derived and claim.derivation is not None:
            # A derived claim cites exactly what its validated input claims cite.
            for input_id in claim.derivation.input_claim_ids:
                inherited = claim_citations.get(input_id, [])
                if not inherited:
                    errors.append(
                        f"Derived claim {claim.claim_id} lacks source citation for {input_id}"
                    )
                    error_codes.append("DERIVED_MISSING_INPUT_CITATIONS")
                citation_ids.extend(value for value in inherited if value not in citation_ids)
        for evidence_id in (
            [] if is_derived and claim.derivation is not None else claim.evidence_ids
        ):
            item = by_id.get(evidence_id)
            if item is None:
                errors.append(f"Claim {claim.claim_id} references unknown evidence {evidence_id}")
                error_codes.append("UNKNOWN_EVIDENCE_ID")
                continue
            if item.page_number < 1:
                errors.append(f"Evidence {evidence_id} has an invalid page")
                error_codes.append("INVALID_PROVENANCE")
                continue
            if item.coverage_status.startswith("excluded_"):
                errors.append(f"Evidence {evidence_id} comes from an excluded page")
                error_codes.append("INVALID_PROVENANCE")
                continue
            if item.citation_snippet not in item.text:
                errors.append(f"Evidence {evidence_id} has a non-verbatim snippet")
                error_codes.append("NON_VERBATIM_SNIPPET")
                continue
            existing = next(
                (citation for citation in citations if citation.chunk_id == item.chunk_id), None
            )
            if is_derived:
                if existing is not None:
                    citation_ids.append(existing.citation_id)
                else:
                    errors.append(
                        f"Derived claim {claim.claim_id} lacks source citation for {evidence_id}"
                    )
                    error_codes.append("DERIVED_MISSING_INPUT_CITATIONS")
                continue
            request = f"{claim.text} {claim.metric or ''} {claim.population_scope or ''}"
            if unrequested_subgroups(" ".join(item.section_path), request):
                # A Scheduled Caste/Tribe or child table never supports a whole-population claim.
                errors_for_subgroup.append(evidence_id)
                unsupported.append(evidence_id)
                continue
            span, diagnostic = select_evidence_span_with_diagnostic(claim, item)
            quote_diagnostics.append(diagnostic)
            if span is None:
                unsupported.append(evidence_id)
                continue
            snippet = resolve_evidence_span(span, item)
            key = (claim.claim_id, item.chunk_id)
            citation_id = citation_by_claim_chunk.setdefault(
                key, f"citation-{len(citation_by_claim_chunk) + 1}"
            )
            citation_ids.append(citation_id)
            if not any(value.citation_id == citation_id for value in citations):
                citations.append(
                    Citation(
                        citation_id=citation_id,
                        document_id=item.document_id,
                        document_title=item.document_title,
                        page_number=item.page_number,
                        snippet=snippet,
                        chunk_id=item.chunk_id,
                        section_path=item.section_path,
                        evidence_span=span,
                    )
                )
        if unsupported:
            if citation_ids and not is_derived:
                pruned.extend((claim.claim_id, evidence_id) for evidence_id in unsupported)
            else:
                errors.extend(
                    f"Evidence {evidence_id} has no claim-supporting quote"
                    for evidence_id in unsupported
                )
                error_codes.append("CLAIM_QUOTE_NOT_FOUND")
                if errors_for_subgroup:
                    error_codes.append("SUBGROUP_TABLE_FOR_WHOLE_POPULATION_CLAIM")
        claim_citations[claim.claim_id] = citation_ids
        claims.append(
            AnswerClaim(
                claim_id=claim.claim_id,
                text=claim.text,
                citation_ids=citation_ids,
                document_derived=claim.document_derived,
                metric=claim.metric,
                region=claim.region,
                year=claim.year,
                population_scope=claim.population_scope,
                residence_scope=claim.residence_scope,
                value=claim.value,
                unit=claim.unit,
                derivation=claim.derivation,
            )
        )
    return CitationValidationResult(
        valid=not errors,
        claims=claims,
        citations=citations,
        errors=errors,
        error_codes=list(dict.fromkeys(error_codes)),
        quote_diagnostics=quote_diagnostics,
        pruned_citations=pruned,
    )


def render_citation(citation: Citation) -> str:
    section = " > ".join(citation.section_path)
    suffix = f"; {section}" if section else ""
    return f"[{citation.document_title}, p. {citation.page_number}{suffix} — “{citation.snippet}”]"
