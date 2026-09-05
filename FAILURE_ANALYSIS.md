# Failure Analysis

## Unverified visual OCR

Twelve Karnataka chart and thematic-map pages produced non-empty Tesseract output, but inspection
showed missing labels, corrupted rotated text, and lost spatial associations. Even pages with mean
token confidence above 90 percent omitted visible chart labels. Token confidence therefore cannot
establish that a district is paired with the correct value, that a map region belongs to a legend
range, or that a percentage belongs to the correct chart category.

The raw OCR artifacts are quarantined under `data/processed/page-review/`. Their pages are marked
`excluded_unverified_visual`, and independent guards reject unverified OCR before chunking,
embedding, and Qdrant upload. Reviewed blank/decorative pages are explicit non-failure exclusions.
Every remaining page must receive an indexed status or `failed_page_mapping`.

This reduces retrieval coverage: questions answered only by excluded charts or maps may not have
reliable indexed evidence. The correct limitation is: “The available indexed text does not provide
reliable evidence for this question. Some chart or map pages were excluded because their labels and
values could not be extracted with citation-safe accuracy.” The system must not claim the fact is
absent from the authoritative PDF.

Future recovery requires either a human-reviewed transcription with an approved status and matching
PDF checksum, or layout-aware multimodal extraction followed by human verification. Automated OCR,
Codex, or Gemini output alone never qualifies as approved transcription.

## Comparison quote and arithmetic failures

Run `30e4237c-b44d-4703-aaae-93fb9bcbab71` retrieved valid evidence for both states but rejected
otherwise safe quotes because fluent claim filler was treated as literal source terminology. In the
Odisha table, `persons` was carried by the trusted section context; Karnataka's claim began with
`For`, which the old heuristic mistakenly treated as a required proper noun. The narrative source
also expressed statewide scope as “the State” rather than the word `total`. These are validation
representation mismatches, not reasons to weaken contiguous-source provenance.

The same run used relative percentage difference (about 3.37%) where a direct comparison of rates
required an arithmetic difference of 2.46 percentage points. Calculation operation and derived
claim construction are now application-owned. Safe quote-selection diagnostics record only evidence
ID, match type, reason code, and span length.

## Source-page follow-up failure

Run `6e064eef-bd15-4076-9218-94ec1436e965` resolved “those values” to 2011 Karnataka and Odisha
literacy, but classified the request as a generic lookup. Retrieval returned zero candidates for
the provenance-oriented wording, so evidence assessment stopped at `graceful_response`. The
checkpoint contained conversational messages but no structured validated-claim history, causing a
source request to be treated as a new factual lookup.

Explicit source-support intent now resolves only against bounded successful validated claims. It
rehydrates current Qdrant points by previously validated IDs and rejects missing points or changed
document/page/chunk/checksum provenance. Conversation prose is never cited. Derived differences
retain both input citations, while full exact table spans stay in structured citation objects
instead of being repeated in the visible answer.
