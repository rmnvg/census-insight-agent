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
