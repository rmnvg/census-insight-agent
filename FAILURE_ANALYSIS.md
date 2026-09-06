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

## Artifact execution failures

Run `ce1f60be-9074-4216-8fdb-1859af80b45a` failed before artifact execution. A combined Karnataka
and Odisha search retrieved ten candidates, including Odisha's statewide page-82 table at rank ten.
Global assessment packing admitted nine chunks (11,669 of 12,000 characters) and silently excluded
that table. The evidence assessor therefore correctly reported that its input lacked a direct Odisha
value. This was neither an executor failure nor a provider hallucination.

Artifact requests now carry a typed source-data requirement separate from chart/table output intent.
Every comparison target is searched with its own region filter and normalized query. Assessment
packing reserves one high-value statewide candidate per target before filling remaining capacity;
maps, graph descriptions, definitions, and contents cannot consume the only slot for another target.
If all target reservations cannot fit, processing stops with `EVIDENCE_BUDGET_INSUFFICIENT` rather
than pretending the missing side was assessed. The 12,000-character budget is unchanged.

Session `252701e42a004f2a9660c82727c9df27` failed in `prepare_artifact` before the executor was called.
The provider rejected the full internal `ArtifactDataset` constrained-output schema with
`INVALID_ARGUMENT` because it produced too many serving states. That schema included nested source
provenance and derivation definitions, a checksum regex, numeric and array bounds, enums, and
arbitrary row dictionaries. Those are necessary internal trust constraints but unnecessary model
choices. Artifact preparation now requests only a small semantic proposal and deterministically
hydrates the unchanged internal contract from current-run trusted evidence. Provider schema,
timeout, rate-limit, authentication, invalid-output, and availability failures become sanitized
typed operational responses; they never become evidence refusals and never enter execution repair.

Session `72e07c82a2484a18910d2236612c6ce7` later proved that the smaller schema was accepted, but the
model-controlled free-form `artifact_type` caused deterministic hydration to reject a semantically
equivalent presentation label with `WRONG_ARTIFACT_TYPE`. The exact returned label was not retained
in the sanitized trace, so it cannot be reconstructed safely. Artifact identity has now been removed
from the proposal entirely and is copied from the application-controlled requirement. The same fix
canonicalizes “total persons” to Total residence before evidence validation. Non-timeout hydration
failures omit timeout durations and record authoritative type/scopes plus
`executor_submitted=false`.

A subsequent fresh live chart request returned HTTP 422 while constructing `SourceRecord`: hybrid
search had retrieved Qdrant points with valid checksums but omitted `source_checksum` when mapping
payloads into `RetrievedEvidence`. The model defaulted the missing field to an empty string, so the
strict checksum regex correctly stopped artifact creation. Hybrid retrieval now copies the checksum
explicitly, missing checksums are provenance errors, and hydration rejects malformed trusted
checksums with a typed operational failure before executor submission. The first faulty boundary
was `HybridRetrievalService._evidence` in `backend/app/retrieval/service.py`; Qdrant itself retained
valid checksums. `RetrievedEvidence` no longer has an empty default and enforces lowercase SHA-256.
Artifact evidence is checked at retrieval completion and again at hydration. Internal provenance or
Pydantic contract failures now return sanitized `INTERNAL_PROVENANCE_INVALID` HTTP 500 responses
and persist a failed trace; request-body validation continues to use HTTP 422.

The next live retry (`aaaad1db-d6af-4aa8-88f4-6dbd7ccf2467`) passed checksum validation and selected
the correct Karnataka page-50 and Odisha page-82 rows, but hydration returned
`UNSUPPORTED_OR_WRONG_TABLE_CELL`. The classifier supplied the semantically equivalent plural
metric `literacy rates`, while the trusted tables and prior validated claims use `literacy rate`.
Exact metric-token matching therefore rejected the otherwise correct rows. The shared artifact
requirement canonicalizer now maps only explicit literacy-rate aliases (including plural and
“effective” variants) to `literacy rate`; it does not loosen table-cell, scope, or value matching.

Run `50a9dce1-3c85-4e17-ac96-a4568233f208` then passed evidence selection, proposal hydration, and
dataset validation, but stopped at `code_policy_validation`. The generated program assigned literal
safe `Path("output/...")` values to variables before calling `savefig`, `to_csv`, and `write_text`;
the original validator recognized only inline path literals. It also rejected a no-op creation of
the pre-created `output` directory. No executor submission occurred. The policy now performs narrow
static propagation for directly assigned path literals, invalidates aliases on rebinding, and
allows `mkdir` only when the receiver resolves exactly to the job's `output` root. Dynamic paths,
absolute paths, traversal, output subdirectory creation, and writes outside `output` remain
forbidden. Offline replay confirms the exact saved program now passes policy validation.

The same trace exposed a second pre-execution contract defect: the code-generation prompt displayed
the bare dataset even though the executor writes an `input.json` envelope containing `dataset` and
`source_manifest`. A policy-valid version of the old program would therefore have selected the
wrong JSON keys and emitted an incomplete manifest. Generation and one-time runtime-repair prompts
now show the exact envelope, require reads from `payload["dataset"]`, require the supplied manifest
to be copied unchanged, and state that `output` already exists. Code-policy violations remain
non-repairable and no deterministic finished chart is substituted for model-generated code.

On the next live attempt (`1d2282d2-67dd-4c38-a92a-4e6f93391c90`), all evidence and dataset checks
again passed, and the generated program followed the corrected envelope contract. Policy validation
stopped it solely because it used built-in `open()` to read `input.json` and write the declared
manifest path. The security requirement forbids `open` on arbitrary paths, not safe job-scoped I/O.
The policy now admits only constant-mode reads (`r`, `rt`, `rb`) of exactly `input.json` and
constant-mode writes (`w`, `wt`, `wb`) beneath `output/`, including statically proven path aliases.
Other paths, append/update modes, dynamic paths or modes, absolute paths, and traversal remain
blocked. The generation prompt still prefers restricted `Path.read_text`/`Path.write_text` calls.

- **Generated-code timeout:** the worker kills the generated program's process group at the bounded
  deadline, records `EXECUTION_TIMEOUT`, and returns no artifact. Timeouts are not repaired because
  retrying potentially hostile or unbounded behavior is unsafe.
- **Code-policy violation:** forbidden imports, dynamic execution, dunder access, suspicious
  attributes, and unsafe paths produce `CODE_POLICY_VIOLATION` before subprocess execution. Policy
  failures never enter the repair loop.
- **Chart with incorrect data:** a visually valid PNG is insufficient. The backend compares the
  companion CSV cell-for-cell with the approved `ArtifactDataset` and validates the complete source
  manifest. A mismatch becomes `INVALID_ARTIFACT`; nothing is published.
- **Executor unavailable:** a missing/stale heartbeat or result deadline produces
  `EXECUTOR_UNAVAILABLE`. The agent returns a graceful artifact failure and does not imply that an
  artifact was created. Existing successful conversation memory remains intact.
