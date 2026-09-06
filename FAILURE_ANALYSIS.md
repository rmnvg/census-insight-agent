# Failure Analysis

## Case summaries

### Unsafe chart/map OCR — mitigated

- **Input:** Twelve image-only Karnataka chart and thematic-map pages.
- **Observed behavior:** Tesseract returned words and high token confidence but detached labels, legends, regions, and values.
- **Root cause:** Plain OCR discards two-dimensional visual relationships; token confidence measures recognition, not semantic association.
- **Safety impact:** Ingestion could cite a real page while asserting the wrong district/value or chart-category/percentage pair.
- **Current mitigation:** Raw OCR is quarantined, all pages are `excluded_unverified_visual`, and independent chunk/embed/upload guards reject it.
- **Proposed future fix:** Human-approved checksum-bound transcription, or layout-aware multimodal extraction followed by human verification.

### Global comparison budget dropped Odisha evidence — fixed

- **Input:** A Karnataka/Odisha literacy chart request.
- **Observed behavior:** Valid Odisha statewide evidence ranked tenth but was omitted from a globally packed 12,000-character assessment context.
- **Root cause:** Earlier packing optimized rank globally and did not reserve capacity per requested region.
- **Safety impact:** The system safely refused, but answerable comparisons and artifacts degraded asymmetrically.
- **Current mitigation:** Target-filtered retrieval and balanced packing reserve one complete value-bearing candidate per region and fail explicitly if reservations cannot fit.
- **Proposed future fix:** Evaluate learned per-target reranking only after preserving the deterministic minimum-evidence guarantee.

### Streamlit rerun loop hid a completed answer — fixed

- **Input:** A normal chat submission from the Streamlit UI.
- **Observed behavior:** FastAPI completed successfully, while the UI remained pending and repeatedly triggered document/Qdrant log traffic.
- **Root cause:** An unconditional post-submit rerun collided with the in-progress guard; sidebar coverage was recomputed on every rerun.
- **Safety impact:** No answer was fabricated or duplicated, but the successful response was unavailable to the user and operational noise obscured diagnosis.
- **Current mitigation:** Completion renders in the same run, duplicate guarded events return normally, and sidebar metadata is session-cached and manifest-backed.
- **Proposed future fix:** Add streamed server progress and URL-restorable sessions without retrying ambiguous paid POST requests.

### Ranking over an incomplete row proposal — mitigated

- **Input:** A superlative question over an unbounded row set, e.g. "Which district had the highest sex ratio in Madhya Pradesh?", against a ~50-row Statement table.
- **Observed risk:** The application computes the winning row deterministically from the model's proposed dataset rows, not from generated code. If the proposal silently omitted rows (e.g. 12 of 50 districts, all lower than the true maximum), the deterministic computation would still run and return a confident, citation-backed, but factually wrong "highest" answer — a plausible-looking wrong answer is worse than a refusal.
- **Root cause:** Nothing previously verified that a model's row proposal was complete; `ArtifactDatasetProposal.rows` has no required cardinality relative to the source table.
- **Safety impact:** Would have been the same failure class as the OCR case above — citing a real page while asserting an unverified fact — but for computed rankings instead of extracted values.
- **Current mitigation:** `count_table_entity_rows` (`backend/app/execution/hydration.py`) independently counts distinct entity labels in the trusted evidence, excluding repeated headers and the state/UT aggregate row, and `prepare_artifact` refuses (`MODEL_OUTPUT_INVALID` / `RANKING_COVERAGE_INCOMPLETE`) when the hydrated dataset covers fewer rows than detected. A second, narrower guard (`RANKING_INCLUDES_AGGREGATE_ROW`) rejects the case where the winning row is the state total rather than an individual entity, since the aggregate row shares the same table and could otherwise win a "lowest" query. Both are covered by regression tests in `backend/tests/test_agent_ranking.py`.
- **Proposed future fix:** The row-completeness check is a heuristic lower bound on distinct labels, not an exact table parser; a genuinely adversarial or malformed table could still evade it. A stronger fix would independently extract the full row set deterministically (not via the model at all) before ever calling the model for a proposal, removing the completeness question rather than detecting it after the fact.

## Streamlit boundary failures

The first live UI request completed successfully in the backend but remained visually pending. A
rerun that encountered the active submission guard still called `st.rerun()` unconditionally,
creating a rapid rerun loop that could interrupt the run responsible for committing and displaying
the response. The same reruns rebuilt the sidebar and repeatedly called document coverage routes;
those routes each scrolled all 2,058 Qdrant payloads. This caused the visible Qdrant log flood.

Submission now renders its result in the completing run and never forces a post-submit rerun.
Guarded duplicate events return normally. Sidebar health/document/coverage data is loaded once per
UI session and refreshed only by an explicit button. Public document metadata comes from the
checksum-bound generated manifests, while Qdrant readiness remains a separate health check, so
coverage display no longer scans vector payloads. Citation technical details also no longer use an
unsupported expander nested inside the citation expander.

If the UI read timeout expires, the backend may still complete the request. The client deliberately
does not retry `POST /chat`, because a retry could create a second paid model run or artifact. The UI
shows a sanitized operational error; the evaluator can inspect backend traces using any returned
trace ID and submit a new turn only after deciding whether the original completed.

A valid answer can outlive a trace or artifact presentation failure. Temporary trace unavailability
leaves the answer and citations visible. An artifact download with a wrong MIME type, excessive size,
invalid PNG, malformed CSV/manifest, or backend error is rejected locally and shown as “Artifact
could not be displayed”; the chat request is never repeated and the UI never claims that artifact
was validated.

Streamlit browser state is not durable conversation evidence. A refresh may lose displayed turns or
create a new backend session until URL/session restoration exists, but it cannot corrupt or delete
the prior LangGraph checkpoint. The new-conversation action intentionally clears only the current UI
display and allocates a distinct backend session.

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
