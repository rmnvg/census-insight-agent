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

### Ranking row matching against the real corpus — fixed

- **Input:** The same live query above, run repeatedly against the running stack (real Qdrant collection, real Vertex Gemini) rather than synthetic fixtures, as part of final pre-submission verification.
- **Observed behavior:** Five distinct, sequential deterministic-hydration rejections before the request finally succeeded, none of which any hand-written test fixture had reproduced:
  1. `WRONG_REQUIRED_REGION` — `build_artifact_requirement` folded the classifier's document-scoping `regions: ["Madhya Pradesh"]` into the artifact requirement's target-region list even for a `rank_all` request, so every district row was rejected for not being the named state.
  2. `MISSING_AUTHORITATIVE_RESIDENCE_SCOPE` / year — the classifier legitimately left `residence_scope`/`year` unset for a query that never says "2011" or "total"; `_same_scope` unconditionally requires both.
  3. `UNSUPPORTED_OR_WRONG_TABLE_CELL` for every row in several split-table fragments — the 50-district Statement 6 table spans multiple ingested chunks, and only the *first* fragment repeats the full two-level header (year group + Total/Rural/Urban); later fragments repeat only the top-level year header, so residence could not be verified textually.
  4. The same error for the remaining rows even in fragments with a complete header — the model described the unit as `"females per 1000 males"`, a phrase that lives only in the table's title chunk, not the row-fragment chunks being matched.
  5. `count_table_entity_rows` (the guard from the case above) itself over-counted at 63, then 51, against a true 50 districts: `collect_metric_table_rows`'s "sex ratio" keyword filter also matched the distinct "Child Sex Ratio" and "Sex Ratio among Scheduled Castes/Tribes" statement tables, and separately treated `<br>`-tag-normalized "State/ District Code" (extra space from `_HTML.sub(" ", ...)`) as a 51st district.
- **Root cause:** Every one of these was a real gap, not a synthetic-fixture artifact: region-scoping conflated "which document to search" with "which rows are valid," several required fields have no sensible universal default, split-chunk headers are not fully repeated by ingestion, and a plain word-subset filter cannot distinguish a metric from its qualified variants.
- **Safety impact:** Each was a hard, correctness-preserving refusal (no wrong answer was ever returned) — this is the citation-safety design working as intended under real-world table fragmentation, not a near-miss.
- **Current mitigation:** `build_artifact_requirement` forces `regions=[]` for `rank_all` regardless of classifier output; `hydrate_artifact_dataset` defaults `population_scope`/`residence_scope`/`year` to the documented Census convention only when `rank_all` and the field is unset; `_residence_by_group_position` (`backend/app/execution/hydration.py`) falls back to the fixed Total/Rural/Urban column-position convention, verified consistently across every multi-residence table in this corpus, only when the textual sub-header is locally absent — it never changes what text is cited, only which column is considered; unit verification is skipped for `rank_all` rows since it cannot affect which cell is selected; `collect_metric_table_rows` excludes chunks whose context contains a disqualifying qualifier word (child/scheduled/caste/tribe) absent from the request's own metric, and its entity-label blocklist normalizes `<br>`-introduced spacing before matching. All six fixes are covered by regression tests in `backend/tests/test_agent_ranking.py`, several built from real text fetched from the live collection rather than hand-written fixtures.
- **Proposed future fix:** The residence group-position fallback encodes a real but corpus-specific convention (Total, Rural, Urban always in that fixed order); a future corpus with a different column order would need this made configurable rather than assumed. The qualifier-word list is a small, manually curated blocklist, not a general disambiguation mechanism — a metric named for a category not yet in that list would still over-match.

### Code-generation policy false rejections on a large dataset — fixed

- **Input:** The same live ranking request, once evidence selection and dataset hydration succeeded for the first time with a genuine 50-row dataset (existing artifact requests are typically 1-4 rows).
- **Observed behavior:** `generate_artifact_code` produced policy-violating programs twice in a row with different violations each time: first a DataFrame `.rename()` call and a non-literal `to_csv` path, then (after prompting against both) `to_csv(index=False)` called with no path argument at all, writing the returned CSV string via a separate `Path(...).write_text()` call. Per this codebase's design, code-policy violations never enter the one-repair loop, so each was an immediate refusal.
- **Root cause:** The code-generation prompt's existing "preserve dataset rows exactly" guidance did not anticipate a model reaching for pandas idioms (renaming columns for a nicer table, building an output path with `/`, or using `to_csv()`'s string-return form) that are only more likely to surface once there is real column/row volume to manipulate. Separately, `executor/policy.py`'s static resolver did not recognize `Path("output") / "table.csv"` as a literal path at all, which is a validator gap independent of ranking.
- **Safety impact:** None — every violation was correctly caught before reaching the executor; the impact was pure availability (a safe artifact request refused unnecessarily).
- **Current mitigation:** The code-generation prompt now explicitly forbids `.rename()`, forbids building output paths with `/`, and requires `to_csv` to take its literal output path as a direct argument. Separately, `_resolve_path` in `executor/policy.py` was extended to recognize `ast.BinOp` `/`-joins where both operands already resolve to static paths — the same strict recursive resolution already applied to every other path form, so this adds no new acceptance of dynamic or unsafe paths; it was verified locally against the actual model output before and after each fix, and covered by new regression tests in `backend/tests/test_executor.py`.
- **Proposed future fix:** These were the two idioms this specific run happened to hit; a different generated program could reach for another pandas/pathlib construction the policy resolver doesn't yet recognize. The policy's AST resolver is intentionally conservative (reject-by-default), so the residual risk is availability (unnecessary refusals), not safety.

### Inconsistency-analysis arithmetic chaining — mitigated, one gap remains

- **Input:** "Check whether the population totals in this table are internally consistent." against Odisha's decadal Rural/Urban/Total population series (1961–2011), run live against the real stack.
- **Observed behavior:** The tool-calling loop called the `calculate` tool with `operation="difference"` directly on the two raw component values (e.g. `difference(164.4, 11.1)` for 1961) instead of first summing the components and then comparing that sum to the stated total. Synthesis then wrote draft claims narrating a `sum` followed by a `difference`-of-sums derivation that no tool call had actually produced, and separately, checking all six census decades exhausted the 6-call tool budget (`list_skills`, `read_skill`, `search_documents`, and 3 `calculate` calls left no budget for the remaining years or the coverage calls the model also attempted).
- **Root cause:** Two independent gaps: (1) `skills/inconsistency_analysis.md` told the model to "use only the safe arithmetic helper" without specifying the two-step sum-then-compare sequence a "do the parts add up to the total" check requires, so the model improvised a single, semantically meaningless `difference` call and then free-formed a narrative in `synthesize` that didn't match any real `CalculationResult`; (2) `agent_max_tool_calls` (6) was sized for simpler single- or few-target requests, not a full multi-decade sweep.
- **Safety impact:** None — `validate_and_materialize_citations` (`backend/app/agent/citations.py`) correctly detected that the claimed `derivation` (operation, operands, result) did not match any validated `CalculationResult` and rejected the whole response (`INVALID_DERIVATION`) rather than presenting fabricated arithmetic. This is the citation-safety design working as intended: a wrong-but-plausible consistency verdict was refused, not shown.
- **Current mitigation:** `agent_max_tool_calls` raised from 6 to 12 so a full six-decade sweep no longer exhausts the budget partway through (confirmed live: all `calculate` and `get_document_coverage` calls now return `ok` instead of `limit_exceeded`). `skills/inconsistency_analysis.md` now explicitly documents the required `sum`-then-`difference` sequence and states that the final derived claim must match one real tool call exactly, with no invented narrative step.
- **Proposed future fix:** The skill-doc wording change did not, on its own, change the model's tool-calling choice in repeated live retests — it kept calling `difference` directly on the two components. A more reliable fix would be a dedicated `check_sum_consistency(total, *components)` tool that performs both steps deterministically in one call, removing the burden of correct multi-step chaining from the model entirely (the same philosophy already applied to ranking's winner computation, which is deterministic application code rather than model-trusted).

### Ambiguous metric phrasing across multiple same-topic tables — known limitation, not fixed

- **Input:** "Show me a chart comparing population growth across Karnataka and Odisha," run live.
- **Observed behavior:** `prepare_artifact` failed with `UNSUPPORTED_OR_WRONG_TABLE_CELL`: the proposed rows' values could not be matched to any cell under the literal metric text "Population Growth."
- **Root cause:** Each state report has at least two distinct tables that could plausibly answer "population growth" — a Statement 1 "Population and decadal change by residence: 2011" table (2001→2011 percentage change) and a separate "Population decadal growth rate: 1951–2011" long time-series table — and "population growth" matches the literal wording of neither table title. `_METRIC_ALIASES` (`backend/app/agent/scopes.py`) only maps known singular/plural/qualifier variants of a single metric (e.g. "literacy rates" → "literacy rate"); it has no entry for this phrase, and rightly so — a phrase that is ambiguous between two structurally different tables (one snapshot decade, one six-decade series) cannot be safely aliased to either without risking citing the wrong one.
- **Safety impact:** None — hydration correctly refused rather than guessing which table's numbers to show or fabricating a value under a metric string absent from both tables.
- **Current mitigation:** None. This is a refusal, not a wrong answer, so it is safe as-is.
- **Proposed future fix:** When a metric phrase's evidence spans candidate rows from more than one differently-titled table for the same requested entity, treat it as a clarification case ("Do you mean the 2001–2011 decadal change, or the longer 1951–2011 growth trend?") instead of proceeding with the first table the model happened to select. This is a genuine free-text-to-table-title matching problem, not something a small alias table can safely generalize to.

### Uncaught provider error outside `prepare_artifact` — known limitation, not fixed

- **Input:** Any request during a transient Vertex `MODEL_UNAVAILABLE`/rate-limit condition, observed live during repeated ranking-feature verification.
- **Observed behavior:** `resolve_query` calls `self.model.resolve(...)` with no local error handling. A transient provider failure there raised a raw `ProviderOperationalError` that propagated uncaught through `AgentService.chat` and FastAPI's default handler, returning a bare "Internal Server Error" HTTP 500 with no JSON body, trace ID, or `retryable` signal — unlike every other operational failure in this system, which returns a sanitized typed `AgentErrorResponse`.
- **Root cause:** Only `prepare_artifact` (and, narrowly, `assess_evidence` for `ProviderCallTimeout`) wraps its model calls to convert provider exceptions into `AgentOperationalError`. `classify_task`, `resolve_query`, `synthesize`, and `repair` call the model directly.
- **Safety impact:** None to citation safety — no claim was fabricated. The impact is purely operational: the evaluator sees a generic, unhelpful 500 instead of a typed, retryable error with a trace ID, for a condition that is otherwise handled gracefully everywhere else in the graph.
- **Current mitigation:** None. This was discovered during this session's live verification and is out of scope for the ranking feature it was found while testing; retrying the request (a fresh `/chat` call, not a retry of the failed one) succeeds normally once the transient condition clears.
- **Proposed future fix:** Apply the same `ProviderOperationalError`/`ProviderCallTimeout` → `AgentOperationalError` conversion already used in `prepare_artifact` to every graph node that calls `self.model`, or push it into a single wrapping layer (e.g. `checkpoint_node`) so no node can leak a raw provider exception.

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
