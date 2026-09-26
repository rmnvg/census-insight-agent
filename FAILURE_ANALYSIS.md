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
- **2026-09-26:** that fix now applies whenever a structured table store resolves the column: the
  dataset is every row of the table, and the heuristic count is used only on the fallback path.

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

### Uncaught provider error outside `prepare_artifact` — fixed

- **Input:** Any request during a transient Vertex `MODEL_UNAVAILABLE`/rate-limit condition, observed live during repeated ranking-feature verification.
- **Observed behavior:** `resolve_query` calls `self.model.resolve(...)` with no local error handling. A transient provider failure there raised a raw `ProviderOperationalError` that propagated uncaught through `AgentService.chat` and FastAPI's default handler, returning a bare "Internal Server Error" HTTP 500 with no JSON body, trace ID, or `retryable` signal — unlike every other operational failure in this system, which returns a sanitized typed `AgentErrorResponse`.
- **Root cause:** Only `prepare_artifact` (and, narrowly, `assess_evidence` for `ProviderCallTimeout`) wrapped its model calls to convert provider exceptions into `AgentOperationalError`. `classify_task`, `resolve_query`, `synthesize`, and `repair` called the model directly.
- **Safety impact:** None to citation safety — no claim was fabricated. The impact was purely operational: the evaluator saw a generic, unhelpful 500 instead of a typed, retryable error with a trace ID, for a condition that was otherwise handled gracefully everywhere else in the graph.
- **Current mitigation:** `checkpoint_node` (`backend/app/agent/checkpoint.py`), which already wraps every graph node for JSON-safe checkpointing, now also catches `ProviderCallTimeout`/`ProviderOperationalError` generically and converts either into the same sanitized `AgentOperationalError` `prepare_artifact` already produced for its own calls — the single wrapping layer this entry's proposed fix called for. `classify_task`, `resolve_query`, `synthesize`, and `repair` no longer need (and don't have) any node-local conversion of their own. One gap remained even after that: `load_memory`'s `summarize_memory` call used a bare `asyncio.wait_for(self.model.ainvoke(...), ...)` with no `map_provider_error` conversion at all, so a non-timeout provider failure there (rate limit, schema rejection, auth) raised a raw exception that was neither `ProviderCallTimeout` nor `ProviderOperationalError` — still uncaught by `checkpoint_node`'s specific `except` clauses. `summarize_memory` (`backend/app/agent/provider.py`) now converts its own exceptions the same way `_structured` does, closing that last case.
- **Proposed future fix:** None remaining for this failure class.

### Comparison follow-up lost its own framing during rewrite — fixed

- **Input:** "How does that compare with Odisha?", a follow-up to a Karnataka literacy lookup, run live 2026-09-23 against the real stack (trace `ae431530-1ea5-4560-ab90-94af6b1f782a`) as part of re-verifying this same class of comparison after the reliability fixes above.
- **Observed behavior:** `resolve_query` rewrote the follow-up to "What was the literacy rate in Karnataka and Odisha in 2011?" — both target names present, evidence assessment found direct-answer evidence for both, but `extract_calculations` returned no calculation (nothing in that query text asked for a computed difference), and `validate_citations`'s response-invariant check then refused the whole turn with `DERIVED_CLAIM_MISSING_INPUT_CITATIONS`, since it unconditionally requires a derived claim whenever `task_type == "comparison"`. This is the exact failure the demo video (`docs/demo/`) shows and narrates.
- **Root cause:** `resolve_query`'s existing "target guard" (see the reliability fixes above) only re-injects comparison framing when a target *name* is missing from the rewritten query. A rewrite can name every target while still losing the comparative framing itself, and `missing_targets` being empty meant the guard never fired — a resolved query can satisfy "every target is named" while failing "still reads as a request to compute a difference," and only the first of those two conditions was ever checked.
- **Safety impact:** None — no wrong number was ever asserted; the citation-safety design refused correctly given what it was handed. The impact was pure availability: a fully answerable comparison refused.
- **Current mitigation:** `resolve_query` now also appends the same "Compare the same metric..." framing sentence whenever `task_type == "comparison"` and target names are present, not only when a target went missing (`comparison_framing_needed`, distinct from and independent of `target_guard_applied`, both recorded in the trace). Verified live after the fix: the same two-turn sequence now returns "Karnataka is higher than Odisha by 2.46 percentage points" with a correctly derived, dual-cited claim. Regression test: `test_comparison_resolution_restores_framing_when_targets_are_present_but_framing_is_lost` (`backend/tests/test_agent_service.py`).
- **Proposed future fix:** None remaining; this closes the specific gap the demo video documents.

### `matplotlib.style.use()` false-flagged as the dangerous backend-switch call — fixed

- **Input:** A live chart-generation request, encountered while hardening `executor/policy.py`'s AST allowlist (adding a check for `matplotlib.use(...)`, a dynamic-import primitive independent of the `import`-statement allowlist) and then re-verifying the full live harness on 2026-09-23.
- **Observed behavior:** A real generated chart program calling `matplotlib.style.use("...")` — a harmless style-sheet call the code-generation prompt's own "deterministic style" instruction makes a plausible, even likely, thing to generate — was rejected with `Forbidden attribute call: use`, the same error meant for the actually-dangerous `matplotlib.use(backend)` call, which dynamically imports an arbitrary `"module://..."` name.
- **Root cause:** The new check matched the bare attribute name `use` anywhere, not the specific call it was meant to catch; `matplotlib.style.use` and `matplotlib.use` share a method name but are unrelated functions on different objects.
- **Safety impact:** None — this was a false positive in a defense-in-depth AST check being added, not a regression in shipped behavior; the impact was pure availability (an otherwise-safe artifact request refused).
- **Current mitigation:** The check is now receiver-aware, using the same import-alias tracking already added for disambiguating `np.load` from `json.load`: it fires only for a call on the bare module bound by `import matplotlib` (or an alias of it), not on any attribute chain reached through it. Verified live after the fix: the same chart and table harness cases both now succeed. Regression test: `test_ast_allows_matplotlib_style_use_but_rejects_backend_use` (`backend/tests/test_executor.py`).
- **Proposed future fix:** None remaining.

### Summary claims split an aggregate into an unsupported rural/urban or male/female breakdown — mitigated

- **Input:** "Summarize the key population findings for Odisha," run live 2026-09-23 as part of the same harness re-verification.
- **Observed behavior:** Consistently across two separate live runs, several claims asserted a rural/urban or male/female breakdown of a total (e.g. "Of the total Scheduled Tribe population, 8,994,967 were in rural areas") whose own number could not be found verbatim in the cited evidence chunk (`rejection_reason_code: VALUE_ABSENT`), while the aggregate figure itself was correctly supported. One bounded repair improved but did not fully resolve it either time; the whole turn refused.
- **Root cause:** The retrieved evidence chunk for that table row states only the aggregate; the summary-synthesis prompt had guidance against asserting a *computed* difference not stated verbatim, but no equivalent guidance against inventing a plausible-looking breakdown *structure* (rural/urban, male/female) for a value that is only ever given as one number.
- **Safety impact:** None — `validate_citations` correctly refused claims whose asserted numbers were not textually supported, exactly the citation-safety design working as intended; the impact was availability (a good-faith summary refused over 2-3 unsupported claims out of ~19).
- **Current mitigation:** Both the `synthesize` and `repair` prompts (`backend/app/agent/provider.py`) now explicitly forbid splitting an aggregate into a rural/urban/gender breakdown unless each part's own number is itself an explicit value in the cited evidence, not just the aggregate. Verified live after the fix: the same summary request succeeded twice in a row (11/11 and 10/10 claims cited) in subsequent harness runs.
- **Proposed future fix:** None remaining for this specific pattern; a model could still invent a different kind of unsupported structure not covered by this specific instruction, in which case citation validation remains the backstop that prevents a wrong answer from being shown.

### Uploaded region treated as an unresolved referent — fixed

- **Input:** "What was the literacy rate of Northvale in 2011?", asked live on 2026-09-24 right after uploading a text-layer test report for the fictional region "Northvale" through the new document-upload endpoint.
- **Observed behavior:** Upload, indexing, and the document catalog all succeeded, but the agent replied with a clarification ("Could you please provide a valid region name or document ID for 'Northvale'?") instead of answering. The trace showed `task_classification` → `clarification` with no retrieval at all.
- **Root cause:** The classifier and resolver prompts described the corpus only as "supplied Indian Census reports" and never listed the supplied documents, so region validity came entirely from the model's world knowledge. The three bundled states worked only because the model already knew they were Indian states.
- **Safety impact:** None; the agent asked rather than guessed. The impact was availability for any uploaded report whose region the model does not recognise.
- **Current mitigation:** `GeminiAgentModel` now appends the live document catalog (IDs, titles, and exact region names only, never document text) to the classify and resolve requests, and tells both steps that listed regions are in scope. Verified live after the fix: the same question answered with a citation to page 2 of the uploaded PDF. An A/B rerun with the catalog disabled showed that the chart and comparison outcomes below do not depend on it. Regression tests: `backend/tests/test_provider_catalog.py`.
- **Proposed future fix:** None for this pattern.

### Generated chart code using `if __name__ == "__main__":` refused by code policy — mitigated

- **Input:** "Create a bar chart comparing the literacy rates of Karnataka and Odisha.", run live repeatedly on 2026-09-24.
- **Observed behavior:** The same request produced a chart on some runs and a refusal on others. Classification, evidence, and hydration were identical down to the chunk IDs; failing runs stopped at `code_policy_validation` with `Dunder name is forbidden: __name__`. Later runs also showed `Forbidden attribute call: rename` and plain runtime crashes on large ranking tables.
- **Root cause:** The model sometimes wraps its program in the `__main__` idiom, which the AST policy's dunder ban rejects. Policy violations deliberately never enter the repair loop, so the turn refused.
- **Safety impact:** None; the policy failed closed.
- **Current mitigation:** The policy and the no-repair rule are unchanged. Feeding policy errors back would let a hostile program iterate toward an evasion, and uploads now put untrusted document text in front of the model. Prevention happens upstream instead: the generation prompt forbids double-underscore names and `main()` guards, and `skills/table.md` and `skills/chart.md` now carry a known-good reference program. Both were verified to pass the policy, execute, and pass lineage validation on three real artifact inputs, and `test_skill_reference_programs_pass_the_executor_code_policy` keeps them from drifting. Verified live: the two-region chart went from 1 of 2 to 2 of 2, and rankings stopped failing at code generation.
- **Proposed future fix:** None required. An exact allowlist for the read-only `__name__ == "__main__"` comparison would be a policy change and is not needed while prevention holds.

### Three-region chart refused although all three values were found — fixed

- **Input:** "Create a bar chart comparing the literacy rates of Karnataka, Odisha and Madhya Pradesh.", refused on all five live runs on 2026-09-24.
- **Observed behavior:** The assessor's explanation claimed all three states had direct answers, but the verdict was insufficient.
- **Root cause:** The explanation was wrong, not the verdict. The per-target trace showed Madhya Pradesh had no direct-answer chunk. Its Statement 19 table is split into eight fragments that repeat one header; only fragment `1f82157d` holds the state row (69.3), and it was not in the top 20 results. Lookups and comparisons already fetched sibling fragments of the best pages; charts and tables did not.
- **Safety impact:** None; the deterministic per-target check correctly refused.
- **Current mitigation:** Named-target artifacts now get the same sibling-page expansion (`call_tools`), and packing reserves the state-row fragment first. Verified live: 3 of 3 successful runs, plotted values Odisha 72.9, Madhya Pradesh 69.3, Karnataka 75.36, all matching the source. Regression test: `test_named_artifact_expands_pages_so_split_table_state_row_is_reserved`, which fails without the fix.
- **Proposed future fix:** None remaining.

### Scheduled Tribes sex ratio reported as the state's sex ratio — fixed

- **Input:** "What was the sex ratio of Karnataka in 2011?", plus a Karnataka vs Odisha sex-ratio chart, run live 2026-09-24.
- **Observed behavior:** One answer said "the total sex ratio in Karnataka was 990", with a citation whose cell really does read 990. The chart proposal also used 990. Karnataka's sex ratio is 973.
- **Root cause:** In the source Markdown the table title ("Sex Ratio … among Scheduled Tribes by residence") is a plain paragraph under `### Statement 17`, not a heading. The chunker builds breadcrumbs from headings, so every fragment of that table said only "Statement 17", and nothing in the chunk mentioned Scheduled Tribes. Odisha's Statement 13 (Scheduled Castes, 987 against the true 979) has the same shape.
- **Safety impact:** Real. A subgroup value could be presented as a whole-population figure with a valid-looking citation.
- **Current mitigation:** The title paragraph survives as its own small chunk with the identical breadcrumb, so `AgentTools` re-attaches it to each table fragment's `section_path` at runtime (text and offsets untouched, no re-ingestion). Three deterministic guards then use it: citation validation rejects a whole-population claim quoting a Scheduled Caste, Scheduled Tribe, or child table (`SUBGROUP_TABLE_FOR_WHOLE_POPULATION_CLAIM`); hydration rejects such cells (`SUBGROUP_TABLE`); and ranking scans exclude such tables. The model prompts also show a `SECTION=` line. Verified live: the lookup now answers 973/979/963, and the chart binds 973. Regression tests: `backend/tests/test_subgroup_tables.py`, on the real chunks.
- **Proposed future fix:** Fold statement title paragraphs into chunk breadcrumbs at ingestion. That re-embeds the corpus and changes the 2,058-point baseline, so it is deferred for an explicit decision.

### Sex-ratio charts could never hydrate — fixed

- **Input:** Every "Create a bar chart comparing the sex ratio of …" request on 2026-09-24.
- **Root cause:** An unspecified population defaults to "persons", and hydration requires that word near the cell. Sex-ratio tables have no Persons/Male/Female column, because the metric already relates the two sexes. Rankings were exempt for exactly this reason; charts and tables were not.
- **Current mitigation:** The same exemption now applies to any sex-ratio request with an unspecified population. Subgroup tables are excluded separately by title. Verified live: 2 of 2 charts.

### Correct comparison refused and count difference labelled "percent" — fixed

- **Input:** "Compare the sex ratio of Odisha and Madhya Pradesh.", about 1 in 3 live runs succeeding.
- **Root cause:** The model cited an extra prose chunk that does not quote 931, and one unsupported citation invalidated the whole correct claim. The calculation also cited a second Odisha chunk no claim quoted, which failed a chunk-ID subset match. Separately, the app's own derived-claim renderer labelled every non-rate difference "percent" ("higher by 48 percent" for a 48-point gap).
- **Current mitigation:** Citations that do not quote the claim's value are pruned when another citation verifies it; a claim with no supporting quote still fails, and integrity failures (unknown IDs, excluded pages, non-verbatim snippets) stay hard errors. Derived claims cite exactly their validated inputs and are matched to their calculation by operation and operands. Count differences carry no unit word. Pruned citations are recorded in the trace. Verified live: every completed run valid on the first pass, reading "Odisha is higher than Madhya Pradesh by 48." Regression tests use the real chunks.

### Units stated only in a table title — fixed

- **Input:** "What was the sex ratio in Madhya Pradesh in 2011?", declined live.
- **Root cause:** The assessor marked the right row `has_unit=false` because the cell is a bare 931. The unit appears only in the title, and this fragment's breadcrumb is mis-attributed ("Statement 5: Proportion of rural and urban population").
- **Current mitigation:** The prompt now says a unit in a title or column heading counts. A deterministic backstop treats a stated unit phrase, or a metric whose Census unit is fixed by definition (sex ratio, literacy rate, work participation rate; never raw counts), as a known unit. Verified live: 2 of 2.

### District rankings refusing after the model's output shifted — fixed

- **Input:** "Which district of <state> had the highest/lowest sex ratio?", which had succeeded earlier, refused for all three states on 2026-09-24.
- **Root causes (one per state, found with a hydration spy in the running container):**
  - **Madhya Pradesh:** the model began labelling rows with series "Total", which later split-table fragments identify only by column position.
  - **Karnataka:** the model mis-copied one of 30 chunk UUIDs.
  - **Odisha:** the completeness guard counted 68 "districts" for 30. It was reading table-of-contents lines and a "Decadal Change" graph table that spells districts differently.
- **Current mitigation:**
  - A residence-named series is accepted when the column-position fallback confirms the column.
  - For rankings only, an unknown chunk ID rebinds to the single chunk containing a cell that passes every check; an ambiguous or missing match still fails.
  - The completeness count ignores contents entries and counts only tables sharing the proposal's header layout.
  - Separately, a "Total" population scope now means unspecified.
- **Verified live:** Madhya Pradesh Balaghat 1,021, Karnataka Udupi 1,094, Odisha lowest Nayagarh 915 and highest Rayagada 1,051, each matching the source table.

### Artifact repair path hit the graph step limit — fixed

- **Root cause:** The longest legitimate path (artifact, one repair, refusal) takes 16 supersteps, exactly `AGENT_MAX_STEPS`. It escaped as an untyped `GraphRecursionError` (seen as a research section crash).
- **Current mitigation:** The limit is 24 in Settings, `docker-compose.yml`, and `.env.example`. Any recursion-limit hit is now a typed, retryable `AGENT_STEP_LIMIT`, and a research section never sinks its brief.

### District ranking by literacy rate — fixed

- **Input:** "Which district of Karnataka had the highest literacy rate?"
- **Observed behavior:** Hydration refused (`UNSUPPORTED_OR_WRONG_TABLE_CELL`). The benchmark case
  `od-top-literacy` (Khordha 86.9) ended in `MODEL_OUTPUT_INVALID` on both runs.
- **Root cause:** Literacy tables put 2001 and 2011, each with Total/Rural/Urban, under one
  "Literacy Rate" header. The chunker keeps only the first header row on later fragments, so a
  value in those fragments could not be tied to its year and residence.
- **Current mitigation:** Rankings read from structured tables parsed from the whole Markdown table
  (DESIGN.md, "Structured tables"). The dataset holds every district row with no model proposal,
  and each row is cited from the chunk that contains it. Deep Research may now plan literacy
  rankings.
- **Verified live:** Karnataka Dakshina Kannada 88.57; Odisha Khordha 86.9 (lowest Nabarangapur
  46.4); Madhya Pradesh lowest Alirajpur 36.1; female, Bhopal 74.9; rural Karnataka, Dakshina
  Kannada 85.33. Each matches its source table. The four benchmark ranking cases, run twice,
  passed 8 of 8 (previously 5 of 8), at a median of 5.8 seconds (previously 15 to 29).
- **Found live:** the first answer for a female-literacy question read "the highest Literacy Rate
  (74.9 percent) among Literacy Rate by district in Madhya Pradesh", which did not say "female".
  The dataset title now names any non-default population or residence.

### Query resolution turned a ranking into a lookup — fixed

- **Input:** A Deep Research section asking "Which district of Odisha had the highest literacy
  rate?"
- **Observed behavior:** `MODEL_OUTPUT_INVALID`, although the same question answered correctly on
  its own.
- **Root cause:** The classifier set `rank_all`, but the query resolver's rewrite ("What was the
  literacy rate of Odisha in 2011?") came back as a one-region comparison. The merge kept the
  classifier's regions and scopes but let the resolver drop the ranking.
- **Current mitigation:** A ranking the classifier found survives resolution. The regression test
  replays the resolver output from the live trace. On a rerun, all three sections of the brief
  answered.

### Research sections failed with `database is locked` — fixed

- **Input:** A research brief, or any turns that start together against a fresh SQLite state file.
- **Observed behavior:** One section intermittently ended as "could not be completed". The
  research tests failed in 5 of 25 runs.
- **Root cause:** Every turn opened its own checkpoint connection, and each new connection ran
  LangGraph's setup script (a WAL journal-mode switch plus table DDL) on first use. Concurrent
  first turns collided on the same file.
- **Current mitigation:** `SqliteStateBackend` runs the checkpoint and session schema setup once,
  under a process lock, and marks each per-turn connection as already set up. A regression test
  starts eight first turns together on a fresh database. It failed in 6 of 12 runs before the
  fix and passes in 20 of 20 after; the research tests pass in 25 of 25.

### Per-residence comparisons refused although the draft was correct — fixed

- **Input:** "How does that compare with Madhya Pradesh?" after Odisha's sex ratio, which drafts
  total, rural, and urban comparisons.
- **Observed behavior:** A correct draft (differences 48, 53, 14) was rejected with
  `INVALID_DERIVATION`, and the repair dropped a target, so the answer was refused.
- **Root cause:** Derived claims were paired with the first app-computed calculation whose
  evidence they cited. All three calculations cite the same two table chunks, so the rural and
  urban claims were checked against the total calculation.
- **Current mitigation:** A derived claim is paired with the calculation that has exactly its
  operation and operands, falling back to evidence only when it states none. Derived sentences now
  name the residence ("In rural areas, …") so the three differences are distinguishable. The
  benchmark case `followup-compare` covers it.

### Trust scorer passed misattributed answers — fixed

- **Input:** Real recorded answers with one deliberate corruption each.
- **Observed behavior:** The scorer passed swapped region labels, a rural value stated as the
  total, a 2011 value labelled 2001, and a value moved to another region's row.
- **Root cause:** Expected numbers were matched anywhere in the answer, labels anywhere in the
  text, and grounding meant the number appeared anywhere in a table quote holding dozens of
  numbers.
- **Current mitigation:** Fact tuples matched against each claim's own fields, contradictions
  counted as wrong answers, and provenance checked by locating the value on its region's row and
  column in the cited quote. See DESIGN.md, "Trust scorecard".

### Questions about 2001 declined — known limitation

- **Input:** "What was the sex ratio of Karnataka in 2001?" (and Odisha).
- **Observed behavior:** The agent declines, or asks whether 2011 is meant instead.
- **Root cause:** Task classification treats the corpus as 2011-only, although its tables carry
  2001 columns beside the 2011 ones.
- **Safety impact:** None; it is the safe failure, and the benchmark counts it as an unnecessary
  refusal rather than a wrong answer.
- **Proposed future fix:** Scope questions by the years the tables actually contain, then rely on
  the column provenance checks to keep 2001 and 2011 values apart.

### Failed-upload cleanup could leave evidence without its source PDF — fixed

- **Input:** An upload whose indexing fails after its points were upserted, while Qdrant is
  unreachable.
- **Observed behavior:** The error from removing the points was suppressed, and the source PDF was
  deleted anyway. The points stayed retrievable, citing a PDF that no longer existed. A late
  failure also left the generated manifest behind, so the failed document still appeared in the
  library, and uploading the same PDF again was rejected as a duplicate.
- **Root cause:** Cleanup deleted the source first and treated point removal as best-effort.
- **Current mitigation:** A failed document is first withdrawn with a durable marker. Retrieval and
  the scroll-based collectors exclude it from then on. Its points are removed next. Only then are
  its PDF, manifests, and marker deleted. An unfinished withdrawal is retried at startup and before
  each job. A re-upload of the same PDF retries it first and returns 503 while Qdrant is still
  down.
- **Found live:** Against a real Qdrant 1.15 server, a search naming the withdrawn document still
  returned it: the server ignores `must_not` when `must` matches the same `document_id`. Local-mode
  Qdrant does not reproduce this, so the unit test passed. Retrieval now removes withdrawn IDs from
  the requested set instead, and `backend/tests/test_qdrant_server.py` runs against a real server
  in `make integration-test` and CI.

### Executor restart stranded in-flight jobs — fixed

- **Input:** The executor container is killed (OOM, restart) while running a job.
- **Observed behavior:** The claimed request stayed in `processing/` forever with its partial output
  under `jobs/`. The backend learned nothing until its result deadline passed.
- **Root cause:** The worker only consumed `inbox/`; nothing reconciled claims from a previous
  process.
- **Current mitigation:** At startup the worker fails every claimed job with
  `EXECUTION_INTERRUPTED` and deletes its unvalidated output. It does not re-run the job: the job may
  be what killed the worker, and re-running it would crash-loop the only consumer. Verified live by
  killing the real executor container mid-job. After a restart, the waiting backend received the
  failure within seconds, and the next job succeeded.

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
- **Executor restarted mid-job:** the restarted worker reports the job as `EXECUTION_INTERRUPTED`
  and removes its partial output. It is not repaired or re-run.
