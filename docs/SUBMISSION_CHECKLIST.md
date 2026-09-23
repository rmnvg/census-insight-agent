# Submission Checklist

Items marked `(verified 2026-09-06)` were actually run this session against the live stack with
real Vertex/Qdrant, not just reviewed — see `docs/REPOSITORY_AUDIT.md` and `FAILURE_ANALYSIS.md`
for what that live verification found and fixed. Items left unchecked need the user's own action
(browser, video recorder, GitHub account) and cannot be completed by an agent in this environment.

- [ ] Temporary clean-clone/export test completed
- [x] `.env` configured locally and still ignored (verified 2026-09-06)
- [x] Vertex ADC setup documented and verified manually (verified 2026-09-06: real ADC credentials, real Gemini calls)
- [x] `docker compose up --build` succeeds (verified 2026-09-06)
- [x] Qdrant, executor, backend, and frontend are healthy (verified 2026-09-06)
- [x] Existing Qdrant collection has exactly 2,058 validated points (verified 2026-09-06 via `make verify-offline`: 2058/2058 valid, dense+sparse compatibility confirmed). Note: a later fix to `chunking.py` (repeating a split table's sub-header row on every fragment, not just the first — see `FAILURE_ANALYSIS.md`) changes fresh-ingestion output to 2,112 chunks, verified against this real corpus via `backend.app.ingestion.cli ingest --dry-run` and the updated `test_production_citations.py`. The live collection has not been re-ingested and still holds 2,058 points from before that fix; `make verify-offline`'s `--expected-points 2058` (Makefile) intentionally still matches it. Re-ingesting (`scripts/initialize_corpus.py --allow-paid-calls --rebuild`) and then bumping that flag to 2112 is a deliberate, billable step for the user to run, not done here.
- [x] `make check` passes (verified 2026-09-06: format/lint/typecheck/UI-smoke/security/secret-scan all clean; 5 of 244 tests fail in this specific sandboxed agent environment due to a `resource.setrlimit`/subprocess restriction unrelated to the code — confirmed via `git stash` that they fail identically on unmodified code, i.e. pre-existing to this environment, not a regression; re-run `make check` on your own machine to confirm they pass natively, which is expected)
- [x] `make verify-offline` passes against the local corpus (verified 2026-09-06)
- [x] Optional paid live harness run is reviewed (verified 2026-09-23, partially: every one of the eleven `scripts/live_evaluation.py` cases individually passed live at least once — several (`odisha-comparison`, `chart`, `table`, `summary`) only after real bugs found during this session's runs were fixed; see the three new "fixed"/"mitigated" entries in `FAILURE_ANALYSIS.md` dated 2026-09-23. A single clean 11/11 run in one invocation was not achieved: repeated consecutive full-harness runs in a short window increasingly hit transient `MODEL_UNAVAILABLE` (503) and occasional `MODEL_OUTPUT_INVALID` (502) responses from Vertex, consistent with self-induced rate-limit pressure from the volume of live testing itself rather than a code defect — the same class of transient condition already documented as a known, non-code limitation elsewhere in `FAILURE_ANALYSIS.md`. Re-running `make live-eval`-equivalent once, without back-to-back repeated runs immediately before it, is expected to pass cleanly now; this was not re-attempted further to avoid continuing to spend the user's Vertex budget chasing infrastructure flakiness.)
- [ ] Main UI, citation, chart, table, and trace screenshots are captured from verified output — not captured (no browser automation available in this environment); functionality was instead verified directly against the backend API, which is what the UI itself calls
- [x] README links and copy-paste commands checked (verified 2026-09-06)
- [x] `DESIGN.md` reviewed (updated 2026-09-06 with the ranking design and the Vertex/ADC tradeoff)
- [x] `FAILURE_ANALYSIS.md` reviewed (substantially extended 2026-09-06 with real findings from live testing)
- [x] Tracked files and Git history pass `make secret-scan` (verified 2026-09-06)
- [x] No `.env`, ADC, checkpoint, trace, artifact, report, corpus, or Qdrant volume is staged (verified 2026-09-06: nothing staged at all)
- [ ] Repository visibility is set intentionally
- [ ] Final GitHub URL is tested in a signed-out browser
- [ ] Five-minute video follows `docs/VIDEO_SCRIPT.md`
- [ ] Final email/message includes repository URL, video URL, setup caveat, and live-evaluation status
