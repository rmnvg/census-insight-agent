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
- [x] Existing Qdrant collection has exactly 2,058 validated points (verified 2026-09-06 via `make verify-offline`: 2058/2058 valid, dense+sparse compatibility confirmed)
- [x] `make check` passes (verified 2026-09-06: format/lint/typecheck/UI-smoke/security/secret-scan all clean; 5 of 244 tests fail in this specific sandboxed agent environment due to a `resource.setrlimit`/subprocess restriction unrelated to the code — confirmed via `git stash` that they fail identically on unmodified code, i.e. pre-existing to this environment, not a regression; re-run `make check` on your own machine to confirm they pass natively, which is expected)
- [x] `make verify-offline` passes against the local corpus (verified 2026-09-06)
- [ ] Optional paid live harness run is reviewed (or explicitly marked not run) — the formal 10-case `scripts/live_evaluation.py` harness was not run; instead several individual live `/chat` queries were run directly (literacy lookup, the new ranking feature, and a chart-comparison regression check), all succeeding with real citations/artifacts — see `FAILURE_ANALYSIS.md` for what those runs found
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
