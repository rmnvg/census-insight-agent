# Assignment coverage

| Requirement | Implementation / how to inspect |
|---|---|
| Python and FastAPI | `backend/app/main.py`, typed public routes in `backend/app/api.py` |
| Streamlit chat and inline artifacts | `frontend/app.py`, `frontend/renderers/` |
| Real tool-capable LLM | Vertex Gemini through `ChatGoogleGenerativeAI`, ADC only; `backend/app/providers/chat.py` and `backend/app/agent/provider.py` |
| Task routing | LangGraph classification and conditional routes for lookup, summary, comparison, source follow-up, chart, table, ranking, and inconsistency analysis |
| Retrieval | Qdrant dense Vertex embeddings plus local BM25, fused with RRF; `backend/app/retrieval/` |
| Every factual claim cited | Structured claims, exact source offsets, physical PDF pages, deterministic quote checks and semantic support assessment |
| Model writes and executes code | `generate_artifact_code` writes Python; filesystem queue hands it to the isolated executor; charts are not pre-generated templates |
| stdout/stderr and failures | Bounded pipe capture, process-group timeouts, typed errors, one bounded repair for eligible runtime/output failures |
| Conversation memory | SQLite LangGraph checkpoints and bounded validated claim history; per-session serialization |
| Reusable runtime instructions | Discovered Markdown files in `skills/`, loaded by the agent as tools |
| Filesystem workspace | `workspace/checkpoints.sqlite`, `workspace/sessions/<id>/traces`, `workspace/sessions/<id>/artifacts`, `workspace/execution-queue` |
| Graceful refusal | Out-of-scope classification, insufficient-evidence responses, coverage limitations, citation repair/refusal |
| Clear contracts | Pydantic models for claims, citations, tool arguments, evidence, artifact proposals/datasets, execution requests/results |
| Observability | UI execution details and `GET /runs/{run_id}/trace`: tool inputs, bounded summaries, statuses, counts, timing and sanitized failures |
| Important tests | `backend/tests/` covers retrieval, citations, tools, memory and execution; `frontend/tests/` covers rendering and submission flow |
| Reproducibility | Python 3.12, pinned dependencies and `uv.lock`, digest-pinned containers, Linux CI |
| Setup | README; `sh scripts/setup.sh --allow-paid-calls` for initial setup; `docker compose up` thereafter |
| Design and limitations | `DESIGN.md`, `FAILURE_ANALYSIS.md` |
| Submission | Clean ZIP with source documents; walkthrough and transcript provided alongside it |

This is a local evaluator deployment. It intentionally requires the evaluator's own Vertex-enabled GCP
project and ADC; distributing working credentials would be inappropriate. No offline model substitute
is presented as live inference. The UI's offline smoke test uses explicit fixtures only in tests.
