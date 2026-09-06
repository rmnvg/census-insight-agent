# Interview Notes

## 60-second architecture explanation

The browser talks only to Streamlit, which calls a typed FastAPI boundary. FastAPI runs a LangGraph workflow that classifies the request, loads a runtime skill, performs balanced hybrid retrieval, assesses evidence, and either returns validated claims or a clear limitation. Dense Vertex embeddings and local BM25 vectors are fused by Qdrant RRF. Every citation is an exact slice of a trusted chunk tied to a one-based physical PDF page and PDF checksum. Conversation state uses an async SQLite LangGraph checkpointer. For charts and tables, the model proposes minimal presentation data; application code hydrates provenance and sends generated Python through a filesystem queue to an unprivileged, network-disabled executor. Only validated artifacts are published.

## Technology choices

- **Vertex Gemini + ADC:** GCP identity and governance without API keys; chat uses `ChatGoogleGenerativeAI`, embeddings use `google-genai`.
- **Qdrant:** One database supports named dense/sparse vectors, filters, and server-side fusion.
- **LangGraph:** Explicit state transitions, checkpoint memory, failure nodes, and observable runs.
- **FastAPI/Streamlit:** Strict service contracts plus a fast evaluator-facing UI without coupling the UI to retrieval internals.
- **Filesystem executor queue:** Simple single-host isolation and atomic handoff; a broker is the scaling alternative.

## Request walkthrough and guarantees

The query is resolved against bounded successful memory, then retrieved independently for required regions. Evidence is deduplicated and packed with per-target reservations. A model may assess and propose, but it cannot author provenance. The application validates entity/year/metric/category/unit, calculates derived values, and resolves citation offsets against current-run chunks. A citation page is always the physical one-based PDF page. Missing evidence yields refusal or a coverage limitation, never a guessed answer.

Short-term memory contains validated claim metadata and messages; filesystem working memory contains ignored checkpoints, sanitized traces, queue jobs, and artifacts. Runtime skills are discovered from `skills/*.md` and influence task behavior without changing core trust checks.

Generated code receives only an approved dataset. It has no network, ADC, Qdrant, Docker socket, source PDFs, or other sessions. AST restrictions, resource limits, output allowlists, CSV equality, and source-manifest lineage are layered checks. Docker is adequate for this controlled demo, not hostile multi-tenant execution.

## Failure handling and limitations

Provider timeouts become typed 504 responses with persisted traces. Invalid model output, provenance, policy, executor, and artifact failures remain operational errors—not factual “no evidence” claims. Failed turns do not add invented answers to memory. Visual OCR is quarantined, so some map/chart-only facts are intentionally unavailable. Other limits are no streaming, local SQLite/Qdrant deployment, synchronous ingestion, and browser sessions not restored from URLs.

## Likely evaluator questions

1. **Why not trust retrieval score?** Retrieval proposes candidates; claim and exact-span validation determine support.
2. **Why is Markdown preferred if PDF is authoritative?** Markdown preserves clean text/tables, while the PDF supplies identity, checksum, page count, and citation coordinates.
3. **Can the model forge artifact data?** No. Proposal rows must hydrate from current-run evidence; manifests and companion data are validated before publication.
4. **What happens after a timeout?** A typed, sanitized operational response and failed trace are saved; successful prior memory remains unchanged and no POST is auto-retried.
5. **What would you productionize first?** Stronger executor isolation, authenticated service edges, durable job/checkpoint stores, and verified visual extraction.

## AI-assisted development

AI helped draft code and diagnose recorded failures. Trust came from direct PDF checks, deterministic contracts, regression replays, full offline tests, read-only Qdrant audits, and user-authorized live runs—not from accepting model output as correct.
