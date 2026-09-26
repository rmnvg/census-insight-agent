# Interview Notes

## 60-second architecture explanation

The browser talks only to a Next.js client whose same-origin proxy forwards an allowlist of routes to a typed FastAPI boundary. FastAPI runs a LangGraph workflow that classifies the request, loads a runtime skill, performs balanced hybrid retrieval, assesses evidence, and either returns validated claims or a clear limitation. Dense Vertex embeddings and local BM25 vectors are fused by Qdrant RRF. Every citation is an exact slice of a trusted chunk tied to a one-based physical PDF page and PDF checksum. Conversation state uses an async SQLite LangGraph checkpointer. For charts and tables, the model proposes minimal presentation data; application code hydrates provenance and sends generated Python through a filesystem queue to an unprivileged, network-disabled executor. Only validated artifacts are published.

## Technology choices

- **Vertex Gemini + ADC:** GCP identity and governance without API keys; chat uses `ChatGoogleGenerativeAI`, embeddings use `google-genai`.
- **Qdrant:** One database supports named dense/sparse vectors, filters, and server-side fusion.
- **LangGraph:** Explicit state transitions, checkpoint memory, failure nodes, and observable runs.
- **FastAPI/Next.js:** Strict service contracts plus a UI that reaches the backend only through an allowlisted proxy, never retrieval internals. The original Streamlit client remains as an optional Compose profile.
- **Filesystem executor queue:** Simple single-host isolation and atomic handoff; a broker is the scaling alternative.

## Request walkthrough and guarantees

The query is resolved against bounded successful memory, then retrieved independently for required regions. Evidence is deduplicated and packed with per-target reservations. A model may assess and propose, but it cannot author provenance. The application validates entity/year/metric/category/unit, calculates derived values, and resolves citation offsets against current-run chunks. A citation page is always the physical one-based PDF page. Missing evidence yields refusal or a coverage limitation, never a guessed answer.

Short-term memory contains validated claim metadata and messages; filesystem working memory contains ignored checkpoints, sanitized traces, queue jobs, and artifacts. Runtime skills are discovered from `skills/*.md` and influence task behavior without changing core trust checks.

Generated code receives only an approved dataset. It has no network, ADC, Qdrant, Docker socket, source PDFs, or other sessions. AST restrictions, resource limits, output allowlists, CSV equality, and source-manifest lineage are layered checks. Docker is adequate for this controlled demo, not hostile multi-tenant execution.

## Failure handling and limitations

Provider timeouts become typed 504 responses with persisted traces. Invalid model output, provenance, policy, executor, and artifact failures remain operational errors—not factual “no evidence” claims. Failed turns do not add invented answers to memory. Visual OCR is quarantined, so some map/chart-only facts are intentionally unavailable. The default deployment is single-process (SQLite checkpoints, in-process session locks, in-process uploads, the executor on the host kernel); opt-in modes add Postgres state, a durable upload worker, and gVisor. There is no user authentication.

## Likely evaluator questions

1. **Why not trust retrieval score?** Retrieval proposes candidates; claim and exact-span validation determine support.
2. **Why is Markdown preferred if PDF is authoritative?** Markdown preserves clean text/tables, while the PDF supplies identity, checksum, page count, and citation coordinates.
3. **Can the model forge artifact data?** No. Proposal rows must hydrate from current-run evidence; manifests and companion data are validated before publication.
4. **What happens after a timeout?** A typed, sanitized operational response and failed trace are saved; successful prior memory remains unchanged and no POST is auto-retried.
5. **What would you productionize first?** Authentication and executor isolation, because both are prerequisites for exposing the service at all. Shared state, a durable upload worker, gVisor, and evaluation history are built as opt-in modes. What remains is an authenticated gateway with TLS, a microVM per job, and object storage across hosts.
6. **How does it scale horizontally?** `docker-compose.scale.yml` runs two replicas over Postgres. Swapping `AsyncSqliteSaver` for `AsyncPostgresSaver` was necessary but not sufficient. The session tables shared the SQLite file, and the per-session lock was in-process, so two replicas could both extend the same stale checkpoint. The lock is now a Postgres advisory lock held on a dedicated connection; a crashed replica's connection drops and releases it. A test runs overlapping turns on two replicas and fails if the lock is removed. Starting replicas together exposed a real deadlock: a blocked advisory-lock call counts as an open transaction, and LangGraph's `CREATE INDEX CONCURRENTLY` migration waits on it. The cycle ran through the application, so Postgres could not detect it. Migration waiters now poll.
7. **Is Docker enough for AI-generated code?** For this demo it has no network, no credentials, a read-only root, all capabilities dropped, resource limits, and an AST gate, but it shares the host kernel. The gVisor overlay puts generated code on a user-space kernel (gVisor is not a VM). Verified under `runsc` on real Linux, and in CI. Firecracker or Kata microVMs are the step for hostile multi-tenant load. The Linux run also exposed an older `queue-init` bug that Docker Desktop had hidden.
8. **Why Celery for uploads?** Indexing already ran in a thread, so the event loop was never blocked. The real gaps were losing in-flight jobs on restart and sharing CPU with the API. Late acknowledgement plus Redis redelivery covers crashes, and deterministic point IDs make reruns idempotent. A worker killed mid-job with `SIGKILL` was redelivered and finished with exactly the right point count. On GCP, Cloud Tasks would remove Redis.
9. **How does evaluation keep up as models change?** The trust benchmark is hand-verified ground truth expressed as facts, e.g. (Odisha, sex ratio, 2011, urban, all persons, 932), scored by code independent of the agent. A claim must state that exact slot; a claim for the same slot with another value is a wrong answer. Each number must also be found independently on its region's row in the cited table, in a column whose headers agree with the claim. The first scorer only matched numbers anywhere, and passed four deliberately corrupted real answers; the current one fails them all. Unnecessary refusals are counted apart from wrong answers, runs repeat to expose instability, and raw responses are recorded so scorer changes can be re-scored for free. Scorecards publish to Langfuse as experiments, and an LLM judge never decides numeric grounding.

## AI-assisted development

AI helped draft code and diagnose recorded failures. Trust came from direct PDF checks, deterministic contracts, regression replays, full offline tests, read-only Qdrant audits, and user-authorized live runs—not from accepting model output as correct.
