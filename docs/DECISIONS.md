# Architecture Decisions

## AI platform and authentication

Gemini requests are sent through Vertex AI on Google Cloud. Services authenticate with Application Default Credentials (ADC), using `GOOGLE_APPLICATION_CREDENTIALS` when a service-account credential file is mounted. API-key authentication through `GEMINI_API_KEY` or `GOOGLE_API_KEY` is prohibited.

LangChain integrations must use `langchain_google_genai.ChatGoogleGenerativeAI` in Vertex AI mode. The deprecated `ChatVertexAI` integration is not used.

## Data and retrieval

Qdrant is the project's only vector database.

Original PDFs remain authoritative. Provided Markdown is the preferred extracted representation. PyMuPDF4LLM is reserved for pages without supplied Markdown and for newly received PDFs.

Statement tables are parsed once, from the whole Markdown table, into a derived per-document JSON store. Each row is bound to the indexed chunk containing its exact line, so the store never becomes a second source of evidence: answers cite Qdrant chunks, re-read and checked at query time.

## Trust boundary

The executor is an isolated service. It must never receive Google Cloud credentials, credential volumes, or Google authentication environment variables. Any future executor protocol must pass only the minimum task data required.

## Ingestion and retrieval

PDF identity and page numbering remain authoritative. Supplied Markdown is preferred only after explicit-marker parsing or confident deterministic page alignment. Missing pages use PyMuPDF4LLM. Chunks never cross page boundaries and citation snippets remain verbatim.

Retrieval combines 768-dimensional Gemini dense vectors and local FastEmbed BM25 sparse vectors in one Qdrant collection. Qdrant performs reciprocal-rank fusion, and every result must carry valid page provenance.

## Deployment modes

The default deployment is one API process on one host: SQLite conversation state, in-process
upload indexing, and the executor under Docker's default runtime. Two opt-in Compose overlays
change that without changing any trust rule:

- `docker-compose.scale.yml`: Postgres holds LangGraph checkpoints, sessions, and transcripts, and
  the per-session lock is a Postgres advisory lock, so API replicas are stateless. Uploads are
  indexed by a Celery worker with Redis as the broker. Postgres is conversation state only; Qdrant
  remains the only vector database.
- `docker-compose.gvisor.yml`: the executor runs under gVisor (`runsc`). No other service changes.

Langfuse tracing is optional and off unless configured. It is a monitoring sink, never an input
to an answer, and it receives no prompt, evidence, or answer text unless
`LANGFUSE_CAPTURE_CONTENT=true`. Evaluation scores come from code, never from an LLM judge.

## Deferred work

Streamed token output (answers are released only after citation validation), a microVM per
executor job (Firecracker or Kata Containers), object storage for uploads and artifacts across
hosts, a broker in place of the filesystem executor queue, an authenticated gateway with TLS, and
migrating existing SQLite history into Postgres. See `DESIGN.md`'s "Production modes" and
"Tradeoffs, alternatives, and intentionally skipped work".
