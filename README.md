# census-insight-agent

Citation-safe ingestion and hybrid evidence retrieval services for a census insight agent. See [docs/DECISIONS.md](docs/DECISIONS.md) for the fixed architecture decisions.

## Development

Requires Python 3.12 and `uv`.

```shell
cp .env.example .env
uv sync --frozen
make check
```

Run the services locally with the `run-backend`, `run-frontend`, and `run-executor` Make targets, or use Docker Compose after setting `GOOGLE_CREDENTIALS_HOST_PATH` to an ADC JSON file.

## Vertex AI prerequisites

1. Create or select a Google Cloud project with billing configured.
2. Enable the Vertex AI API in that project.
3. Install the Google Cloud CLI and create local Application Default Credentials:

   ```shell
   gcloud auth application-default login
   gcloud auth application-default set-quota-project census-insight-agent
   ```

4. Copy `.env.example` to `.env` and set:

   - `GOOGLE_GENAI_USE_VERTEXAI=true`
   - `GOOGLE_CLOUD_PROJECT` to the target project ID
   - `GOOGLE_CLOUD_LOCATION` (the default is `global`)
   - `GEMINI_CHAT_MODEL`
   - `GEMINI_EMBEDDING_MODEL`
   - `GEMINI_EMBEDDING_DIMENSION=768`
   - `GOOGLE_CREDENTIALS_HOST_PATH` to the absolute host ADC JSON path
   - `GOOGLE_APPLICATION_CREDENTIALS=/var/secrets/google/adc.json` for containers

Gemini is accessed only through Vertex AI with ADC. API-key authentication and provider fallback are intentionally unsupported.

### Live verification

The verification performs billable text generation, structured tool-call, and embedding requests. It is intentionally excluded from the unit test suite.

For local verification, the script uses `GOOGLE_CREDENTIALS_HOST_PATH` when the container credential path is not present:

```shell
uv run --frozen python scripts/verify_vertex.py
```

For verification inside the backend container:

```shell
docker compose run --rm backend python scripts/verify_vertex.py
```

If a configured model is unavailable, the command reports the Vertex API error. Change `GEMINI_CHAT_MODEL` or `GEMINI_EMBEDDING_MODEL` explicitly after confirming model availability in the configured project and location; the application will not silently select another model.

Never copy ADC JSON files into this repository or a Docker image, and never commit credentials. Docker Compose mounts the configured ADC file read-only into the backend only.

## Documents and ingestion

Place authoritative PDFs in `data/source/pdf/` and preferred supplied Markdown in `data/source/markdown/`. Files pair only when their normalized stems match exactly, such as `Karnataka Census.pdf` and `karnataka_census.md`. For different names or explicit metadata, add a `*.override.json` file under `data/manifests/` with `document_id`, `title`, `region`, `pdf_filename`, and optional `markdown_filename`. Private absolute paths are never written to generated manifests.

Reliable Markdown page markers use one of these forms:

```markdown
<!-- page: 1 -->
[PAGE 2]
```

Without markers, Markdown blocks are aligned to normalized PDF page anchors only when there is a confident unique match. Ambiguous mappings fail ingestion. `--review-override` explicitly discards unresolved Markdown mapping and uses page-specific PyMuPDF4LLM extraction; it never invents a page number.

Run a free dry run, which parses and chunks but does not call Gemini or write Qdrant points:

```shell
docker compose run --rm --no-deps --build backend python -m backend.app.ingestion.cli ingest --source-dir /app/data/source --dry-run --report-output /app/data/processed/dry-run-report.json
```

The report contains a checksum-bound page coverage contract for every document, including page
lists for indexed Markdown, indexed fallback, reviewed blank/decorative exclusions, quarantined
visual exclusions, approved manual transcriptions, and failed mappings. Inspect coverage with:

```shell
python -m json.tool data/processed/dry-run-report.json
```

A percentage below 100 means some authoritative PDF pages were intentionally excluded or failed;
it does not mean their information is absent from the PDF. Raw OCR under
`data/processed/page-review/` is explicitly rejected as an ingestion source.

Run real ingestion after reviewing the report and estimated embedding requests:

```shell
docker compose run --rm --build backend python -m backend.app.ingestion.cli ingest --source-dir /app/data/source --report-output /app/data/processed/ingestion-report.json
```

Ingestion is idempotent for unchanged PDF checksums and ingestion versions. Rebuilding an incompatible collection is intentionally explicit and destructive to that collection only:

```shell
docker compose run --rm --build backend python -m backend.app.ingestion.cli ingest --source-dir /app/data/source --rebuild --report-output /app/data/processed/rebuild-report.json
```

## Offline OCR review

The page-review OCR utility uses Debian's pinned Tesseract 5 package and Pillow
inside a dedicated Docker service. The service has networking disabled, receives
no Google credentials, and is not connected to the ingestion workflow or Qdrant.
It processes only the explicitly approved Karnataka pages and preserves raw output
one PDF page per text file:

```bash
docker compose run --rm --build ocr-review
```

Results are written under `data/processed/page-review/`. OCR from charts and maps
is review material only: label/value and label/legend relationships must be
checked manually against the authoritative PDF page before any later integration.

Human-reviewed recovery records belong under `data/manifests/manual-transcriptions/`. A record must
contain `document_id`, one-based `page_number`, non-empty `transcription`, optional
`structured_values`, a human `reviewer`, timezone-aware `reviewed_at`, `source_checksum`,
`verification_notes`, and `status: "approved"`. Approval is accepted only while the checksum matches
the authoritative PDF.

### Retrieval

The development API performs Qdrant RRF hybrid search over Gemini dense and local BM25 sparse vectors:

```shell
curl -X POST http://localhost:8000/retrieval/search \
  -H 'content-type: application/json' \
  -d '{"query":"population of Mysuru","regions":["Karnataka"],"top_k":5,"debug":true}'
```

Administrative ingestion currently runs synchronously. This keeps take-home deployment simple, but a production service should enqueue ingestion so long PDF extraction and embedding jobs do not occupy an API worker.

The checked-in, manually source-verified retrieval cases are in
`evals/real_corpus_cases.json`. Run the live dense + sparse + Qdrant RRF evaluation without
generating answers:

```shell
docker compose run --rm --no-deps backend uv run --frozen python evals/run_retrieval.py --cases evals/real_corpus_cases.json --output /app/data/processed/retrieval-evaluation-report.json
```

Recall@5 and Recall@10 are target-page recall across answerable gold targets. MRR uses the first
matching document/page rank per answerable case. Citation-page accuracy verifies positive page
provenance and the exact citation-substring invariant for every returned result. Filter accuracy
requires every result to obey its requested document or region filter.

Audit every stored point, payload, and named vector without changing Qdrant:

```shell
docker compose run --rm --no-deps backend uv run --frozen python -m backend.app.retrieval.validation \
  --expected-points 2058 \
  --output /app/data/processed/qdrant-validation-report.json
```

Inspect a single hybrid query and all three diagnostic rankings:

```shell
docker compose run --rm --no-deps backend uv run --frozen python -m backend.app.retrieval.cli search --query "What was the literacy rate in Karnataka?" --top-k 5 --debug
```

An unchanged ingestion is skipped before dense or sparse embedding and before any Qdrant upsert.
The ingestion report exposes embedding-request, sparse-embedding, upsert-operation, and upserted-point
counters so a zero-work rerun is auditable.

## Conversational agent API

Start the persistent API and its existing Qdrant dependency:

```shell
docker compose up -d --build qdrant backend
```

Create a session, retaining the returned `session_id`:

```shell
curl -X POST http://localhost:8000/sessions
```

Send a turn using that validated session ID:

```shell
curl -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"session_id":"SESSION_ID","message":"What was the literacy rate in Karnataka in 2011?"}'
```

The response `trace_id` identifies its safe structured trace:

```shell
curl http://localhost:8000/runs/TRACE_ID/trace
```

Check session metadata with `GET /sessions/{session_id}`. Checkpoints persist in
`workspace/checkpoints.sqlite`; run traces persist under
`workspace/sessions/{session_id}/traces/{run_id}.json`. Neither contains credentials or vectors.
New checkpoints carry schema version 4 and store application state as JSON-safe primitives,
including a bounded history of validated claim metadata. Pre-v3 successful traces are migrated
without treating assistant prose as evidence.

Replay the saved Prompt 4F comparison without Gemini and without Qdrant writes:

```shell
docker compose run --rm --no-deps backend uv run --frozen python scripts/replay_turn2.py \
  --trace /app/workspace/sessions/25f5f5982d0645d19ee909be14fa9a84/traces/30e4237c-b44d-4703-aaae-93fb9bcbab71.json \
  --qdrant-url http://qdrant:6333 \
  --collection census_documents \
  --output /app/data/processed/agent-turn2-offline-replay.json
```

The four-turn live smoke test is deliberately manual because it invokes paid Vertex Gemini calls:

```shell
docker compose exec backend uv run --frozen python scripts/smoke_agent_memory.py \
  --base-url http://localhost:8000
```

Replay the source-page follow-up from current Qdrant points without Gemini, embeddings, or writes:

```shell
docker compose run --rm --no-deps backend uv run --frozen python scripts/replay_turn3.py \
  --session-id 98194e7f49d4468f823359f020522028 \
  --output /app/data/processed/agent-turn3-offline-replay.json
```

After Turns 1 and 2 pass, resume that session at Turn 3 without repeating the paid turns:

```shell
docker compose exec backend uv run --frozen python scripts/smoke_agent_memory.py \
  --base-url http://localhost:8000 --session-id SESSION_ID --start-turn 3
```

Agent requests have a configurable overall deadline (`AGENT_REQUEST_TIMEOUT_SECONDS`, default 180)
and a per-provider-call budget (`AGENT_PROVIDER_TIMEOUT_SECONDS`, default 60). The application
defaults to no timeout retry so one provider attempt receives the complete call budget; SDK retries
are also disabled so deadlines do not stack. Comparison assessment input is bounded by
`AGENT_ASSESSMENT_MAX_CHARACTERS` (default 12000) and `AGENT_ASSESSMENT_MAX_CHUNKS` (default 12),
while retaining complete chunks from both regions. Evidence-assessment timeouts return HTTP 504
with a retryable error and persisted trace ID.
A failed turn is not advanced as the session's successful checkpoint. Retry by sending the same
user message once after inspecting the failed trace; the failed attempt is not duplicated in model
conversation context.

Runtime Markdown skills are read only from `skills/`. Summary and inconsistency skills guide the
agent. Chart and table skills guide task-specific Python generation after a citation-safe dataset is
validated.

## Isolated chart and table artifacts

The backend submits typed JSON jobs atomically under `workspace/execution-queue/`. A dedicated
non-root executor claims jobs by rename, validates generated Python's AST, runs it with process and
output limits, validates every declared output, and returns a typed result through the same queue.
The executor has no network, host port, Docker socket, ADC mount, Google environment variables,
Qdrant connection, or access to session directories. Its root filesystem is read-only; only `/tmp`
and the narrowly scoped queue are writable.

Accepted artifacts are moved into:

```text
workspace/sessions/{session_id}/artifacts/{artifact_id}/
├── artifact.json
├── generated.py
├── input.json
├── execution-result.json
├── source-manifest.json
└── chart.png / plotted-data.csv / table.csv / table.md
```

Generated code remains locally inspectable but is never served by the public API. List and download
validated artifacts with:

```shell
curl http://localhost:8000/sessions/SESSION_ID/artifacts
curl http://localhost:8000/sessions/SESSION_ID/artifacts/ARTIFACT_ID
curl -OJ http://localhost:8000/sessions/SESSION_ID/artifacts/ARTIFACT_ID/files/table.csv
curl http://localhost:8000/health/executor
```

Build and run the offline executor tests without Gemini:

```shell
docker compose build executor
docker compose run --rm --no-deps executor python scripts/verify_executor_runtime.py
docker compose up -d executor
docker compose run --rm --no-deps backend uv run --frozen python scripts/smoke_executor_handoff.py
```

Replay the Prompt 5 comparison-packing failure and execute its synthetic approved dataset entirely
offline. The fixture preserves the failed ten-candidate ordering but contains no production Census
values:

```shell
docker compose build executor
docker compose run --rm --no-deps executor python scripts/replay_artifact_failure.py
```

Gemini receives only a small semantic `ArtifactDatasetProposal`; authoritative chart/table identity
is omitted and comes from the validated application requirement. Gemini never receives the strict
internal dataset, source manifest, checksums, paths, or execution protocol. Inspect the deterministic
schema-complexity guard with:

```shell
uv run --frozen python scripts/report_artifact_schema.py \
  --output docs/artifact-schema-complexity.json
```

After starting Qdrant, the backend, and executor, replay trusted existing comparison evidence
through proposal hydration and the isolated executor without Gemini, embeddings, or Qdrant writes:

```shell
docker compose run --rm --no-deps backend uv run --frozen python \
  scripts/replay_artifact_proposal.py \
  --session-id 98194e7f49d4468f823359f020522028 \
  --trace-id 0a4cdb00-c807-47a0-8635-648f1bdf5147 \
  --output data/processed/artifact-proposal-offline-replay.json
```

For multi-region artifacts, the backend performs one region-filtered data search per target. It
reserves direct, value-bearing evidence for each target within the existing assessment budget before
adding supporting candidates. Generic excluded-visual limitations are shown only when an excluded
page is material to the requested data; safe indexed statewide tables take precedence.

Container isolation and AST filtering reduce risk but are not a hardened multi-tenant sandbox.
Resource controls vary by Docker host, AST policy cannot prove program intent, and a production
multi-tenant deployment should use stronger per-job isolation such as microVMs or a dedicated
sandbox runtime.
