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
docker compose run --rm --no-deps backend uv run --frozen python -m backend.app.retrieval.validation --expected-points 2058
```

Inspect a single hybrid query and all three diagnostic rankings:

```shell
docker compose run --rm --no-deps backend uv run --frozen python -m backend.app.retrieval.cli search --query "What was the literacy rate in Karnataka?" --top-k 5 --debug
```

An unchanged ingestion is skipped before dense or sparse embedding and before any Qdrant upsert.
The ingestion report exposes embedding-request, sparse-embedding, upsert-operation, and upserted-point
counters so a zero-work rerun is auditable.
