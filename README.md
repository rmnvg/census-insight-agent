# census-insight-agent

Initial service scaffold for a census insight agent. See [docs/DECISIONS.md](docs/DECISIONS.md) for the fixed architecture decisions.

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
