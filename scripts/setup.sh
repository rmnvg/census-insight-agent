#!/bin/sh
# Run from the repository root after configuring .env and supplying the corpus.
set -eu

if [ "${1:-}" != "--allow-paid-calls" ]; then
    echo "Usage: sh scripts/setup.sh --allow-paid-calls"
    echo "First-time indexing uses billable Vertex embeddings. Existing valid data is retained."
    exit 2
fi
if [ ! -f .env ]; then
    echo "Copy .env.example to .env and configure your GCP project and ADC path first."
    exit 1
fi
docker compose config --quiet
docker compose build
docker compose run --rm --no-deps backend uv run --frozen python -m backend.app.ingestion.cli ingest --source-dir /app/data/source --dry-run --report-output /app/data/processed/dry-run-report.json
docker compose up -d --wait qdrant
docker compose run --rm --no-deps backend uv run --frozen python scripts/initialize_corpus.py --allow-paid-calls
docker compose up -d --wait
echo "Ready: http://localhost:8501 (API: http://localhost:8000/docs)"
