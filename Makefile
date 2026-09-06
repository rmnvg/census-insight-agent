.PHONY: sync test format format-check lint typecheck check verify-vertex ingest-dry-run run-backend run-frontend run-executor executor-smoke artifact-replay artifact-schema-report

sync:
	uv sync --frozen

test:
	uv run --frozen pytest

format:
	uv run --frozen ruff format .

format-check:
	uv run --frozen ruff format --check .

lint:
	uv run --frozen ruff check .

typecheck:
	uv run --frozen mypy

check: format-check lint typecheck test

verify-vertex:
	uv run --frozen python scripts/verify_vertex.py

ingest-dry-run:
	uv run --frozen python -m backend.app.ingestion.cli ingest --source-dir data/source --dry-run

run-backend:
	uv run uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

run-frontend:
	uv run streamlit run frontend/app.py

run-executor:
	uv run python -m executor.worker --root workspace/execution-queue

executor-smoke:
	docker compose run --rm --no-deps executor python scripts/verify_executor_runtime.py

artifact-replay:
	docker compose run --rm --no-deps executor python scripts/replay_artifact_failure.py

artifact-schema-report:
	uv run --frozen python scripts/report_artifact_schema.py --output docs/artifact-schema-complexity.json
