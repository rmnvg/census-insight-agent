.PHONY: sync test format format-check lint typecheck check verify-offline qdrant-readonly security secret-scan verify-vertex ingest-dry-run initialize-corpus run-backend run-frontend run-executor executor-smoke artifact-replay artifact-schema-report ui-smoke frontend-security live-eval

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

check:
	sh scripts/check.sh

verify-offline: check qdrant-readonly

qdrant-readonly:
	uv run --frozen python -m backend.app.retrieval.validation --expected-points 2058

security:
	uv run --frozen python scripts/verify_security.py

secret-scan:
	uv run --frozen python scripts/scan_secrets.py --history

verify-vertex:
	uv run --frozen python scripts/verify_vertex.py

ingest-dry-run:
	uv run --frozen python -m backend.app.ingestion.cli ingest --source-dir data/source --dry-run

initialize-corpus:
	uv run --frozen python scripts/initialize_corpus.py

live-eval:
	uv run --frozen python scripts/live_evaluation.py

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

ui-smoke:
	uv run --frozen python scripts/smoke_ui_offline.py

frontend-security:
	docker compose exec frontend python -m frontend.security_check
