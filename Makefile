.PHONY: build-tables rescore-trust-benchmark integration-test up-scale up-gvisor executor-smoke-gvisor publish-trust-scorecard trust-benchmark web-install web-check web-dev sync test format format-check lint typecheck check verify-offline qdrant-readonly security secret-scan verify-vertex ingest-dry-run initialize-corpus run-backend run-frontend run-executor executor-smoke artifact-replay artifact-schema-report ui-smoke frontend-security live-eval prune-checkpoints prune-checkpoints-dry-run

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

# Postgres state and the real-Redis worker path against throwaway containers (needs Docker).
integration-test:
	sh scripts/integration_test.sh

# Two API replicas over Postgres state plus the durable Celery/Redis upload worker.
up-scale:
	docker compose -f docker-compose.yml -f docker-compose.scale.yml up --build --detach --wait

# The executor under gVisor (Linux hosts with runsc registered as a Docker runtime).
up-gvisor:
	docker compose -f docker-compose.yml -f docker-compose.gvisor.yml up --build --detach --wait

executor-smoke-gvisor:
	docker compose -f docker-compose.yml -f docker-compose.gvisor.yml exec -T executor python scripts/verify_executor_runtime.py --expect-gvisor

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

# No model calls: derive table stores for an already-indexed corpus (ingestion writes them itself).
build-tables:
	uv run --frozen python -m backend.app.tables.cli

live-eval:
	uv run --frozen python scripts/live_evaluation.py

# Billable: every case is a full agent turn. Copy the output to evals/ to ship it with the repo.
trust-benchmark:
	uv run --frozen python scripts/trust_benchmark.py --allow-paid-calls --repeat 2 --output data/processed/trust-scorecard.json --responses-out data/processed/trust-responses.jsonl

# No model calls: re-score the last recorded run with the current scorer.
rescore-trust-benchmark:
	uv run --frozen python scripts/rescore_trust_benchmark.py --responses data/processed/trust-responses.jsonl --scorecard data/processed/trust-scorecard.json --output data/processed/trust-scorecard.json

# No model calls: records a saved scorecard as a Langfuse experiment run (needs LANGFUSE_* set).
publish-trust-scorecard:
	uv run --frozen python scripts/publish_trust_scorecard.py --scorecard evals/trust-scorecard.json

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

prune-checkpoints-dry-run:
	uv run --frozen python scripts/prune_checkpoints.py

prune-checkpoints:
	uv run --frozen python scripts/prune_checkpoints.py --allow-delete

web-install:
	cd web && npm ci --no-audit --no-fund

web-check:
	cd web && npm run check && npm run build

web-dev:
	cd web && CENSUS_API_BASE_URL=http://127.0.0.1:8000 npm run dev
