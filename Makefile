.PHONY: sync test format format-check lint typecheck check verify-vertex run-backend run-frontend run-executor

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

run-backend:
	uv run uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

run-frontend:
	uv run streamlit run frontend/app.py

run-executor:
	uv run uvicorn executor.main:app --host 0.0.0.0 --port 8001 --reload
