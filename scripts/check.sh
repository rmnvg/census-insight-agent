#!/bin/sh
set -eu

uv run --frozen ruff format --check .
uv run --frozen ruff check .
uv run --frozen mypy
uv run --frozen pytest -q
uv run --frozen python scripts/smoke_ui_offline.py
uv run --frozen python scripts/verify_security.py
uv run --frozen python scripts/scan_secrets.py
