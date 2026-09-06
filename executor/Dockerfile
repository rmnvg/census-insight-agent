FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
COPY backend ./backend
COPY executor ./executor
COPY scripts/smoke_executor.py ./scripts/smoke_executor.py
COPY scripts/verify_executor_runtime.py ./scripts/verify_executor_runtime.py
COPY scripts/replay_artifact_failure.py ./scripts/replay_artifact_failure.py
COPY backend/tests/fixtures/prompt5_failed_candidates.json ./backend/tests/fixtures/prompt5_failed_candidates.json
RUN uv sync --frozen --no-dev --extra executor \
    && addgroup --system --gid 10001 executor \
    && adduser --system --uid 10001 --ingroup executor --no-create-home executor

ENV PATH="/app/.venv/bin:$PATH"

USER 10001:10001
CMD ["python", "-m", "executor.worker", "--root", "/execution"]
