FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra frontend --no-install-project \
    && addgroup --system --gid 10002 frontend \
    && adduser --system --uid 10002 --ingroup frontend --no-create-home frontend
COPY frontend ./frontend
RUN uv sync --frozen --no-dev --extra frontend

ENV PATH="/app/.venv/bin:$PATH" \
    HOME="/tmp/frontend-home" \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501
USER 10002:10002
CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
