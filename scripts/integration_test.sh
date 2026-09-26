#!/bin/sh
# Run the Postgres-state, real-Redis worker, and Qdrant-server tests against throwaway containers.
# No credentials or model calls; containers use high loopback ports and are removed afterwards.
set -eu

suffix="$$"
postgres="census-it-postgres-$suffix"
redis="census-it-redis-$suffix"
qdrant="census-it-qdrant-$suffix"
password="integration-$suffix"
cleanup() { docker rm -f "$postgres" "$redis" "$qdrant" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

docker run -d --name "$postgres" -e POSTGRES_PASSWORD="$password" -p 127.0.0.1:55433:5432 \
    postgres:17.6-alpine@sha256:ef257d85f76e48da1c64832459b59fcaba1a4dac97bf5d7450c77753542eee94 >/dev/null
docker run -d --name "$redis" -p 127.0.0.1:56380:6379 \
    redis:7.4.6-alpine@sha256:3b73847e72874be07e6657b129a94761662b79bc0f679273757d4218573b2a98 >/dev/null
docker run -d --name "$qdrant" -p 127.0.0.1:56333:6333 \
    qdrant/qdrant:v1.15.4@sha256:6ac4807063bbecddca0250bfbcff52acf18c22263b904d12919349e6d0a408f1 >/dev/null
until docker exec "$postgres" pg_isready -U postgres >/dev/null 2>&1; do sleep 1; done
until docker exec "$redis" redis-cli ping >/dev/null 2>&1; do sleep 1; done
until curl -fsS http://127.0.0.1:56333/readyz >/dev/null 2>&1; do sleep 1; done

TEST_POSTGRES_URL="postgresql://postgres@127.0.0.1:55433/postgres" PGPASSWORD="$password" \
    TEST_REDIS_URL="redis://127.0.0.1:56380/0" TEST_QDRANT_URL="http://127.0.0.1:56333" \
    uv run --frozen pytest -rA -p no:warnings backend/tests/test_postgres_state.py \
    backend/tests/test_ingestion_worker.py backend/tests/test_qdrant_server.py
