#!/usr/bin/env python3
"""Static, value-free audit of Compose trust-boundary controls."""

import re
from pathlib import Path


def _service_block(compose: str, service: str, next_service: str | None) -> str:
    del next_service
    # Anchor to a line that starts with exactly two spaces of indent (the top-level service
    # key under `services:`), not a substring search: an unanchored `.index()` can match a
    # more deeply indented `depends_on: <service>:` reference in an earlier service block,
    # since e.g. "      executor:\n" (6-space depends_on entry) contains "  executor:\n" as a
    # trailing substring and would otherwise be found first.
    match = re.search(rf"(?m)^  {re.escape(service)}:\n", compose)
    if match is None:
        raise ValueError(f"service block for '{service}' not found in compose file")
    start = match.start()
    following = re.search(r"\n(?:  [a-z][\w-]*:|[a-z][\w-]*:)", compose[start + 1 :])
    end = start + 1 + following.start() if following else len(compose)
    return compose[start:end]


def main() -> int:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    frontend = _service_block(compose, "frontend", "executor")
    executor = _service_block(compose, "executor", "qdrant")
    backend = _service_block(compose, "backend", "frontend")
    checks = {
        "frontend_non_root": 'user: "10002:10002"' in frontend,
        "frontend_read_only": "read_only: true" in frontend,
        "frontend_capabilities_dropped": "cap_drop:\n      - ALL" in frontend,
        "frontend_no_credentials": "GOOGLE_" not in frontend and "/var/secrets" not in frontend,
        "frontend_no_sensitive_mounts": all(
            value not in frontend for value in ("execution-queue", "/app/workspace", "docker.sock")
        ),
        "executor_non_root": 'user: "10001:10001"' in executor,
        "executor_no_network": "network_mode: none" in executor,
        "executor_read_only": "read_only: true" in executor,
        "executor_capabilities_dropped": "cap_drop:\n      - ALL" in executor,
        "executor_no_credentials": "GOOGLE_" not in executor and "/var/secrets" not in executor,
        "executor_only_queue_mount": executor.count("source:") == 1
        and "./workspace/execution-queue" in executor,
        "backend_adc_read_only": "/var/secrets/google/adc.json" in backend
        and "read_only: true" in backend,
        "backend_no_docker_socket": "docker.sock" not in backend,
        "qdrant_storage_named_volume": "qdrant_data:/qdrant/storage" in compose,
    }
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
