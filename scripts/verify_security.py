#!/usr/bin/env python3
"""Static, value-free audit of Compose trust-boundary controls, including the opt-in overlays."""

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


def _top_level_services(compose: str) -> list[str]:
    services = compose.split("\nservices:\n", 1)[1] if "\nservices:\n" in compose else compose
    services = re.split(r"(?m)^[a-z][\w-]*:", services, maxsplit=1)[0]
    return re.findall(r"(?m)^  ([a-z][\w-]*):\n", services)


def _scale_checks(compose: str) -> dict[str, bool]:
    postgres = _service_block(compose, "postgres", None)
    redis = _service_block(compose, "redis", None)
    worker = _service_block(compose, "ingest-worker", None)
    backend = _service_block(compose, "backend", None)
    return {
        "scale_state_stores_unpublished": "ports:" not in postgres and "ports:" not in redis,
        "scale_state_stores_only_on_state_network": all(
            block.split("networks:", 1)[1].split() == ["-", "state"] for block in (postgres, redis)
        ),
        "scale_state_network_internal": re.search(r"(?m)^  state:\n    internal: true$", compose)
        is not None,
        "scale_database_url_has_no_password": re.search(
            r"STATE_DATABASE_URL:\s*\"?postgres(?:ql)?://[^\s:/@\"]+@", backend
        )
        is not None,
        "scale_worker_minimal_mounts": all(
            value not in worker
            for value in ("/app/workspace", "execution-queue", "docker.sock", "frontend_api")
        ),
        "scale_executor_untouched": "executor" not in _top_level_services(compose),
    }


def _gvisor_checks(compose: str) -> dict[str, bool]:
    executor = _service_block(compose, "executor", None)
    settings = [line.strip() for line in executor.splitlines()[1:] if line.strip()]
    return {
        "gvisor_only_changes_executor_runtime": _top_level_services(compose) == ["executor"]
        and settings == ["runtime: runsc"],
    }


def main() -> int:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    frontend = _service_block(compose, "frontend", "executor")
    executor = _service_block(compose, "executor", "qdrant")
    backend = _service_block(compose, "backend", "frontend")
    web = _service_block(compose, "web", "frontend")
    checks = {
        "web_non_root": 'user: "1000:1000"' in web,
        "web_read_only": "read_only: true" in web,
        "web_capabilities_dropped": "cap_drop:\n      - ALL" in web,
        "web_no_credentials": "GOOGLE_" not in web and "/var/secrets" not in web,
        "web_no_mounts": "volumes:" not in web and "docker.sock" not in web,
        "web_only_frontend_network": "- frontend_api" in web and "backend_data" not in web,
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
        "only_backend_gets_langfuse_keys": all(
            "LANGFUSE_" not in block for block in (executor, web, frontend)
        ),
        "executor_only_queue_mount": executor.count("source:") == 1
        and "./workspace/execution-queue" in executor,
        "backend_adc_read_only": "/var/secrets/google/adc.json" in backend
        and "read_only: true" in backend,
        "backend_no_docker_socket": "docker.sock" not in backend,
        "qdrant_storage_named_volume": "qdrant_data:/qdrant/storage" in compose,
        **_scale_checks(Path("docker-compose.scale.yml").read_text(encoding="utf-8")),
        **_gvisor_checks(Path("docker-compose.gvisor.yml").read_text(encoding="utf-8")),
    }
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
