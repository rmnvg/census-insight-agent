"""Sanitized runtime assertions for the isolated Streamlit container."""

from __future__ import annotations

import os
import socket
from pathlib import Path


def _effective_capabilities() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("CapEff:"):
                return int(line.split(":", 1)[1].strip(), 16)
    except (OSError, ValueError):
        return None
    return None


def _root_is_read_only() -> bool:
    try:
        mounts = Path("/proc/mounts").read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for mount in mounts:
        fields = mount.split()
        if len(fields) >= 4 and fields[1] == "/":
            return "ro" in fields[3].split(",")
    return False


def _qdrant_dns_is_unavailable() -> bool:
    try:
        socket.getaddrinfo("qdrant", 6333)
    except socket.gaierror:
        return True
    return False


def runtime_checks() -> dict[str, bool]:
    """Return checks without exposing environment values or filesystem contents."""
    forbidden_environment = {
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GEMINI_CHAT_MODEL",
        "GEMINI_EMBEDDING_MODEL",
        "QDRANT_URL",
    }
    return {
        "non_root_user": os.geteuid() != 0,
        "no_google_or_qdrant_environment": not (forbidden_environment & os.environ.keys()),
        "no_adc_file": not Path("/var/secrets/google/adc.json").exists(),
        "no_execution_queue": not Path("/execution").exists(),
        "no_session_artifact_workspace": not Path("/app/workspace/sessions").exists(),
        "no_docker_socket": not Path("/var/run/docker.sock").exists(),
        "no_direct_qdrant_network_route": _qdrant_dns_is_unavailable(),
        "no_effective_capabilities": _effective_capabilities() == 0,
        "read_only_root_filesystem": _root_is_read_only(),
    }


def main() -> int:
    checks = runtime_checks()
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
