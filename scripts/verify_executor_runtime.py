import asyncio
import json
import os
import socket
from pathlib import Path

from scripts.smoke_executor import smoke


def network_is_blocked() -> bool:
    connection = socket.socket()
    connection.settimeout(0.2)
    try:
        connection.connect(("1.1.1.1", 53))
    except OSError:
        return True
    finally:
        connection.close()
    return False


def main() -> int:
    credential_variables = sorted(
        key
        for key in os.environ
        if key.startswith("GOOGLE_") or key in {"GEMINI_API_KEY", "GOOGLE_API_KEY"}
    )
    adc_paths = [
        Path("/var/secrets/google/adc.json"),
        Path("/.config/gcloud/application_default_credentials.json"),
    ]
    report = asyncio.run(smoke(Path("/tmp/executor-smoke")))
    root_read_only = False
    try:
        Path("/app/executor-write-test").write_text("unsafe", encoding="utf-8")
    except OSError:
        root_read_only = True
    checks = {
        "network_blocked": network_is_blocked(),
        "credential_variables_absent": not credential_variables,
        "adc_files_absent": not any(path.exists() for path in adc_paths),
        "non_root_user": os.getuid() != 0,
        "root_filesystem_read_only": root_read_only,
        "offline_executor_cases_passed": report["passed"],
    }
    output = {
        "checks": checks,
        "passed": all(checks.values()),
        "credential_variable_names": credential_variables,
        "offline_smoke": report,
    }
    print(json.dumps(output, indent=2))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
