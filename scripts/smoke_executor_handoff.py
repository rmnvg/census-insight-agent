import asyncio
import json
from pathlib import Path
from uuid import uuid4

from backend.app.agent.persistence import SessionStore
from backend.app.execution.client import ArtifactStore, ExecutionQueueClient
from scripts.smoke_executor import TABLE_CODE, expected_table, request


async def main() -> int:
    workspace = Path("/app/workspace")
    queue_root = workspace / "execution-queue"
    sessions = SessionStore(workspace)
    await sessions.initialize()
    session = await sessions.create()
    value = request(TABLE_CODE, expected_table()).model_copy(
        update={"session_id": session.session_id, "run_id": str(uuid4())}
    )
    client = ExecutionQueueClient(queue_root)
    client.submit(value)
    result = await client.wait(value.job_id, value.timeout_seconds + 5)
    if result.status != "succeeded":
        print(result.model_dump_json(indent=2))
        return 1
    descriptor = ArtifactStore(workspace, queue_root).accept(
        value,
        result,
        artifact_type="table",
        title="Offline executor handoff",
        primary_filename="table.md",
    )
    artifact_root = workspace / "sessions" / session.session_id / "artifacts"
    checks = {
        "backend_received_result": result.status == "succeeded",
        "session_workspace_persisted": (artifact_root / descriptor.artifact_id).is_dir(),
        "generated_code_inspectable": (
            artifact_root / descriptor.artifact_id / "generated.py"
        ).is_file(),
        "download_metadata_present": bool(descriptor.download_url),
    }
    print(
        json.dumps(
            {
                "checks": checks,
                "passed": all(checks.values()),
                "session_id": session.session_id,
                "artifact": descriptor.model_dump(mode="json"),
            },
            indent=2,
        )
    )
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
