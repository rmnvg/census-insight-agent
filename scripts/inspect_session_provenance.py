import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.app.agent.persistence import TraceStore
from executor.policy import validate_code

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


async def inspect_session(
    session_id: str,
    workspace: Path,
    *,
    code_only: bool = False,
    include_code: bool = False,
) -> dict[str, object]:
    checkpoints: list[dict[str, object]] = []
    run_ids: set[str] = set()
    async with AsyncSqliteSaver.from_conn_string(str(workspace / "checkpoints.sqlite")) as saver:
        async for item in saver.alist({"configurable": {"thread_id": session_id}}):
            values = item.checkpoint.get("channel_values", {})
            run_id = values.get("run_id")
            if isinstance(run_id, str):
                run_ids.add(run_id)
            evidence_summary: dict[str, object] = {}
            for key in () if code_only else ("retrieved_evidence", "selected_evidence"):
                evidence = values.get(key, [])
                if not isinstance(evidence, list):
                    continue
                evidence_summary[key] = [
                    {
                        "chunk_id": row.get("chunk_id"),
                        "document_id": row.get("document_id"),
                        "page_number": row.get("page_number"),
                        "checksum_present": bool(row.get("source_checksum")),
                        "checksum_valid": bool(
                            isinstance(row.get("source_checksum"), str)
                            and _SHA256.fullmatch(row["source_checksum"])
                        ),
                    }
                    for row in evidence
                    if isinstance(row, dict)
                ]
            generated_code = values.get("generated_code")
            code_summary: dict[str, object] = {}
            if isinstance(generated_code, str):
                policy = validate_code(generated_code)
                code_summary = {
                    "generated_code_sha256": hashlib.sha256(generated_code.encode()).hexdigest(),
                    "generated_code_policy_valid": policy.valid,
                    "generated_code_policy_errors": list(policy.errors),
                }
                if include_code:
                    code_summary["generated_code"] = generated_code
            checkpoint = {
                "checkpoint_id": item.config["configurable"]["checkpoint_id"],
                "run_id": run_id,
                **evidence_summary,
                **code_summary,
            }
            if not code_only or code_summary:
                checkpoints.append(checkpoint)
    traces = TraceStore(workspace)
    return {
        "session_id": session_id,
        "checkpoint_count": len(checkpoints),
        "run_ids": sorted(run_ids),
        "persisted_trace_run_ids": sorted(
            run_id for run_id in run_ids if traces.read(run_id) is not None
        ),
        "checkpoints": checkpoints,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect checksum presence in saved agent state")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument("--include-code", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(
                inspect_session(
                    args.session_id,
                    args.workspace,
                    code_only=args.code_only,
                    include_code=args.include_code,
                )
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
