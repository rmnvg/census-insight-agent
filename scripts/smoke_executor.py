import argparse
import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from uuid import uuid4

from backend.app.execution.client import ExecutionQueueClient
from backend.app.execution.contracts import ExecutionRequest, ExecutionResult, ExpectedArtifact
from executor.worker import Worker

SOURCE = {
    "source_record_id": "source-1",
    "row_id": "karnataka",
    "field": "literacy_percent",
    "raw_value": "75.36",
    "normalized_numeric_value": 75.36,
    "unit": "percent",
    "document_id": "census-2011-karnataka-pca-highlights",
    "page_number": 50,
    "chunk_id": "c1663038-700a-5623-94a1-1ca1d3a425c0",
    "exact_supporting_quote": "KARNATAKA literacy rate 75.36",
    "source_checksum": "a" * 64,
}
INPUT = {
    "dataset": {
        "title": "Karnataka literacy",
        "task_type": "artifact_chart",
        "rows": [{"row_id": "karnataka", "literacy_percent": 75.36}],
        "columns": ["row_id", "literacy_percent"],
        "units": {"literacy_percent": "percent"},
        "source_records": [SOURCE],
        "computed_values": [],
        "requested_output": "bar chart",
    },
    "source_manifest": {
        "dataset_title": "Karnataka literacy",
        "source_records": [SOURCE],
        "computed_values": [],
    },
}

CHART_OUTPUTS = [
    ExpectedArtifact(
        artifact_type="chart",
        title="Chart",
        filename="chart.png",
        media_type="image/png",
    ),
    ExpectedArtifact(
        artifact_type="data",
        title="Data",
        filename="plotted-data.csv",
        media_type="text/csv",
        expected_columns=["row_id", "literacy_percent"],
    ),
    ExpectedArtifact(
        artifact_type="manifest",
        title="Sources",
        filename="source-manifest.json",
        media_type="application/json",
    ),
]


def expected_table() -> list[ExpectedArtifact]:
    return [
        ExpectedArtifact(
            artifact_type="data",
            title="Table data",
            filename="table.csv",
            media_type="text/csv",
            expected_columns=["row_id", "literacy_percent"],
        ),
        ExpectedArtifact(
            artifact_type="table",
            title="Table",
            filename="table.md",
            media_type="text/markdown",
        ),
        CHART_OUTPUTS[-1],
    ]


SUCCESS_CODE = """
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

payload = json.loads(Path("input.json").read_text(encoding="utf-8"))
frame = pd.DataFrame(payload["dataset"]["rows"], columns=payload["dataset"]["columns"])
frame.to_csv(Path("output/plotted-data.csv"), index=False)
Path("output/source-manifest.json").write_text(
    json.dumps(payload["source_manifest"], indent=2) + "\\n", encoding="utf-8"
)
plt.style.use("seaborn-v0_8-whitegrid")
figure, axis = plt.subplots(figsize=(6, 4))
axis.bar(frame["row_id"], frame["literacy_percent"], color="#3366cc")
axis.set_title(payload["dataset"]["title"])
axis.set_xlabel("Region")
axis.set_ylabel("Literacy rate (percent)")
axis.set_ylim(bottom=0)
figure.tight_layout()
figure.savefig(Path("output/chart.png"), dpi=140)
print("artifact created")
""".strip()

TABLE_CODE = (
    SUCCESS_CODE.replace("plotted-data.csv", "table.csv")
    .replace(
        'plt.style.use("seaborn-v0_8-whitegrid")',
        'Path("output/table.md").write_text('
        '"| Region | Literacy rate (percent) |\\n|---|---:|\\n| Karnataka | 75.36 |\\n", '
        'encoding="utf-8")\nplt.style.use("seaborn-v0_8-whitegrid")',
    )
    .replace(
        'figure.savefig(Path("output/chart.png"), dpi=140)',
        "plt.close(figure)",
    )
)


def request(code: str, outputs: list[ExpectedArtifact], timeout: float = 5) -> ExecutionRequest:
    return ExecutionRequest(
        job_id=str(uuid4()),
        session_id=str(uuid4()),
        run_id=str(uuid4()),
        code=code,
        code_sha256=hashlib.sha256(code.encode()).hexdigest(),
        input_data=INPUT,
        expected_artifacts=outputs,
        timeout_seconds=timeout,
    )


async def execute(root: Path, value: ExecutionRequest) -> ExecutionResult:
    client = ExecutionQueueClient(root)
    client.submit(value)
    Worker(root).run_once()
    return await client.wait(value.job_id, 2)


async def smoke(root: Path) -> dict[str, object]:
    successful = await execute(root, request(SUCCESS_CODE, CHART_OUTPUTS))
    runtime = await execute(root, request("raise RuntimeError('expected failure')", CHART_OUTPUTS))
    timeout = await execute(root, request("while True:\n    pass", CHART_OUTPUTS, 0.2))
    policy = await execute(root, request("import socket\nprint('unsafe')", CHART_OUTPUTS))
    invalid_code = SUCCESS_CODE.replace(
        'figure.savefig(Path("output/chart.png"), dpi=140)',
        'Path("output/chart.png").write_text("not png", encoding="utf-8")',
    )
    invalid = await execute(root, request(invalid_code, CHART_OUTPUTS))
    table = await execute(root, request(TABLE_CODE, expected_table()))
    checks = {
        "success": successful.status == "succeeded" and "artifact created" in successful.stdout,
        "runtime_error": runtime.error_code == "EXECUTION_FAILED"
        and "RuntimeError" in runtime.stderr,
        "timeout": timeout.error_code == "EXECUTION_TIMEOUT" and timeout.timed_out,
        "policy_violation": policy.error_code == "CODE_POLICY_VIOLATION"
        and policy.exit_code is None,
        "invalid_artifact": invalid.error_code == "INVALID_ARTIFACT",
        "table": table.status == "succeeded"
        and any(item.filename == "table.csv" for item in table.artifacts),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "successful_artifacts": [item.model_dump(mode="json") for item in successful.artifacts],
        "runtime_stderr_captured": bool(runtime.stderr),
        "timeout_terminated": timeout.timed_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline isolated executor smoke suite")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.root:
        args.root.mkdir(parents=True, exist_ok=True)
        report = asyncio.run(smoke(args.root))
    else:
        with tempfile.TemporaryDirectory() as temporary:
            report = asyncio.run(smoke(Path(temporary)))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
