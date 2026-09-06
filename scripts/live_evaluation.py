#!/usr/bin/env python3
"""Manual, paid end-to-end evaluation; never invoked by automated checks."""

import argparse
import csv
import io
import json
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class EvaluationFailure(RuntimeError):
    pass


def request(
    base_url: str, method: str, path: str, body: dict[str, str] | None = None
) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{base_url}{path}", data=data, method=method, headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=210) as response:
            raw = response.read()
            content_type = response.headers.get("content-type", "")
            if "json" in content_type:
                return response.status, json.loads(raw)
            return response.status, raw
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {"error_code": "NON_JSON_HTTP_ERROR", "message": "Non-JSON HTTP error"}
        return error.code, payload


def validated_trace_citations(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    trusted: dict[str, dict[str, Any]] = {}
    for event in trace.get("events", []):
        if event.get("event") == "validated_citations":
            for citation in event.get("details", {}).get("citations", []):
                trusted[citation["citation_id"]] = citation
    return trusted


def assert_grounded(result: dict[str, Any], trace: dict[str, Any]) -> None:
    if result.get("refusal") or not result.get("claims") or not result.get("citations"):
        raise EvaluationFailure("Answerable case lacks claims or citations")
    trusted = validated_trace_citations(trace)
    for citation in result["citations"]:
        if not all(citation.get(key) for key in ("document_id", "page_number", "snippet")):
            raise EvaluationFailure("Citation lacks document, physical page, or quote")
        if citation.get("page_number", 0) < 1:
            raise EvaluationFailure("Citation page is not one-based")
        saved = trusted.get(citation.get("citation_id"))
        if saved is None or any(
            citation.get(key) != saved.get(key)
            for key in ("document_id", "page_number", "chunk_id", "snippet")
        ):
            raise EvaluationFailure("Public citation is not current-run validated evidence")
    cited = {item for claim in result["claims"] for item in claim.get("citation_ids", [])}
    if not cited or not cited <= {item["citation_id"] for item in result["citations"]}:
        raise EvaluationFailure("Claim/citation mapping is incomplete")


def artifact_files(base_url: str, session_id: str, artifact: dict[str, Any]) -> None:
    names = (
        ("chart.png", "plotted-data.csv", "source-manifest.json")
        if artifact["artifact_type"] == "chart"
        else ("table.csv", "table.md", "source-manifest.json")
    )
    for name in names:
        status, content = request(
            base_url,
            "GET",
            f"/sessions/{session_id}/artifacts/{artifact['artifact_id']}/files/{name}",
        )
        if status != 200 or not isinstance(content, bytes) or not content:
            raise EvaluationFailure(f"Missing validated artifact file: {name}")
        if name.endswith(".png") and not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise EvaluationFailure("Chart output is not a PNG")
        if name.endswith(".csv") and not list(csv.reader(io.StringIO(content.decode()))):
            raise EvaluationFailure("Artifact CSV is empty")
        if name == "source-manifest.json":
            manifest = json.loads(content)
            if not manifest.get("source_records"):
                raise EvaluationFailure("Artifact manifest lacks source lineage")


def run_case(
    base_url: str,
    session_id: str,
    case_id: str,
    message: str,
    trace_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    print(f"RUN {case_id}")
    status, result = request(
        base_url, "POST", "/chat", {"session_id": session_id, "message": message}
    )
    if status in {500, 422}:
        raise EvaluationFailure(f"{case_id} returned unexpected HTTP {status}")
    if status != 200:
        code = (
            result.get("error_code", "UNTYPED_ERROR")
            if isinstance(result, dict)
            else "UNTYPED_ERROR"
        )
        raise EvaluationFailure(f"{case_id} returned HTTP {status} ({code})")
    trace_id = result.get("trace_id")
    if not isinstance(trace_id, str) or trace_id in trace_ids:
        raise EvaluationFailure("Trace ID is missing or duplicated")
    trace_ids.add(trace_id)
    trace_status, trace = request(base_url, "GET", f"/runs/{trace_id}/trace")
    if trace_status != 200 or trace.get("run_id") != trace_id:
        raise EvaluationFailure("Trace is not retrievable")
    summary = {
        "case_id": case_id,
        "http_status": status,
        "trace_id": trace_id,
        "refusal": bool(result.get("refusal")),
        "claim_count": len(result.get("claims", [])),
        "citation_count": len(result.get("citations", [])),
        "artifact_count": len(result.get("artifacts", [])),
    }
    print(json.dumps(summary))
    return result, trace, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-paid-calls", action="store_true")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.allow_paid_calls:
        print("Refusing to run: pass --allow-paid-calls to acknowledge Vertex charges.")
        return 2
    output = args.output or Path(
        f"evals/reports/live-evaluation-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    _, session = request(args.base_url, "POST", "/sessions")
    session_id = session["session_id"]
    trace_ids: set[str] = set()
    summaries: list[dict[str, Any]] = []
    cases = [
        ("karnataka-literacy", "What was Karnataka's literacy rate in 2011?", "grounded"),
        ("odisha-comparison", "How does that compare with Odisha?", "comparison"),
        ("source-pages", "Which source pages support those values?", "comparison"),
        ("france-refusal", "What was France's unemployment rate in 2011?", "refusal"),
        (
            "chart",
            "Create a bar chart comparing the 2011 total persons literacy rates "
            "for Karnataka and Odisha.",
            "chart",
        ),
        (
            "table",
            "Create a table comparing the 2011 total, rural, and urban literacy rates "
            "for Karnataka and Odisha.",
            "table",
        ),
        ("summary", "Summarize the key population findings for Odisha.", "grounded"),
        (
            "inconsistency",
            "Analyze whether the supplied reports contain inconsistent literacy-rate evidence "
            "for Karnataka.",
            "safe",
        ),
        ("unanswerable", "What color represents Mysuru on an excluded thematic map?", "refusal"),
    ]
    try:
        for case_id, message, expectation in cases:
            result, trace, summary = run_case(
                args.base_url, session_id, case_id, message, trace_ids
            )
            if expectation in {"grounded", "comparison", "chart", "table"}:
                assert_grounded(result, trace)
            if expectation == "comparison":
                documents = {citation["document_id"] for citation in result["citations"]}
                if not any("karnataka" in value for value in documents) or not any(
                    "odisha" in value for value in documents
                ):
                    raise EvaluationFailure(f"{case_id} is not supported by both documents")
                for claim in result["claims"]:
                    if claim.get("derivation") and len(claim.get("citation_ids", [])) < 2:
                        raise EvaluationFailure("Derived comparison does not cite both inputs")
            if expectation == "refusal" and (not result.get("refusal") or result.get("claims")):
                raise EvaluationFailure(f"{case_id} did not safely refuse")
            if expectation == "safe" and not result.get("refusal"):
                assert_grounded(result, trace)
            if expectation in {"chart", "table"}:
                artifacts = [
                    item
                    for item in result.get("artifacts", [])
                    if item.get("artifact_type") == expectation and "artifact_id" in item
                ]
                if not artifacts:
                    raise EvaluationFailure(f"{case_id} produced no validated artifact")
                artifact_files(args.base_url, session_id, artifacts[0])
            summaries.append(summary)
        _, isolated = request(args.base_url, "POST", "/sessions")
        result, _, summary = run_case(
            args.base_url,
            isolated["session_id"],
            "new-session-isolation",
            "Which source pages support those values?",
            trace_ids,
        )
        if not result.get("refusal") or result.get("claims"):
            raise EvaluationFailure("New session inherited prior validated claims")
        summaries.append(summary)
    except (EvaluationFailure, KeyError, TypeError, ValueError) as error:
        report = {"passed": False, "error": str(error), "cases": summaries}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"FAIL: {error}; report: {output}", file=sys.stderr)
        return 1
    report = {"passed": True, "session_id": session_id, "cases": summaries}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(summaries)} live cases; report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
