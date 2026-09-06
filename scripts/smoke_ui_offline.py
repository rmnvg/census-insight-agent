import argparse
import base64
import json
from pathlib import Path

from frontend.models import ChatResponse, OperationalApiError, TraceResponse
from frontend.view_models import (
    citations_for_claim,
    parse_csv,
    sanitize_trace,
    validate_png,
    validate_source_manifest,
)


def run_smoke(fixture_path: Path) -> dict[str, object]:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    lookup = ChatResponse.model_validate(fixture["lookup"])
    comparison = ChatResponse.model_validate(fixture["comparison"])
    refusal = ChatResponse.model_validate(fixture["refusal"])
    error = OperationalApiError.model_validate(fixture["operational_error"])
    trace = TraceResponse.model_validate(fixture["trace"])
    chart = fixture["chart"]
    table = fixture["table"]
    validate_png(base64.b64decode(chart["png_base64"]))
    chart_frame = parse_csv(chart["csv"].encode())
    table_frame = parse_csv(table["csv"].encode())
    validate_source_manifest(json.dumps(chart["manifest"]).encode())
    validate_source_manifest(json.dumps(table["manifest"]).encode())
    checks = {
        "lookup": bool(citations_for_claim(lookup.claims[0], lookup.citations)),
        "comparison": len(comparison.claims) == 3 and len(comparison.citations) == 2,
        "derived_claim_has_both_sources": len(comparison.claims[-1].citation_ids) == 2,
        "chart": chart_frame.shape == (2, 4),
        "table": table_frame.shape == (2, 4) and bool(table["markdown"]),
        "refusal": refusal.refusal and not refusal.claims,
        "operational_error": error.error_code == "MODEL_OUTPUT_INVALID",
        "trace": len(sanitize_trace(trace)) == 7,
    }
    return {"passed": all(checks.values()), "checks": checks, "external_requests": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate safe offline UI fixtures")
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("frontend/tests/fixtures/ui-smoke.json"),
    )
    args = parser.parse_args()
    result = run_smoke(args.fixtures)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
