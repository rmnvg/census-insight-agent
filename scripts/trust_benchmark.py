#!/usr/bin/env python3
"""Run the trust benchmark against a live API and write the scorecard the UI displays.

Billable: each case is a full agent turn on Vertex Gemini. Requires --allow-paid-calls.
`--repeat N` asks every case N times and reports cases whose runs disagree; `--responses-out`
records raw responses so `scripts/rescore_trust_benchmark.py` can re-score them for free.

    uv run --frozen python scripts/trust_benchmark.py --allow-paid-calls \
        --output data/processed/trust-scorecard.json

Copy the output to evals/trust-scorecard.json to ship it with the repository.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.trust import CaseResult, TrustCase, score_case, summarize  # noqa: E402

RETRYABLE = {
    "BACKEND_UNREACHABLE",
    "MODEL_TIMEOUT",
    "EVIDENCE_ASSESSMENT_TIMEOUT",
    "AGENT_REQUEST_TIMEOUT",
    "MODEL_UNAVAILABLE",
    "MODEL_RATE_LIMITED",
}


def post(base: str, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return 0, {"error_code": "BACKEND_UNREACHABLE"}


def converse(base: str, case: TrustCase) -> tuple[int, dict[str, Any]]:
    """Ask a case in a fresh session: any setup turns first, then the scored question."""
    status, session = post(base, "/sessions", {})
    if status != 200:
        return status, session
    for message in [*case.setup_turns, case.question]:
        status, body = post(
            base, "/chat", {"session_id": session["session_id"], "message": message}
        )
        if status != 200:
            return status, body
    return status, body


def run_case(base: str, case: TrustCase) -> tuple[CaseResult, dict[str, Any] | None]:
    attempts, started = 0, time.monotonic()
    while True:
        attempts += 1
        status, body = converse(base, case)
        if status == 200:
            result = score_case(
                case, body, latency_seconds=time.monotonic() - started, attempts=attempts
            )
            return result, body
        code = body.get("error_code") or f"HTTP_{status}"
        # One retry for transient provider errors only; the attempt count is reported.
        if code not in RETRYABLE or attempts >= 2:
            result = score_case(
                case,
                None,
                latency_seconds=time.monotonic() - started,
                attempts=attempts,
                error_code=code,
                trace_id=body.get("trace_id"),
            )
            return result, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=Path, default=Path("evals/trust_benchmark.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/trust-scorecard.json"))
    parser.add_argument("--model", default=os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--only", nargs="*", help="Run only these case IDs")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Ask every case this many times; cases with mixed outcomes are reported as unstable.",
    )
    parser.add_argument(
        "--responses-out",
        type=Path,
        help="Also save each raw API response (JSON lines) so the run can be re-scored offline.",
    )
    parser.add_argument("--allow-paid-calls", action="store_true")
    parser.add_argument(
        "--publish-langfuse",
        action="store_true",
        help="Also record the run as a Langfuse dataset experiment (needs LANGFUSE_* settings).",
    )
    args = parser.parse_args()
    if not args.allow_paid_calls:
        print(
            "Refusing to run: every case is a billable Vertex agent turn. Pass --allow-paid-calls."
        )
        return 2
    cases = [
        TrustCase.model_validate(item)
        for item in json.loads(args.cases.read_text(encoding="utf-8"))
    ]
    if args.only:
        cases = [case for case in cases if case.case_id in set(args.only)]
    results = []
    recorded = args.responses_out.open("w", encoding="utf-8") if args.responses_out else None
    for repeat in range(1, args.repeat + 1):
        for case in cases:
            result, response = run_case(args.base_url.rstrip("/"), case)
            result = result.model_copy(update={"repeat": repeat})
            results.append(result)
            if recorded is not None:
                line = {"case_id": case.case_id, "repeat": repeat, "response": response}
                recorded.write(json.dumps(line) + "\n")
                recorded.flush()
            mark = "PASS" if result.passed else "FAIL"
            detail = result.error_code or "; ".join(result.failures[:2]) or ""
            line_text = (
                f"{mark} {case.case_id:<22} {result.outcome:<9} {result.latency_seconds:>6.1f}s"
            )
            print(f"{line_text} {detail}", flush=True)
    if recorded is not None:
        recorded.close()
    card = summarize(results, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(card.model_dump_json(indent=2) + "\n", encoding="utf-8")
    summary = [
        f"{card.cases_passed}/{card.cases_total} passed",
        f"answerable {card.answerable_passed}/{card.answerable_total}",
        f"refusals {card.refusal_passed}/{card.refusal_total}",
        f"wrong answers {card.wrong_answers}",
        f"unsupported {card.unsupported_answers}",
        f"unnecessary refusals {card.false_refusals}",
        f"incomplete {card.incomplete_answers}",
        f"median {card.median_latency_seconds}s",
    ]
    if card.unstable_cases:
        summary.append(f"unstable: {', '.join(card.unstable_cases)}")
    print("\n" + " · ".join(summary) + f" -> {args.output}")
    if args.publish_langfuse:
        from scripts.publish_trust_scorecard import publish

        return publish(args.output, args.cases, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
