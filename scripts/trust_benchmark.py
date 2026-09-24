#!/usr/bin/env python3
"""Run the trust benchmark against a live API and write the scorecard the UI displays.

Billable: each case is a full agent turn on Vertex Gemini. Requires --allow-paid-calls.

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

from backend.app.trust import TrustCase, score_case, summarize  # noqa: E402

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


def run_case(base: str, case: TrustCase) -> Any:
    attempts, started = 0, time.monotonic()
    while True:
        attempts += 1
        status, session = post(base, "/sessions", {})
        if status == 200:
            status, body = post(
                base, "/chat", {"session_id": session["session_id"], "message": case.question}
            )
        else:
            body = session
        if status == 200:
            return score_case(
                case, body, latency_seconds=time.monotonic() - started, attempts=attempts
            )
        code = body.get("error_code") or f"HTTP_{status}"
        # One retry for transient provider errors only; the attempt count is reported.
        if code not in RETRYABLE or attempts >= 2:
            return score_case(
                case,
                None,
                latency_seconds=time.monotonic() - started,
                attempts=attempts,
                error_code=code,
                trace_id=body.get("trace_id"),
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=Path, default=Path("evals/trust_benchmark.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/trust-scorecard.json"))
    parser.add_argument("--model", default=os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--only", nargs="*", help="Run only these case IDs")
    parser.add_argument("--allow-paid-calls", action="store_true")
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
    for case in cases:
        result = run_case(args.base_url.rstrip("/"), case)
        results.append(result)
        mark = "PASS" if result.passed else "FAIL"
        detail = result.error_code or ", ".join(result.missing + result.ungrounded_claims[:1]) or ""
        line = f"{mark} {case.case_id:<20} {result.outcome:<9} {result.latency_seconds:>6.1f}s"
        print(f"{line} {detail}", flush=True)
    card = summarize(results, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(card.model_dump_json(indent=2) + "\n", encoding="utf-8")
    summary = [
        f"{card.cases_passed}/{card.cases_total} passed",
        f"answerable {card.answerable_passed}/{card.answerable_total}",
        f"refusals {card.refusal_passed}/{card.refusal_total}",
        f"ungrounded claims {card.ungrounded_claims}",
        f"wrong answers {card.wrong_answers}",
        f"median {card.median_latency_seconds}s",
    ]
    print("\n" + " · ".join(summary) + f" -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
