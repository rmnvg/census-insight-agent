#!/usr/bin/env python3
"""Re-score recorded benchmark responses with the current scorer. No model calls.

`scripts/trust_benchmark.py --responses-out` records every raw API response; this replays them
through `backend/app/trust.py`, so a scorer change can be checked against real answers for free.

    uv run --frozen python scripts/rescore_trust_benchmark.py \
        --responses run.jsonl --output card.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.trust import Scorecard, TrustCase, score_case, summarize  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("evals/trust_benchmark.json"))
    parser.add_argument("--model", default="recorded")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--scorecard",
        type=Path,
        help="The live run's scorecard: keeps its timestamp, model, latencies, retries, error "
        "codes, and trace IDs, which responses alone do not record.",
    )
    args = parser.parse_args()
    live = (
        Scorecard.model_validate_json(args.scorecard.read_text(encoding="utf-8"))
        if args.scorecard
        else None
    )
    recorded = {(item.case_id, item.repeat): item for item in live.results} if live else {}
    cases = {
        item["case_id"]: TrustCase.model_validate(item)
        for item in json.loads(args.cases.read_text(encoding="utf-8"))
    }
    results = []
    for line in args.responses.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        case = cases.get(record["case_id"])
        if case is None:
            continue
        repeat = record.get("repeat", 1)
        original = recorded.get((case.case_id, repeat))
        result = score_case(
            case,
            record["response"],
            latency_seconds=original.latency_seconds if original else 0.0,
            attempts=original.attempts if original else 1,
            error_code=original.error_code if original else None,
            trace_id=original.trace_id if original else None,
        )
        results.append(result.model_copy(update={"repeat": repeat}))
        detail = "; ".join(result.failures[:3])
        print(f"{result.verdict:<14} {case.case_id:<24} {detail}")
    card = summarize(results, live.model if live else args.model)
    if live:
        card = card.model_copy(update={"generated_at": live.generated_at})
    print(
        f"\n{card.cases_passed}/{card.cases_total} passed · wrong {card.wrong_answers} · "
        f"unsupported {card.unsupported_answers} · incomplete {card.incomplete_answers} · "
        f"false refusals {card.false_refusals} · misattributed claims {card.misattributed_claims}"
    )
    if args.output:
        args.output.write_text(card.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
