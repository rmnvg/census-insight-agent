#!/usr/bin/env python3
"""Publish a saved trust scorecard to Langfuse as a dataset experiment run.

Makes no model calls: it records an existing scorecard (for example the committed
`evals/trust-scorecard.json`, or a fresh `make trust-benchmark` output) so runs across models and
prompt changes can be compared in Langfuse. Reads LANGFUSE_BASE_URL, LANGFUSE_PUBLIC_KEY, and
LANGFUSE_SECRET_KEY from the environment.

    uv run --frozen python scripts/publish_trust_scorecard.py --scorecard evals/trust-scorecard.json
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.trust import Scorecard, TrustCase, run_metrics  # noqa: E402
from backend.app.trust_publish import publish_scorecard  # noqa: E402

REQUIRED = ("LANGFUSE_BASE_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


def default_run_name(card: Scorecard) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    stamp = card.generated_at.strftime("%Y-%m-%d %H:%M")
    return f"{card.model} {stamp}Z" + (f" @{revision}" if revision else "")


def publish(scorecard: Path, cases_path: Path, run_name: str | None) -> int:
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        print(f"Set {', '.join(missing)} to publish to Langfuse.")
        return 2
    from langfuse import Langfuse

    card = Scorecard.model_validate_json(scorecard.read_text(encoding="utf-8"))
    cases = [
        TrustCase.model_validate(item)
        for item in json.loads(cases_path.read_text(encoding="utf-8"))
    ]
    client = Langfuse()
    url = publish_scorecard(client, cases, card, run_name=run_name or default_run_name(card))
    client.shutdown()
    metrics = " · ".join(f"{name} {value:g}" for name, value in run_metrics(card).items())
    print(f"Published {len(card.results)} cases: {metrics}")
    if url:
        print(url)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, default=Path("evals/trust-scorecard.json"))
    parser.add_argument("--cases", type=Path, default=Path("evals/trust_benchmark.json"))
    parser.add_argument("--run-name", help="Defaults to '<model> <generated_at> @<git sha>'")
    args = parser.parse_args()
    return publish(args.scorecard, args.cases, args.run_name)


if __name__ == "__main__":
    raise SystemExit(main())
