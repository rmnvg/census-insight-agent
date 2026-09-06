import argparse
import json
from pathlib import Path

from backend.app.execution.contracts import ArtifactDataset, ArtifactDatasetProposal
from backend.app.execution.schema_complexity import assert_proposal_schema_safe, schema_complexity


def main() -> int:
    parser = argparse.ArgumentParser(description="Report artifact structured-output complexity")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "rejected_llm_facing_schema": schema_complexity(ArtifactDataset),
        "replacement_llm_facing_schema": assert_proposal_schema_safe(ArtifactDatasetProposal),
        "regression_guard_note": (
            "These conservative local checks reduce schema complexity but do not guarantee "
            "acceptance by a model provider."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
