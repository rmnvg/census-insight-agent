"""Publish a trust scorecard to Langfuse as a dataset experiment run.

Each benchmark case becomes a dataset item with its hand-verified expectation, and each scorecard
becomes one experiment run. Per-case scores (pass, ungrounded claims, latency) and run-level
scores (answer accuracy, refusal precision and recall, grounded-claim rate) let runs for different
models or prompts be compared side by side in Langfuse.

Scores come from the independent checks in `backend/app/trust.py`, never from an LLM judge:
numeric grounding is decided by code.
"""

from typing import TYPE_CHECKING, Any

from backend.app.trust import CaseResult, Scorecard, TrustCase, run_metrics

if TYPE_CHECKING:
    from langfuse import Evaluation, Langfuse

DATASET = "census-trust-benchmark"


def _item_id(case_id: str) -> str:
    return f"trust-{case_id}"


def _expected(case: TrustCase) -> dict[str, Any]:
    return {
        "refuse": case.expect_refusal,
        "facts": [fact.model_dump() for fact in case.expected_facts],
        "computed": case.expected_derived,
        "artifact": case.expected_artifact,
    }


def item_evaluations(result: CaseResult) -> list["Evaluation"]:
    from langfuse import Evaluation

    return [
        Evaluation(name="passed", value=result.passed, data_type="BOOLEAN"),
        Evaluation(name="verdict", value=result.verdict, data_type="CATEGORICAL"),
        Evaluation(name="outcome", value=result.outcome, data_type="CATEGORICAL"),
        Evaluation(name="misattributed_claims", value=len(result.misattributed_claims)),
        Evaluation(name="ungrounded_claims", value=len(result.ungrounded_claims)),
        Evaluation(name="arithmetic_errors", value=len(result.arithmetic_errors)),
        Evaluation(name="latency_seconds", value=result.latency_seconds),
    ]


def publish_scorecard(
    client: "Langfuse", cases: list[TrustCase], card: Scorecard, *, run_name: str
) -> str | None:
    """Upsert the benchmark dataset, record this scorecard as a run, and return its URL."""
    from langfuse import Evaluation

    client.create_dataset(
        name=DATASET,
        description="Hand-verified Census 2011 questions with expected values or refusals.",
    )
    by_case = {case.case_id: case for case in cases}
    scored = [result for result in card.results if result.case_id in by_case]
    for case_id in dict.fromkeys(result.case_id for result in scored):
        case = by_case[case_id]
        client.create_dataset_item(
            dataset_name=DATASET,
            id=_item_id(case_id),
            input={"question": case.question, "setup_turns": case.setup_turns},
            expected_output=_expected(case),
            metadata={"case_id": case_id, "category": case.category, "source": case.source_note},
        )
    wanted = {_item_id(result.case_id) for result in scored}
    items = [item for item in client.get_dataset(DATASET).items if item.id in wanted]
    repeats = sorted({result.repeat for result in scored})
    url: str | None = None
    # A repeated benchmark becomes one experiment run per repeat, so runs compare item by item.
    for repeat in repeats:
        results = {result.case_id: result for result in scored if result.repeat == repeat}

        def task(*, item: Any, _results: dict[str, CaseResult] = results, **_: Any) -> Any:
            result = _results[item.metadata["case_id"]]
            # The run already happened; this replays its recorded outcome into the experiment.
            return {
                "outcome": result.outcome,
                "answer_excerpt": result.answer_excerpt,
                "observed_values": result.observed_values,
                "agent_run_id": result.trace_id,
                "case_id": result.case_id,
                "verdict": result.verdict,
                "repeat": result.repeat,
            }

        def evaluate(
            *, output: dict[str, Any], _results: dict[str, CaseResult] = results, **_: Any
        ) -> list[Evaluation]:
            return item_evaluations(_results[output["case_id"]])

        def evaluate_run(**_: Any) -> list[Evaluation]:
            return [Evaluation(name=name, value=value) for name, value in run_metrics(card).items()]

        experiment = client.run_experiment(
            name=DATASET,
            run_name=run_name if len(repeats) == 1 else f"{run_name} · run {repeat}",
            description=f"Trust scorecard for {card.model}, {card.generated_at.isoformat()}",
            data=[item for item in items if item.metadata["case_id"] in results],
            task=task,
            evaluators=[evaluate],
            run_evaluators=[evaluate_run],
            metadata={
                "model": card.model,
                "generated_at": card.generated_at.isoformat(),
                "repeat": str(repeat),
            },
            max_concurrency=4,
        )
        url = experiment.dataset_run_url or url
    client.flush()
    return url
