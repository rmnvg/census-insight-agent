"""Scorecard metrics and the Langfuse experiment payload, without a Langfuse server."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from backend.app.trust import Scorecard, TrustCase, run_metrics
from backend.app.trust_publish import DATASET, publish_scorecard

SHIPPED = Path("evals/trust-scorecard.json")
CASES = Path("evals/trust_benchmark.json")


def shipped() -> tuple[list[TrustCase], Scorecard]:
    cases = [TrustCase.model_validate(item) for item in json.loads(CASES.read_text("utf-8"))]
    return cases, Scorecard.model_validate_json(SHIPPED.read_text("utf-8"))


def test_shipped_scorecard_is_a_repeated_v2_run_with_no_wrong_answers() -> None:
    cases, card = shipped()
    assert card.scorer_version == 2 and card.repeats == 2
    assert card.cases_total == 2 * len(cases)
    assert (card.wrong_answers, card.unsupported_answers, card.misattributed_claims) == (0, 0, 0)


def test_run_metrics_follow_their_definitions() -> None:
    _, card = shipped()
    metrics = run_metrics(card)
    refused = [item for item in card.results if item.outcome == "refused"]
    correct = sum(item.passed for item in refused if item.category == "refusal")
    assert metrics["pass_rate"] == round(card.cases_passed / card.cases_total, 4)
    assert metrics["refusal_recall"] == round(card.refusal_passed / card.refusal_total, 4)
    assert metrics["refusal_precision"] == round(correct / len(refused), 4)
    assert metrics["false_refusal_rate"] == round(card.false_refusals / card.answerable_total, 4)
    assert metrics["grounded_claim_rate"] == 1.0
    assert metrics["unstable_cases"] == len(card.unstable_cases)
    assert metrics["p95_latency_seconds"] >= metrics["median_latency_seconds"]


def test_false_refusal_lowers_precision_not_recall() -> None:
    _, card = shipped()
    before = run_metrics(card)
    results = list(card.results)
    index = next(i for i, item in enumerate(results) if item.category == "lookup" and item.passed)
    results[index] = results[index].model_copy(
        update={"outcome": "refused", "passed": False, "verdict": "false_refusal"}
    )
    after = run_metrics(
        card.model_copy(
            update={
                "results": results,
                "cases_passed": card.cases_passed - 1,
                "answerable_passed": card.answerable_passed - 1,
                "false_refusals": card.false_refusals + 1,
            }
        )
    )
    assert after["refusal_recall"] == before["refusal_recall"]
    assert after["refusal_precision"] < before["refusal_precision"]
    assert after["false_refusal_rate"] > before["false_refusal_rate"]


class FakeLangfuse:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}
        self.experiments: list[dict[str, Any]] = []
        self.flushed = False

    def create_dataset(self, **kwargs: Any) -> None:
        assert kwargs["name"] == DATASET

    def create_dataset_item(self, **kwargs: Any) -> None:
        self.items[kwargs["id"]] = kwargs

    def get_dataset(self, name: str) -> Any:
        stale = SimpleNamespace(id="trust-removed-case", metadata={"case_id": "removed-case"})
        current = [
            SimpleNamespace(id=key, metadata=value["metadata"]) for key, value in self.items.items()
        ]
        return SimpleNamespace(items=[stale, *current])

    def run_experiment(self, **kwargs: Any) -> Any:
        self.experiments.append(kwargs)
        return SimpleNamespace(dataset_run_url=f"http://langfuse.local/run/{len(self.experiments)}")

    def flush(self) -> None:
        self.flushed = True


def test_publish_upserts_items_and_records_one_run_per_repeat() -> None:
    cases, card = shipped()
    client = FakeLangfuse()
    url = publish_scorecard(client, cases, card, run_name="gemini-2.5-flash test")  # type: ignore[arg-type]

    assert url == "http://langfuse.local/run/2" and client.flushed
    assert len(client.items) == len(cases)
    assert client.items["trust-refuse-gdp"]["expected_output"]["refuse"] is True
    [fact] = client.items["trust-ka-literacy"]["expected_output"]["facts"]
    assert (fact["region"], fact["value"], fact["year"]) == ("Karnataka", 75.36, 2011)
    assert client.items["trust-followup-urban"]["input"]["setup_turns"]

    assert [run["run_name"] for run in client.experiments] == [
        "gemini-2.5-flash test · run 1",
        "gemini-2.5-flash test · run 2",
    ]
    first = client.experiments[0]
    # Items in the dataset but not in this scorecard are not part of the run.
    assert {item.id for item in first["data"]} == set(client.items)
    item = next(item for item in first["data"] if item.id == "trust-ka-literacy")
    output = first["task"](item=item)
    assert output["outcome"] == "answered" and 75.36 in output["observed_values"]
    [evaluate] = first["evaluators"]
    scores = {evaluation.name: evaluation.value for evaluation in evaluate(output=output)}
    assert scores["passed"] is True and scores["ungrounded_claims"] == 0
    assert scores["verdict"] == "passed" and scores["misattributed_claims"] == 0
    [evaluate_run] = first["run_evaluators"]
    run_scores = {evaluation.name: evaluation.value for evaluation in evaluate_run(item_results=[])}
    assert run_scores["refusal_recall"] == 1.0
