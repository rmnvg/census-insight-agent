import json
from pathlib import Path

from backend.app.trust import Scorecard, TrustCase, score_case, summarize

CASES = [
    TrustCase.model_validate(item)
    for item in json.loads(Path("evals/trust_benchmark.json").read_text(encoding="utf-8"))
]


def case(case_id: str) -> TrustCase:
    return next(item for item in CASES if item.case_id == case_id)


def response(
    claims: list[dict[str, object]], snippets: dict[str, str], **extra: object
) -> dict[str, object]:
    return {
        "answer": " ".join(str(claim["text"]) for claim in claims) + "\n\nSources:\n- x",
        "claims": claims,
        "citations": [{"citation_id": key, "snippet": value} for key, value in snippets.items()],
        "artifacts": [],
        "refusal": False,
        "trace_id": "t",
        **extra,
    }


def test_benchmark_cases_are_well_formed() -> None:
    assert len({item.case_id for item in CASES}) == len(CASES) >= 15
    assert all(item.expect_refusal == (item.category == "refusal") for item in CASES)
    assert all(item.expected_values or item.expect_refusal for item in CASES)


def test_grounded_correct_answer_passes() -> None:
    result = score_case(
        case("ka-sex-ratio"),
        response(
            [{"text": "Karnataka's sex ratio was 973.", "value": 973, "citation_ids": ["c1"]}],
            {"c1": "| - | <b>KARNATAKA</b> | <b>965</b> | <b>973</b> |"},
        ),
        latency_seconds=20,
    )
    assert result.passed and result.ungrounded_claims == []


def test_wrong_subgroup_value_fails_even_when_quoted() -> None:
    # The Scheduled Tribes figure is grounded in its own quote but is the wrong answer.
    result = score_case(
        case("ka-sex-ratio"),
        response(
            [{"text": "Karnataka's sex ratio was 990.", "value": 990, "citation_ids": ["c1"]}],
            {"c1": "| - | KARNATAKA | 972 | 990 |"},
        ),
        latency_seconds=20,
    )
    assert not result.passed and result.missing == ["973"]


def test_number_absent_from_its_own_quote_is_ungrounded() -> None:
    result = score_case(
        case("ka-sex-ratio"),
        response(
            [{"text": "Karnataka's sex ratio was 973.", "value": 973, "citation_ids": ["c1"]}],
            {"c1": "The sex ratio of the state improved."},
        ),
        latency_seconds=20,
    )
    assert not result.passed and len(result.ungrounded_claims) == 1


def test_derived_values_must_recompute() -> None:
    claims = [
        {"text": "Odisha 979", "value": 979, "citation_ids": ["c1"]},
        {"text": "MP 931", "value": 931, "citation_ids": ["c2"]},
        {
            "text": "Odisha is higher by 48.",
            "value": 48,
            "citation_ids": ["c1", "c2"],
            "derivation": {"operation": "difference", "operands": [979, 931], "result": 48},
        },
    ]
    snippets = {"c1": "| ODISHA | 979 |", "c2": "| MADHYA PRADESH | 931 |"}
    assert score_case(case("od-mp-sex-ratio"), response(claims, snippets), latency_seconds=1).passed
    claims[2]["derivation"] = {"operation": "difference", "operands": [979, 931], "result": 50}
    broken = score_case(case("od-mp-sex-ratio"), response(claims, snippets), latency_seconds=1)
    assert not broken.passed and broken.arithmetic_errors


def test_refusal_cases_and_summary() -> None:
    refused = score_case(
        case("refuse-2021"),
        {
            "answer": "I can only answer from the 2011 reports.",
            "claims": [],
            "citations": [],
            "refusal": True,
        },
        latency_seconds=3,
    )
    answered = score_case(
        case("refuse-2021"),
        response(
            [{"text": "Karnataka's literacy was 75.36.", "value": 75.36, "citation_ids": ["c1"]}],
            {"c1": "| KARNATAKA | 75.36 |"},
        ),
        latency_seconds=3,
    )
    errored = score_case(case("ka-literacy"), None, latency_seconds=120, error_code="MODEL_TIMEOUT")
    assert refused.passed and not answered.passed and errored.outcome == "error"
    card = summarize([refused, answered, errored], model="gemini-2.5-flash")
    assert isinstance(card, Scorecard)
    assert (card.cases_passed, card.refusal_passed, card.wrong_answers) == (1, 1, 1)
