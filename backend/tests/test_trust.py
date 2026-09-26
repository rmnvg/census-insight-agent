"""The trust scorer, checked against a real recorded benchmark run.

`fixtures/trust_live_responses.jsonl` is every raw API response from a live 18-case run
(2026-09-26, gemini-2.5-flash). The scorer must pass those real answers, and each mutation below
reproduces one way an answer can be wrong while its numbers still appear somewhere in the quotes.
"""

import copy
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from backend.app.trust import Scorecard, TrustCase, score_case, summarize
from backend.app.trust_provenance import conflicts, locate

CASES = {
    item["case_id"]: TrustCase.model_validate(item)
    for item in json.loads(Path("evals/trust_benchmark.json").read_text(encoding="utf-8"))
}
RECORDED = {
    record["case_id"]: record["response"]
    for record in map(
        json.loads,
        (Path(__file__).parent / "fixtures" / "trust_live_responses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines(),
    )
}


def recorded(case_id: str) -> dict[str, Any]:
    return copy.deepcopy(RECORDED[case_id])


def score(case_id: str, response: dict[str, Any] | None) -> Any:
    return score_case(CASES[case_id], response, latency_seconds=10)


def claim_for(response: dict[str, Any], region: str, residence: str = "total") -> dict[str, Any]:
    return next(
        claim
        for claim in response["claims"]
        if (claim.get("region") or "").casefold() == region.casefold()
        and (claim.get("residence_scope") or "total") == residence
        and not claim.get("derivation")
    )


def test_benchmark_cases_are_well_formed() -> None:
    cases = list(CASES.values())
    assert len(cases) >= 15
    assert all(item.expect_refusal == (item.category == "refusal") for item in cases)
    assert all(item.expected_facts or item.expect_refusal for item in cases)


def test_every_recorded_live_answer_passes() -> None:
    results = [score(case_id, response) for case_id, response in RECORDED.items()]
    assert [(item.case_id, item.failures) for item in results if not item.passed] == []
    card = summarize(results, "gemini-2.5-flash")
    assert (card.wrong_answers, card.unsupported_answers, card.false_refusals) == (0, 0, 0)
    assert card.scorer_version == 2


def test_swapped_region_labels_are_wrong_answers() -> None:
    response = recorded("od-mp-sex-ratio")
    odisha, madhya = claim_for(response, "Odisha"), claim_for(response, "Madhya Pradesh")
    odisha["region"], madhya["region"] = "Madhya Pradesh", "Odisha"
    result = score("od-mp-sex-ratio", response)
    assert result.verdict == "wrong_answer"
    assert len(result.misattributed_claims) == 2
    # Independently, neither number sits on its new region's row in the cited table.
    assert len(result.ungrounded_claims) == 2


def test_subgroup_value_presented_as_the_state_figure_is_wrong() -> None:
    # 990 is the Scheduled Tribes sex ratio; stated for all of Karnataka it is false.
    response = recorded("ka-sex-ratio")
    claim_for(response, "Karnataka")["value"] = 990.0
    result = score("ka-sex-ratio", response)
    assert result.verdict == "wrong_answer" and result.misattributed_claims


def test_subgroup_value_labelled_as_its_subgroup_is_not_a_contradiction() -> None:
    response = recorded("ka-sex-ratio")
    extra = copy.deepcopy(claim_for(response, "Karnataka"))
    extra.update(value=990.0, population_scope="Scheduled Tribes", text="Scheduled Tribes: 990.")
    response["claims"].append(extra)
    result = score("ka-sex-ratio", response)
    assert result.misattributed_claims == []
    # It is still unsupported: 990 is not in the cited all-persons table.
    assert result.verdict == "unsupported"


def test_neighbouring_residence_column_is_caught() -> None:
    # The rural figure (979) relabelled as the state total.
    response = recorded("ka-sex-ratio")
    rural = claim_for(response, "Karnataka", "rural")
    claim_for(response, "Karnataka")["value"] = rural["value"]
    result = score("ka-sex-ratio", response)
    assert result.verdict == "wrong_answer"
    assert any("Rural column" in item for item in result.ungrounded_claims)


def test_wrong_year_label_on_a_correct_number_is_unsupported() -> None:
    response = recorded("ka-literacy")
    claim = claim_for(response, "Karnataka")
    claim["year"] = 2001
    result = score("ka-literacy", response)
    assert result.verdict == "unsupported"
    assert any("2011 column, not 2001" in item for item in result.ungrounded_claims)
    assert result.missing == ["Karnataka literacy 2011 = 75.36"]


def test_value_not_on_its_regions_row_is_unsupported() -> None:
    response = recorded("ka-sex-ratio")
    claim_for(response, "Karnataka", "rural")["region"] = "Mysore"
    result = score("ka-sex-ratio", response)
    assert result.verdict == "unsupported"
    assert any("not on a row or sentence about Mysore" in item for item in result.ungrounded_claims)


def test_declining_an_answerable_question_is_a_false_refusal_not_a_wrong_answer() -> None:
    declined = {
        "answer": "I could not verify that.",
        "claims": [],
        "citations": [],
        "refusal": True,
    }
    result = score("od-literacy", declined)
    assert result.verdict == "false_refusal"
    card = summarize([result], "m")
    assert (card.false_refusals, card.wrong_answers) == (1, 0)


def test_answering_a_question_that_must_be_refused_is_wrong() -> None:
    result = score("refuse-2021", recorded("ka-literacy"))
    assert result.verdict == "wrong_answer"
    assert score("refuse-2021", recorded("refuse-2021")).passed


def test_missing_computed_gap_and_broken_arithmetic() -> None:
    response = recorded("od-mp-sex-ratio")
    derived = [claim for claim in response["claims"] if claim.get("derivation")]
    assert derived
    derived[0]["derivation"]["result"] = 50.0
    assert score("od-mp-sex-ratio", response).verdict == "wrong_answer"
    response = recorded("od-mp-sex-ratio")
    response["claims"] = [claim for claim in response["claims"] if not claim.get("derivation")]
    result = score("od-mp-sex-ratio", response)
    assert result.verdict == "incomplete" and result.missing == ["computed 48"]


def test_errors_and_unstable_repeats() -> None:
    errored = score_case(
        CASES["ka-literacy"], None, latency_seconds=120, error_code="MODEL_TIMEOUT"
    )
    assert errored.verdict == "error" and errored.outcome == "error"
    first = score("od-literacy", recorded("od-literacy"))
    second = score("od-literacy", {"answer": "No.", "claims": [], "citations": [], "refusal": True})
    card = summarize([first, second.model_copy(update={"repeat": 2})], "m")
    assert isinstance(card, Scorecard)
    assert card.unstable_cases == ["od-literacy"] and card.repeats == 2


def test_column_paths_come_from_the_quotes_own_headers() -> None:
    snippet = recorded("ka-literacy")["citations"][0]["snippet"]
    [location] = locate(75.36, "Karnataka", snippet, "Statement 19 (Persons)", Decimal("0.005"))
    assert location.column == ("Literacy Rate", "2011", "Total")
    [earlier] = locate(66.64, "Karnataka", snippet, "Statement 19 (Persons)", Decimal("0.005"))
    assert earlier.column == ("Literacy Rate", "2001", "Total")
    assert conflicts(earlier, year=2011, residence="total", group=None) == [
        "the value is in the 2001 column, not 2011"
    ]


@pytest.mark.parametrize(
    ("context", "group", "expected"),
    [
        ("Sex Ratio (number of females per 1000 males) by residence", None, []),
        ("Literates and Literacy Rate by residence : 2011 (FEMALES)", None, ["females"]),
        ("Literates and Literacy Rate by residence : 2011 (MALES)", "females", ["males"]),
        ("Literates and Literacy Rate by residence : 2011 (FEMALES)", "females", []),
    ],
)
def test_statement_group_is_read_from_the_quote_context(
    context: str, group: str | None, expected: list[str]
) -> None:
    location = locate(
        1.0, "X", "| Code | Region | Value |\n|---|---|---|\n| 1 | X | 1.0 |", context, Decimal("0")
    )[0]
    found = conflicts(location, year=None, residence="total", group=group)
    assert [problem.split("for ")[1].split(",")[0] for problem in found] == expected


ADVERSARIAL = {
    record["case_id"]: record["response"]
    for record in map(
        json.loads,
        (Path(__file__).parent / "fixtures" / "trust_live_adversarial_responses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines(),
    )
}


def test_recorded_adversarial_answers_get_the_right_verdicts() -> None:
    """Live run of the adversarial cases (2026-09-26): scope, residence, and multi-turn."""
    verdicts = {
        case_id: score(case_id, response).verdict for case_id, response in ADVERSARIAL.items()
    }
    assert verdicts == {
        # The agent treats any year but 2011 as out of scope, though the tables hold 2001 columns.
        "ka-sex-ratio-2001": "false_refusal",
        "od-sex-ratio-2001": "false_refusal",
        "od-urban-sex-ratio": "passed",
        "mp-rural-literacy": "passed",
        "od-mp-urban-sex-ratio": "passed",
        "followup-urban": "passed",
        "followup-compare": "passed",
    }


def test_sex_ratio_unit_in_population_scope_is_not_a_group() -> None:
    # The model put "females per 1000 males" in population_scope; that is the unit, not a group.
    response = copy.deepcopy(ADVERSARIAL["od-mp-urban-sex-ratio"])
    assert {claim["population_scope"] for claim in response["claims"]} == {"females per 1000 males"}
    assert score("od-mp-urban-sex-ratio", response).passed


def test_clarifying_question_without_numbers_is_a_false_refusal() -> None:
    response = ADVERSARIAL["ka-sex-ratio-2001"]
    assert not response["refusal"] and not response["claims"]
    assert score("ka-sex-ratio-2001", response).verdict == "false_refusal"


def test_computed_gap_matches_whichever_side_is_subtracted() -> None:
    # Live run 1 of followup-compare (2026-09-26) computed Madhya Pradesh minus Odisha: -48,
    # "lower by 48". That is the verified 48-point gap stated from the other side.
    response = copy.deepcopy(ADVERSARIAL["followup-compare"])
    for claim in response["claims"]:
        derivation = claim.get("derivation")
        if derivation:
            derivation["operands"] = list(reversed(derivation["operands"]))
            derivation["result"] = -derivation["result"]
            claim["value"] = -claim["value"]
    assert score("followup-compare", response).passed
