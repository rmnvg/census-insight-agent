import pytest

from scripts.smoke_agent_memory import (
    KARNATAKA_DOCUMENT,
    ODISHA_DOCUMENT,
    SmokeFailure,
    assert_turn_1,
    assert_turn_2,
)


def citation(document_id: str, page: int, snippet: str = "value 75.36") -> dict:
    return {"document_id": document_id, "page_number": page, "snippet": snippet}


def test_answerable_smoke_turn_rejects_refusal_and_empty_claims() -> None:
    with pytest.raises(SmokeFailure, match="refused"):
        assert_turn_1({"refusal": True, "claims": [], "citations": [], "answer": "no"})
    with pytest.raises(SmokeFailure, match="zero factual claims"):
        assert_turn_1({"refusal": False, "claims": [], "citations": [], "answer": "75.36"})


def test_comparison_smoke_requires_both_documents() -> None:
    result = {
        "refusal": False,
        "claims": [{"claim_id": "one"}],
        "answer": "Karnataka 75.36, Odisha 72.9, difference 2.46 percentage points",
        "citations": [citation(KARNATAKA_DOCUMENT, 10)],
    }
    trace = {
        "events": [
            {
                "event": "resolved_query",
                "details": {"query": "Compare literacy in Karnataka and Odisha"},
            }
        ]
    }
    with pytest.raises(SmokeFailure, match="both comparison documents"):
        assert_turn_2(result, trace)
    complete = {
        **result,
        "citations": [
            citation(KARNATAKA_DOCUMENT, 10),
            citation(ODISHA_DOCUMENT, 9, "value 72.9"),
        ],
    }
    assert_turn_2(complete, trace)


def test_reconstructed_turn_one_passes_with_verified_physical_page_quote() -> None:
    result = {
        "refusal": False,
        "answer": "The literacy rate in Karnataka in 2011 was 75.36 percent.",
        "claims": [{"claim_id": "claim1", "citation_ids": ["citation-1"]}],
        "citations": [
            {
                "citation_id": "citation-1",
                "document_id": KARNATAKA_DOCUMENT,
                "page_number": 10,
                "snippet": (
                    "The Literacy Rate of the State has increased from 66.64 per cent in 2001 "
                    "to 75.36 per cent in 2011."
                ),
            }
        ],
    }
    assert_turn_1(result)
