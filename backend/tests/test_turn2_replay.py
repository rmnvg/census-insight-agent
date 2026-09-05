from backend.tests.test_agent_calculations import rate_evidence
from scripts.replay_turn2 import replay


def test_failed_live_comparison_replays_without_gemini() -> None:
    karnataka = rate_evidence("Karnataka", 75.36).model_copy(
        update={
            "chunk_id": "c1663038-700a-5623-94a1-1ca1d3a425c0",
            "document_id": "census-2011-karnataka-pca-highlights",
            "page_number": 50,
        }
    )
    odisha = rate_evidence("Odisha", 72.9).model_copy(
        update={
            "chunk_id": "b9186688-1446-5831-bb8e-877b56e91746",
            "document_id": "census-2011-odisha-pca-highlights",
            "page_number": 82,
        }
    )
    trace = {
        "run_id": "30e4237c-b44d-4703-aaae-93fb9bcbab71",
        "events": [
            {
                "event": "resolved_query",
                "details": {"query": "Compare literacy in Karnataka and Odisha in 2011"},
            },
            {
                "event": "evidence_assessment",
                "details": {"selected_evidence_ids": [karnataka.chunk_id, odisha.chunk_id]},
            },
            {
                "event": "structured_draft",
                "details": {
                    "answer_preview": "Model draft",
                    "claims": [
                        {
                            "claim_id": "karnataka",
                            "text": "Karnataka literacy was 75.36% in 2011.",
                            "evidence_ids": [karnataka.chunk_id],
                            "document_derived": False,
                        },
                        {
                            "claim_id": "odisha",
                            "text": "Odisha literacy was 72.9% in 2011.",
                            "evidence_ids": [odisha.chunk_id],
                            "document_derived": False,
                        },
                        {
                            "claim_id": "unsafe-model-calculation",
                            "text": "Karnataka was 3.37% higher.",
                            "evidence_ids": [karnataka.chunk_id, odisha.chunk_id],
                            "document_derived": True,
                        },
                    ],
                },
            },
        ],
        "tool_calls": [
            {
                "tool_name": "calculate",
                "status": "ok",
                "arguments": {
                    "description": "Compare literacy rates",
                    "operation": "percentage_difference",
                    "values": [75.36, 72.9],
                    "evidence_ids": [karnataka.chunk_id, odisha.chunk_id],
                },
            }
        ],
    }
    report = replay(trace, [karnataka, odisha])
    assert report["gemini_calls"] == 0
    assert report["citation_validation_passed"] is True
    assert report["smoke_assertion_passed"] is True
    assert report["calculation"][0]["operation"] == "difference"
    assert report["calculation"][0]["result"] == 2.46
