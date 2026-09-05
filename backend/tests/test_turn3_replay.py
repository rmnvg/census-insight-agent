from backend.app.agent.models import ClaimDerivation, ValidatedClaimRecord, ValidatedClaimTurn
from backend.tests.test_agent_calculations import rate_evidence
from scripts.replay_turn3 import replay


def test_turn_3_replays_with_both_sources_and_no_external_calls() -> None:
    karnataka = rate_evidence("Karnataka", 75.36).model_copy(
        update={
            "chunk_id": "karnataka",
            "document_id": "census-2011-karnataka-pca-highlights",
            "page_number": 50,
            "source_checksum": "a" * 64,
        }
    )
    odisha = rate_evidence("Odisha", 72.9).model_copy(
        update={
            "chunk_id": "odisha",
            "document_id": "census-2011-odisha-pca-highlights",
            "page_number": 82,
            "source_checksum": "b" * 64,
        }
    )
    source_claims = [
        ValidatedClaimRecord(
            claim_id="odisha-claim",
            text="Odisha literacy was 72.9%.",
            metric="literacy_rate",
            region="Odisha",
            year=2011,
            population_scope="persons",
            residence_scope="total",
            value=72.9,
            displayed_value="72.9",
            unit="percent",
            evidence_ids=[odisha.chunk_id],
            citation_ids=["citation-1"],
            document_ids=[odisha.document_id],
            page_numbers=[82],
            source_checksums=[odisha.source_checksum],
        ),
        ValidatedClaimRecord(
            claim_id="karnataka-claim",
            text="Karnataka literacy was 75.36%.",
            metric="literacy_rate",
            region="Karnataka",
            year=2011,
            population_scope="persons",
            residence_scope="total",
            value=75.36,
            displayed_value="75.36",
            unit="percent",
            evidence_ids=[karnataka.chunk_id],
            citation_ids=["citation-2"],
            document_ids=[karnataka.document_id],
            page_numbers=[50],
            source_checksums=[karnataka.source_checksum],
        ),
    ]
    derived = ValidatedClaimRecord(
        claim_id="difference",
        text="Odisha is lower than Karnataka by 2.46 percentage points.",
        metric="literacy_rate",
        year=2011,
        value=-2.46,
        displayed_value="2.46",
        unit="percentage_points",
        document_derived=True,
        derivation=ClaimDerivation(
            operation="difference",
            operands=[72.9, 75.36],
            result=-2.46,
            unit="percentage_points",
            input_claim_ids=["odisha-claim", "karnataka-claim"],
        ),
        evidence_ids=[odisha.chunk_id, karnataka.chunk_id],
        citation_ids=["citation-1", "citation-2"],
        document_ids=[odisha.document_id, karnataka.document_id],
        page_numbers=[82, 50],
        source_checksums=[odisha.source_checksum, karnataka.source_checksum],
    )
    turn = ValidatedClaimTurn(
        run_id="saved-turn-2", task_type="comparison", claims=[*source_claims, derived]
    )

    report = replay(turn, [odisha, karnataka])

    assert report["gemini_calls"] == 0
    assert report["embedding_calls"] == 0
    assert report["qdrant_writes"] == 0
    assert report["turn_3_smoke_assertion_passed"] is True
    assert report["turn_4_assertion_passed"] is True
    assert all(item["exact_substring"] for item in report["citations"])
