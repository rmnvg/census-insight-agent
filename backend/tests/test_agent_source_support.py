import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from backend.app.agent.calculations import rounded_subtraction
from backend.app.agent.memory import (
    evidence_references,
    is_source_support_query,
    select_source_support_turn,
)
from backend.app.agent.models import (
    ClaimDerivation,
    EvidenceReference,
    ValidatedClaimRecord,
    ValidatedClaimTurn,
)
from backend.app.agent.skills import SkillRegistry
from backend.app.agent.tools import AgentTools, ArithmeticInput, StaleEvidenceError
from backend.app.retrieval.models import RetrievedEvidence
from backend.app.retrieval.qdrant_store import QdrantStore
from backend.app.retrieval.service import HybridRetrievalService


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def record(region: str, evidence_id: str, value: float) -> ValidatedClaimRecord:
    return ValidatedClaimRecord(
        claim_id=region.casefold(),
        text=f"{region} literacy was {value}% in 2011.",
        metric="literacy_rate",
        region=region,
        year=2011,
        population_scope="persons",
        residence_scope="total",
        value=value,
        displayed_value=str(value),
        unit="percent",
        evidence_ids=[evidence_id],
        citation_ids=[f"citation-{region}"],
        document_ids=[f"doc-{region.casefold()}"],
        page_numbers=[50 if region == "Karnataka" else 82],
        source_checksums=["a" * 64],
    )


def comparison(run_id: str = "comparison") -> ValidatedClaimTurn:
    karnataka = record("Karnataka", "karnataka-id", 75.36)
    odisha = record("Odisha", "odisha-id", 72.9)
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
            input_claim_ids=[odisha.claim_id, karnataka.claim_id],
        ),
        evidence_ids=[odisha.evidence_ids[0], karnataka.evidence_ids[0]],
        citation_ids=[odisha.citation_ids[0], karnataka.citation_ids[0]],
        document_ids=[odisha.document_ids[0], karnataka.document_ids[0]],
        page_numbers=[odisha.page_numbers[0], karnataka.page_numbers[0]],
        source_checksums=["a" * 64, "a" * 64],
    )
    return ValidatedClaimTurn(
        run_id=run_id, task_type="comparison", claims=[odisha, karnataka, derived]
    )


def payload(reference: EvidenceReference) -> dict[str, Any]:
    region = reference.document_id.removeprefix("doc-").title()
    value = 75.36 if region == "Karnataka" else 72.9
    text = f"The literacy rate for total persons in {region} was {value}% in 2011."
    return RetrievedEvidence(
        chunk_id=reference.evidence_id,
        text=text,
        document_title=f"Census 2011 {region}",
        document_id=reference.document_id,
        region=region,
        page_number=reference.page_number,
        citation_snippet=text,
        section_path=["Literacy"],
        extraction_method="provided_markdown",
        coverage_status="indexed_provided_markdown",
        source_checksum=reference.source_checksum,
        retrieval_score=0.0,
    ).model_dump(exclude={"retrieval_score"})


def tools_for(client: Any, tmp_path: Path) -> AgentTools:
    store = SimpleNamespace(client=client, collection_name="test")
    return AgentTools(
        cast(HybridRetrievalService, SimpleNamespace()),
        cast(QdrantStore, store),
        SkillRegistry(tmp_path / "skills"),
        tmp_path,
    )


def test_source_support_detection_and_latest_comparison_resolution() -> None:
    assert is_source_support_query("Which source pages support those values?")
    assert is_source_support_query("Where did that number come from?")
    older = comparison("older")
    latest = comparison("latest")
    selected, clarification = select_source_support_turn(
        "Which pages support those values?", [older, latest]
    )
    assert selected == latest
    assert clarification is None


def test_ambiguous_source_request_requests_clarification() -> None:
    selected, clarification = select_source_support_turn(
        "Show me the sources", [comparison("first"), comparison("second")]
    )
    assert selected is None
    assert clarification and "Which previous" in clarification


def test_evidence_ids_rehydrate_directly_without_embedding(tmp_path: Path) -> None:
    references = evidence_references(comparison().claims)

    class Client:
        calls = 0

        def retrieve(self, collection: str, **kwargs: Any) -> list[Any]:
            del collection
            self.calls += 1
            return [
                SimpleNamespace(id=evidence_id, payload=payload(references[evidence_id]))
                for evidence_id in kwargs["ids"]
            ]

    client = Client()
    result = run(tools_for(client, tmp_path).get_evidence_by_ids(list(references), references))
    assert {item.chunk_id for item in result} == set(references)
    assert client.calls == 1


@pytest.mark.parametrize("failure", ["missing", "checksum"])
def test_missing_or_checksum_changed_evidence_is_rejected(tmp_path: Path, failure: str) -> None:
    references = evidence_references(comparison().claims)

    class Client:
        def retrieve(self, collection: str, **kwargs: Any) -> list[Any]:
            del collection
            ids = kwargs["ids"]
            if failure == "missing":
                ids = ids[:-1]
            records = []
            for evidence_id in ids:
                value = payload(references[evidence_id])
                if failure == "checksum":
                    value["source_checksum"] = "b" * 64
                records.append(SimpleNamespace(id=evidence_id, payload=value))
            return records

    with pytest.raises(StaleEvidenceError):
        run(tools_for(Client(), tmp_path).get_evidence_by_ids(list(references), references))


def test_public_calculation_rounding_has_no_binary_float_residue() -> None:
    assert rounded_subtraction(72.9, 75.36) == -2.46
    assert (
        AgentTools.calculate(ArithmeticInput(operation="difference", values=[72.9, 75.36]))
        == -2.46
    )
    derivation = comparison().claims[-1].derivation
    assert derivation is not None
    assert derivation.model_dump(mode="json")["result"] == -2.46
