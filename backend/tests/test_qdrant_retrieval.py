import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from backend.app.agent.skills import SkillRegistry
from backend.app.agent.tools import AgentTools
from backend.app.ingestion.models import ChunkMetadata, DocumentChunk
from backend.app.retrieval.models import RetrievedEvidence, SparseVectorData
from backend.app.retrieval.qdrant_store import CollectionSchemaError, QdrantStore
from backend.app.retrieval.service import HybridRetrievalService, RetrievalProvenanceError
from backend.app.retrieval.validation import validate_collection


def make_chunk(
    chunk_id: str, *, document_id: str, region: str, page_number: int, text: str
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            document_id=document_id,
            document_title=f"Document {document_id}",
            region=region,
            page_number=page_number,
            section_path=["Population"],
            chunk_index=0,
            citation_snippet=text,
            extraction_method="provided_markdown",
            coverage_status="indexed_provided_markdown",
            source_checksum="a" * 64,
        ),
    )


def make_store(client: QdrantClient, name: str = "test_collection") -> QdrantStore:
    return QdrantStore(
        client,
        name,
        dense_dimensions=3,
        dense_model="test-dense",
        sparse_model="Qdrant/bm25",
    )


def test_collection_schema_creation_and_validation(tmp_path: Path) -> None:
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    store = make_store(client)

    store.ensure_collection()
    store.validate_schema()


def test_incompatible_collection_is_not_silently_reused() -> None:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        "test_collection",
        vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )

    with pytest.raises(CollectionSchemaError, match="dense schema mismatch"):
        make_store(client).ensure_collection()


class DenseProvider:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class SparseEncoder:
    def embed_query(self, text: str) -> SparseVectorData:
        return SparseVectorData(indices=[1], values=[1.0])


def test_hybrid_fusion_filters_and_preserves_provenance(tmp_path: Path) -> None:
    client = QdrantClient(path=str(tmp_path / "hybrid"))
    store = make_store(client)
    store.ensure_collection()
    chunks = [
        make_chunk(
            "2f46558a-89cf-44c2-8234-d74a9f96dcd0",
            document_id="doc-a",
            region="Karnataka",
            page_number=1,
            text="Mysuru population is 100",
        ),
        make_chunk(
            "2ff68f2d-d4c1-497c-b644-2e14e7203371",
            document_id="doc-b",
            region="Kerala",
            page_number=2,
            text="Other evidence",
        ),
    ]
    store.upsert(
        chunks,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [
            SparseVectorData(indices=[1], values=[1.0]),
            SparseVectorData(indices=[2], values=[1.0]),
        ],
    )
    service = HybridRetrievalService(
        client=client,
        collection_name="test_collection",
        dense_provider=cast(Any, DenseProvider()),
        sparse_encoder=cast(Any, SparseEncoder()),
    )

    response = service.search_response(
        query="Mysuru",
        document_ids=["doc-a"],
        regions=["Karnataka"],
        top_k=2,
        debug=True,
    )

    assert [item.document_id for item in response.evidence] == ["doc-a"]
    assert response.evidence[0].page_number == 1
    assert response.evidence[0].citation_snippet == "Mysuru population is 100"
    assert response.evidence[0].coverage_status == "indexed_provided_markdown"
    assert response.evidence[0].source_checksum == "a" * 64
    assert response.debug is not None
    assert response.debug.applied_filters["regions"] == ["Karnataka"]
    assert response.debug.fused_ranking[0].chunk_id == chunks[0].chunk_id
    assert response.debug.dense_candidates
    assert response.debug.sparse_candidates
    assert response.evidence_sufficiency.status == "candidate_evidence"

    unfiltered = service.search("Mysuru", top_k=2)
    assert unfiltered[0].chunk_id == chunks[0].chunk_id


def test_retrieval_refuses_missing_page_metadata() -> None:
    point = type(
        "Point",
        (),
        {"id": "bad-point", "score": 1.0, "payload": {"chunk_id": "bad"}},
    )()

    with pytest.raises(RetrievalProvenanceError, match="missing valid chunk/page provenance"):
        HybridRetrievalService._evidence([point])


def test_retrieval_refuses_missing_source_checksum() -> None:
    point = type(
        "Point",
        (),
        {
            "id": "bad-checksum-point",
            "score": 1.0,
            "payload": {
                "chunk_id": "bad-checksum-point",
                "text": "Trusted text",
                "document_title": "Document",
                "document_id": "document",
                "region": "Karnataka",
                "page_number": 1,
                "citation_snippet": "Trusted text",
                "section_path": [],
                "extraction_method": "provided_markdown",
                "coverage_status": "indexed_provided_markdown",
            },
        },
    )()

    with pytest.raises(RetrievalProvenanceError, match="invalid citation payload"):
        HybridRetrievalService._evidence([point])


@pytest.mark.parametrize("checksum", ["", "not-a-checksum", "A" * 64, "a" * 63])
def test_retrieved_evidence_rejects_invalid_source_checksum(checksum: str) -> None:
    payload = {
        "chunk_id": "chunk",
        "text": "Trusted text",
        "document_title": "Document",
        "document_id": "document",
        "region": "Karnataka",
        "page_number": 1,
        "citation_snippet": "Trusted text",
        "section_path": [],
        "extraction_method": "provided_markdown",
        "coverage_status": "indexed_provided_markdown",
        "source_checksum": checksum,
        "retrieval_score": 1.0,
    }
    with pytest.raises(ValidationError):
        RetrievedEvidence.model_validate(payload)


def test_retrieved_evidence_checksum_serialization_round_trip() -> None:
    checksum = "c" * 64
    value = RetrievedEvidence(
        chunk_id="chunk",
        text="Trusted text",
        document_title="Document",
        document_id="document",
        region="Karnataka",
        page_number=1,
        citation_snippet="Trusted text",
        section_path=[],
        extraction_method="provided_markdown",
        coverage_status="indexed_provided_markdown",
        source_checksum=checksum,
        retrieval_score=1.0,
    )
    assert (
        RetrievedEvidence.model_validate_json(value.model_dump_json()).source_checksum == checksum
    )


def test_source_checksum_survives_sibling_expansion(tmp_path: Path) -> None:
    checksum = "d" * 64
    candidate = RetrievedEvidence(
        chunk_id="candidate",
        text="Trusted text",
        document_title="Document",
        document_id="document",
        region="Karnataka",
        page_number=5,
        citation_snippet="Trusted text",
        section_path=[],
        extraction_method="provided_markdown",
        coverage_status="indexed_provided_markdown",
        source_checksum=checksum,
        retrieval_score=1.0,
    )

    class Client:
        def scroll(self, *_: object, **__: object) -> tuple[list[object], None]:
            payload = candidate.model_dump(exclude={"retrieval_score"})
            return [SimpleNamespace(payload=payload)], None

    store = SimpleNamespace(client=Client(), collection_name="collection")
    tools = AgentTools(
        cast(Any, None), cast(Any, store), SkillRegistry(tmp_path / "skills"), tmp_path
    )
    expanded = asyncio.run(tools.expand_candidate_pages([candidate]))
    assert len(expanded) == 1
    assert expanded[0].source_checksum == checksum


def test_unanswerable_signal_does_not_treat_a_score_as_support() -> None:
    evidence = RetrievedEvidence(
        chunk_id="2f46558a-89cf-44c2-8234-d74a9f96dcd0",
        document_id="doc-a",
        document_title="Document A",
        region="Karnataka",
        page_number=1,
        text="Census population evidence",
        citation_snippet="Census population evidence",
        section_path=[],
        extraction_method="provided_markdown",
        coverage_status="indexed_provided_markdown",
        source_checksum="a" * 64,
        retrieval_score=0.99,
    )
    result = HybridRetrievalService._sufficiency(
        "Mars rover battery voltage",
        [evidence],
    )

    assert result.status == "insufficient_evidence"
    assert result.requires_claim_validation is True


def test_full_collection_validation_checks_vectors_and_payloads(tmp_path: Path) -> None:
    client = QdrantClient(path=str(tmp_path / "validation"))
    store = make_store(client)
    store.ensure_collection()
    chunk = make_chunk(
        "2f46558a-89cf-44c2-8234-d74a9f96dcd0",
        document_id="doc-a",
        region="Karnataka",
        page_number=1,
        text="Verbatim census evidence",
    )
    store.upsert(
        [chunk],
        [[1.0, 0.0, 0.0]],
        [SparseVectorData(indices=[1], values=[1.0])],
    )

    report = validate_collection(client, "test_collection", expected_points=1, dense_dimensions=3)

    assert report.scanned_points == report.valid_points == 1
    assert report.invalid_points == 0
    assert report.invalid_by_field == {}
    assert report.non_empty_sparse_vectors == 1
    assert report.dense_query_results == 1
    assert report.sparse_query_results == 1
    assert report.client_server_compatible is True
