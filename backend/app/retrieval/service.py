import re
import time
from typing import Any

from qdrant_client import QdrantClient, models

from backend.app.providers.embeddings import VertexEmbeddingProvider
from backend.app.retrieval.models import (
    EvidenceSufficiency,
    RetrievalCandidate,
    RetrievalDiagnostics,
    RetrievalSearchResponse,
    RetrievedEvidence,
)
from backend.app.retrieval.qdrant_store import QdrantStore
from backend.app.retrieval.sparse import BM25SparseEncoder


class RetrievalProvenanceError(RuntimeError):
    """Retrieved payload lacks citation-safe page provenance."""


class HybridRetrievalService:
    """Dense plus local-BM25 retrieval fused by Qdrant reciprocal-rank fusion."""

    def __init__(
        self,
        *,
        client: QdrantClient,
        collection_name: str,
        dense_provider: VertexEmbeddingProvider,
        sparse_encoder: BM25SparseEncoder,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.dense_provider = dense_provider
        self.sparse_encoder = sparse_encoder

    def search(
        self,
        query: str,
        document_ids: list[str] | None = None,
        regions: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedEvidence]:
        return self.search_response(
            query=query,
            document_ids=document_ids,
            regions=regions,
            top_k=top_k,
            debug=False,
        ).evidence

    def search_response(
        self,
        *,
        query: str,
        document_ids: list[str] | None,
        regions: list[str] | None,
        top_k: int,
        debug: bool,
    ) -> RetrievalSearchResponse:
        started = time.monotonic()
        if not query.strip():
            raise ValueError("Retrieval query must be non-empty")
        dense = self.dense_provider.embed_query(query)
        sparse = self.sparse_encoder.embed_query(query)
        query_filter = self._filter(document_ids, regions)
        candidate_limit = max(top_k * 4, 20)
        prefetch = [
            models.Prefetch(
                query=dense,
                using=QdrantStore.DENSE_VECTOR,
                filter=query_filter,
                limit=candidate_limit,
            ),
            models.Prefetch(
                query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                using=QdrantStore.SPARSE_VECTOR,
                filter=query_filter,
                limit=candidate_limit,
            ),
        ]
        fused = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        ).points
        evidence = self._evidence(fused)
        diagnostics = None
        if debug:
            dense_points = self.client.query_points(
                collection_name=self.collection_name,
                query=dense,
                using=QdrantStore.DENSE_VECTOR,
                query_filter=query_filter,
                limit=candidate_limit,
                with_payload=False,
                with_vectors=False,
            ).points
            sparse_points = self.client.query_points(
                collection_name=self.collection_name,
                query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                using=QdrantStore.SPARSE_VECTOR,
                query_filter=query_filter,
                limit=candidate_limit,
                with_payload=False,
                with_vectors=False,
            ).points
            diagnostics = RetrievalDiagnostics(
                dense_candidates=self._candidates(dense_points),
                sparse_candidates=self._candidates(sparse_points),
                fused_ranking=self._candidates(fused),
                applied_filters={
                    key: value
                    for key, value in {
                        "document_ids": document_ids,
                        "regions": regions,
                    }.items()
                    if value
                },
                retrieval_time_ms=round((time.monotonic() - started) * 1000, 3),
            )
        return RetrievalSearchResponse(
            evidence=evidence,
            evidence_sufficiency=self._sufficiency(query, evidence),
            debug=diagnostics,
        )

    @staticmethod
    def _sufficiency(query: str, evidence: list[RetrievedEvidence]) -> EvidenceSufficiency:
        stopwords = {"a", "an", "and", "in", "is", "of", "the", "to", "was", "what"}
        terms = {
            token
            for token in re.findall(r"[a-z0-9]+", query.casefold())
            if len(token) > 2 and token not in stopwords
        }
        evidence_text = " ".join(item.text for item in evidence).casefold()
        overlap = {term for term in terms if term in evidence_text}
        if evidence and overlap:
            return EvidenceSufficiency(
                status="candidate_evidence",
                rule=(
                    "At least one substantive query term appears in retrieved evidence; "
                    "claim validation is still required."
                ),
            )
        return EvidenceSufficiency(
            status="insufficient_evidence",
            rule=(
                "No substantive query-term overlap was found in retrieved evidence; retrieval "
                "score alone is not proof of answerability."
            ),
        )

    @staticmethod
    def _filter(document_ids: list[str] | None, regions: list[str] | None) -> models.Filter | None:
        conditions: list[models.Condition] = []
        if document_ids:
            conditions.append(
                models.FieldCondition(key="document_id", match=models.MatchAny(any=document_ids))
            )
        if regions:
            conditions.append(
                models.FieldCondition(key="region", match=models.MatchAny(any=regions))
            )
        return models.Filter(must=conditions) if conditions else None

    @staticmethod
    def _candidates(points: list[Any]) -> list[RetrievalCandidate]:
        return [
            RetrievalCandidate(chunk_id=str(point.id), score=float(point.score)) for point in points
        ]

    @staticmethod
    def _evidence(points: list[Any]) -> list[RetrievedEvidence]:
        results: list[RetrievedEvidence] = []
        seen: set[str] = set()
        for point in points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")
            page_number = payload.get("page_number")
            if not isinstance(chunk_id, str) or not isinstance(page_number, int) or page_number < 1:
                raise RetrievalProvenanceError(
                    f"Qdrant point {point.id} is missing valid chunk/page provenance"
                )
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            try:
                results.append(
                    RetrievedEvidence.model_validate(
                        {
                            "chunk_id": chunk_id,
                            "text": payload["text"],
                            "document_title": payload["document_title"],
                            "document_id": payload["document_id"],
                            "region": payload["region"],
                            "page_number": page_number,
                            "citation_snippet": payload["citation_snippet"],
                            "citation_limitation": payload.get("citation_limitation"),
                            "section_path": payload.get("section_path", []),
                            "extraction_method": payload["extraction_method"],
                            "coverage_status": payload["coverage_status"],
                            "retrieval_score": point.score,
                        }
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise RetrievalProvenanceError(
                    f"Qdrant point {point.id} has invalid citation payload"
                ) from error
        return results
