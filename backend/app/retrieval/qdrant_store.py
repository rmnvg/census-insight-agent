from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient, models

from backend.app.ingestion.models import DocumentChunk
from backend.app.ingestion.safety import validate_chunks_for_indexing
from backend.app.retrieval.models import SparseVectorData
from backend.app.retrieval.sparse import BM25_CONFIG


class CollectionSchemaError(RuntimeError):
    """An existing Qdrant collection is incompatible with this application."""


class QdrantStore:
    DENSE_VECTOR = "dense"
    SPARSE_VECTOR = "sparse"

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        *,
        dense_dimensions: int,
        dense_model: str,
        sparse_model: str,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.dense_dimensions = dense_dimensions
        self.dense_model = dense_model
        self.sparse_model = sparse_model

    def ensure_collection(self, *, rebuild: bool = False) -> None:
        exists = self.client.collection_exists(self.collection_name)
        if exists and rebuild:
            self.client.delete_collection(self.collection_name)
            exists = False
        if exists:
            self.validate_schema()
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.DENSE_VECTOR: models.VectorParams(
                    size=self.dense_dimensions,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self.SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
            metadata={
                "dense_model": self.dense_model,
                "dense_dimensions": self.dense_dimensions,
                "sparse_model": self.sparse_model,
                "sparse_modifier": "idf",
                "sparse_config": BM25_CONFIG,
            },
        )
        for field_name, schema in (
            ("document_id", models.PayloadSchemaType.KEYWORD),
            ("region", models.PayloadSchemaType.KEYWORD),
            ("page_number", models.PayloadSchemaType.INTEGER),
            ("section_path", models.PayloadSchemaType.KEYWORD),
        ):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=schema,
            )

    def validate_schema(self) -> None:
        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        sparse_vectors = info.config.params.sparse_vectors or {}
        if not isinstance(vectors, dict) or self.DENSE_VECTOR not in vectors:
            raise CollectionSchemaError("Qdrant collection is missing named dense vector")
        dense = vectors[self.DENSE_VECTOR]
        if dense.size != self.dense_dimensions or dense.distance != models.Distance.COSINE:
            raise CollectionSchemaError(
                f"Qdrant dense schema mismatch: expected cosine/{self.dense_dimensions}"
            )
        sparse = sparse_vectors.get(self.SPARSE_VECTOR)
        if sparse is None or sparse.modifier != models.Modifier.IDF:
            raise CollectionSchemaError("Qdrant collection is missing IDF-modified sparse vector")
        metadata = getattr(info.config, "metadata", None) or {}
        expected = {
            "dense_model": self.dense_model,
            "dense_dimensions": self.dense_dimensions,
            "sparse_model": self.sparse_model,
            "sparse_modifier": "idf",
            "sparse_config": BM25_CONFIG,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise CollectionSchemaError("Qdrant collection model metadata is incompatible")

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        dense_vectors: Sequence[list[float]],
        sparse_vectors: Sequence[SparseVectorData],
    ) -> None:
        validate_chunks_for_indexing(chunks)
        if not (len(chunks) == len(dense_vectors) == len(sparse_vectors)):
            raise ValueError("Chunk, dense, and sparse vector counts must match")
        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector={
                    self.DENSE_VECTOR: dense,
                    self.SPARSE_VECTOR: models.SparseVector(
                        indices=sparse.indices,
                        values=sparse.values,
                    ),
                },
                payload={
                    "text": chunk.text,
                    "chunk_id": chunk.chunk_id,
                    **chunk.metadata.model_dump(),
                },
            )
            for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors, strict=True)
        ]
        if points:
            self.client.upsert(self.collection_name, points=points, wait=True)

    def delete_points(self, point_ids: Sequence[str]) -> None:
        if point_ids:
            self.client.delete(
                self.collection_name,
                points_selector=models.PointIdsList(points=list(point_ids)),
                wait=True,
            )

    def document_count(self, document_id: str) -> int:
        result = self.client.count(
            self.collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            ),
            exact=True,
        )
        return result.count

    def list_documents(self) -> list[dict[str, Any]]:
        records, offset = self.client.scroll(
            self.collection_name, limit=256, with_payload=True, with_vectors=False
        )
        while offset is not None:
            page, offset = self.client.scroll(
                self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            records.extend(page)
        documents: dict[str, dict[str, Any]] = {}
        for record in records:
            payload = record.payload or {}
            document_id = payload.get("document_id")
            if isinstance(document_id, str):
                documents.setdefault(document_id, payload)
        return list(documents.values())
