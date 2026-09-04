from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from backend.app.retrieval.models import SparseVectorData

BM25_CONFIG: dict[str, str | float | int | bool] = {
    "k": 1.2,
    "b": 0.75,
    "avg_len": 256.0,
    "language": "english",
    "token_max_length": 40,
    "disable_stemmer": False,
}


class SparseResult(Protocol):
    indices: Any
    values: Any


class SparseModel(Protocol):
    def embed(self, documents: list[str], *, batch_size: int) -> Iterable[SparseResult]: ...

    def query_embed(self, query: str) -> Iterable[SparseResult]: ...


class BM25SparseEncoder:
    """Local deterministic FastEmbed BM25 sparse encoding."""

    def __init__(
        self,
        model_name: str = "Qdrant/bm25",
        model: SparseModel | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.model_name = model_name
        if model is None:
            from fastembed import SparseTextEmbedding

            model = SparseTextEmbedding(
                model_name=model_name,
                cache_dir=str(cache_dir) if cache_dir else None,
                k=1.2,
                b=0.75,
                avg_len=256.0,
                language="english",
                token_max_length=40,
                disable_stemmer=False,
            )
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[SparseVectorData]:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Sparse document input must contain non-empty text")
        results = list(self._model.embed(texts, batch_size=64))
        if len(results) != len(texts):
            raise RuntimeError(f"Expected {len(texts)} sparse vectors, received {len(results)}")
        return [self._convert(result) for result in results]

    def embed_query(self, text: str) -> SparseVectorData:
        if not text.strip():
            raise ValueError("Sparse query input must be non-empty")
        results = list(self._model.query_embed(text))
        if len(results) != 1:
            raise RuntimeError(f"Expected one sparse query vector, received {len(results)}")
        return self._convert(results[0])

    @staticmethod
    def _convert(result: SparseResult) -> SparseVectorData:
        indices = [int(index) for index in result.indices.tolist()]
        values = [float(value) for value in result.values.tolist()]
        if not indices or not values:
            raise ValueError("Sparse encoder returned an empty vector")
        if len(indices) != len(values):
            raise ValueError("Sparse vector indices and values have different lengths")
        return SparseVectorData(indices=indices, values=values)
