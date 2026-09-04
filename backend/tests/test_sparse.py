from collections.abc import Iterable

import pytest

from backend.app.retrieval.sparse import BM25SparseEncoder, SparseResult


class Array:
    def __init__(self, values: list[int] | list[float]) -> None:
        self._values = values

    def tolist(self) -> list[int] | list[float]:
        return self._values


class Result:
    def __init__(self, indices: list[int], values: list[float]) -> None:
        self.indices = Array(indices)
        self.values = Array(values)


class Model:
    def embed(self, documents: list[str], *, batch_size: int) -> Iterable[SparseResult]:
        return iter([Result([1, 9], [0.5, 1.0]) for _ in documents])

    def query_embed(self, query: str) -> Iterable[SparseResult]:
        return iter([Result([9], [1.0])])


def test_sparse_vectors_store_indices_and_values_separately() -> None:
    encoder = BM25SparseEncoder(model=Model())

    document = encoder.embed_documents(["Mysuru population"])[0]
    query = encoder.embed_query("Mysuru")

    assert document.indices == [1, 9]
    assert document.values == [0.5, 1.0]
    assert query.indices == [9]


def test_empty_sparse_vector_is_rejected() -> None:
    class EmptyModel(Model):
        def query_embed(self, query: str) -> Iterable[SparseResult]:
            return iter([Result([], [])])

    with pytest.raises(ValueError, match="empty vector"):
        BM25SparseEncoder(model=EmptyModel()).embed_query("anything")
