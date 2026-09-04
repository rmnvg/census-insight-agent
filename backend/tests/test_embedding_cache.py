from pathlib import Path

from backend.app.ingestion.embeddings import EmbeddingCache


def test_embedding_cache_key_records_model_dimensions_and_task(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    document_key = cache.key(
        text="same content",
        model="embedding-model",
        dimensions=768,
        task_type="RETRIEVAL_DOCUMENT",
    )
    query_key = cache.key(
        text="same content",
        model="embedding-model",
        dimensions=768,
        task_type="RETRIEVAL_QUERY",
    )

    cache.put(document_key, [1.0, 2.0])

    assert document_key != query_key
    assert cache.get(document_key) == [1.0, 2.0]
    assert cache.get(query_key) is None
    cache.close()
