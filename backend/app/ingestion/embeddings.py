import hashlib
import json
import sqlite3
import time
from pathlib import Path

from backend.app.ingestion.models import DocumentChunk
from backend.app.ingestion.safety import validate_chunks_for_indexing
from backend.app.providers.embeddings import VertexEmbeddingProvider
from backend.app.providers.errors import ProviderRequestError


class EmbeddingCache:
    """Persistent content-addressed dense embedding cache."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(cache_key TEXT PRIMARY KEY, vector_json TEXT NOT NULL)"
        )

    @staticmethod
    def key(*, text: str, model: str, dimensions: int, task_type: str) -> str:
        payload = json.dumps(
            {
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "model": model,
                "dimensions": dimensions,
                "task_type": task_type,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> list[float] | None:
        row = self._connection.execute(
            "SELECT vector_json FROM embeddings WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        return None if row is None else [float(value) for value in json.loads(row[0])]

    def put(self, cache_key: str, vector: list[float]) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO embeddings(cache_key, vector_json) VALUES (?, ?)",
            (cache_key, json.dumps(vector)),
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()


class CachedDenseEmbedder:
    """Batched Vertex document embeddings with retries and content caching."""

    def __init__(
        self,
        provider: VertexEmbeddingProvider,
        cache: EmbeddingCache,
        *,
        batch_size: int = 16,
        max_attempts: int = 3,
    ) -> None:
        self.provider = provider
        self.cache = cache
        self.batch_size = batch_size
        self.max_attempts = max_attempts

    def embed_chunks(self, chunks: list[DocumentChunk]) -> tuple[list[list[float]], int]:
        """Embed only chunks that satisfy citation-safe indexing invariants."""
        validate_chunks_for_indexing(chunks)
        return self._embed_texts([chunk.text for chunk in chunks])

    def _embed_texts(self, texts: list[str]) -> tuple[list[list[float]], int]:
        settings = self.provider.settings
        vectors: list[list[float] | None] = [None] * len(texts)
        missing: list[tuple[int, str, str]] = []
        for index, text in enumerate(texts):
            key = self.cache.key(
                text=text,
                model=settings.gemini_embedding_model,
                dimensions=settings.gemini_embedding_dimension,
                task_type="RETRIEVAL_DOCUMENT",
            )
            cached = self.cache.get(key)
            if cached is None or len(cached) != settings.gemini_embedding_dimension:
                missing.append((index, text, key))
            else:
                vectors[index] = cached

        request_count = 0
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            embedded = self._embed_with_retry([item[1] for item in batch])
            request_count += 1
            for (index, _, key), vector in zip(batch, embedded, strict=True):
                self.cache.put(key, vector)
                vectors[index] = vector
        if any(vector is None for vector in vectors):
            raise RuntimeError("Dense embedding cache assembly failed")
        return ([vector for vector in vectors if vector is not None], request_count)

    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.provider.embed_documents(texts)
            except ProviderRequestError:
                if attempt == self.max_attempts:
                    raise
                time.sleep(2 ** (attempt - 1))
        raise RuntimeError("Dense embedding retry loop exited unexpectedly")
