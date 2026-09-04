from collections.abc import Sequence
from typing import Protocol, cast

from google import genai
from google.genai import types

from backend.app.config import Settings, get_settings
from backend.app.providers.errors import (
    EmbeddingInputError,
    EmbeddingResponseError,
    ProviderRequestError,
    sanitize_provider_error,
)


class EmbeddingModels(Protocol):
    """Subset of the google-genai models API used by this adapter."""

    def embed_content(
        self,
        *,
        model: str,
        contents: str | list[str],
        config: types.EmbedContentConfig,
    ) -> types.EmbedContentResponse: ...


class EmbeddingClient(Protocol):
    """Provider-neutral client boundary for embedding calls."""

    models: EmbeddingModels


class VertexEmbeddingProvider:
    """Gemini embeddings through the official google-genai Vertex AI client."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: EmbeddingClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client or cast(
            "EmbeddingClient",
            genai.Client(
                vertexai=True,
                project=self.settings.google_cloud_project,
                location=self.settings.google_cloud_location,
            ),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks using the retrieval-document task type."""
        if not texts or any(not text.strip() for text in texts):
            raise EmbeddingInputError("Document embedding input must contain non-empty text")
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        """Embed one query using the retrieval-query task type."""
        if not text.strip():
            raise EmbeddingInputError("Query embedding input must be non-empty")
        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]

    def _embed(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        try:
            response = self._client.models.embed_content(
                model=self.settings.gemini_embedding_model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.settings.gemini_embedding_dimension,
                ),
            )
        except Exception as error:
            detail = sanitize_provider_error(error, self.settings)
            raise ProviderRequestError(f"Vertex embedding request failed: {detail}") from error

        embeddings = response.embeddings
        if embeddings is None or len(embeddings) != len(texts):
            received = 0 if embeddings is None else len(embeddings)
            raise EmbeddingResponseError(
                f"Expected {len(texts)} embeddings from Vertex AI, received {received}"
            )

        vectors = [
            self._require_values(embedding.values, index)
            for index, embedding in enumerate(embeddings)
        ]
        dimensions = {len(vector) for vector in vectors}
        expected = self.settings.gemini_embedding_dimension
        if dimensions != {expected}:
            found = ", ".join(str(dimension) for dimension in sorted(dimensions))
            raise EmbeddingResponseError(
                f"Expected consistent {expected}-dimension embeddings, received dimensions: {found}"
            )
        return vectors

    @staticmethod
    def _require_values(values: Sequence[float] | None, index: int) -> list[float]:
        if not values:
            raise EmbeddingResponseError(f"Embedding at response index {index} has no values")
        return list(values)
