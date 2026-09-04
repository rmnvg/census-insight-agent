import pytest
from google.genai import types

from backend.app.config import Settings
from backend.app.providers.embeddings import EmbeddingModels, VertexEmbeddingProvider
from backend.app.providers.errors import EmbeddingInputError, EmbeddingResponseError


class FakeModels:
    def __init__(self, response: types.EmbedContentResponse) -> None:
        self.response = response

    def embed_content(
        self,
        *,
        model: str,
        contents: str | list[str],
        config: types.EmbedContentConfig,
    ) -> types.EmbedContentResponse:
        return self.response


class FakeClient:
    def __init__(self, response: types.EmbedContentResponse) -> None:
        self.models: EmbeddingModels = FakeModels(response)


def make_settings() -> Settings:
    return Settings.model_validate(
        {
            "google_cloud_project": "test-project",
            "gemini_chat_model": "test-chat-model",
            "gemini_embedding_model": "test-embedding-model",
            "gemini_embedding_dimension": 3,
        }
    )


def test_embedding_provider_rejects_empty_inputs() -> None:
    response = types.EmbedContentResponse(embeddings=[])
    provider = VertexEmbeddingProvider(make_settings(), FakeClient(response))

    with pytest.raises(EmbeddingInputError, match="non-empty"):
        provider.embed_documents([])
    with pytest.raises(EmbeddingInputError, match="non-empty"):
        provider.embed_query("  ")


def test_embedding_provider_rejects_inconsistent_dimensions() -> None:
    response = types.EmbedContentResponse(
        embeddings=[
            types.ContentEmbedding(values=[1.0, 2.0, 3.0]),
            types.ContentEmbedding(values=[1.0, 2.0]),
        ]
    )
    provider = VertexEmbeddingProvider(make_settings(), FakeClient(response))

    with pytest.raises(EmbeddingResponseError, match="received dimensions: 2, 3"):
        provider.embed_documents(["first", "second"])


def test_embedding_provider_rejects_missing_values() -> None:
    response = types.EmbedContentResponse(embeddings=[types.ContentEmbedding(values=None)])
    provider = VertexEmbeddingProvider(make_settings(), FakeClient(response))

    with pytest.raises(EmbeddingResponseError, match="has no values"):
        provider.embed_query("query")
