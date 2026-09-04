import pytest
from pydantic import ValidationError

from backend.app.config import Settings


def test_settings_accept_complete_vertex_configuration() -> None:
    settings = Settings.model_validate(
        {
            "google_cloud_project": "test-project",
            "google_cloud_location": "global",
            "gemini_chat_model": "test-chat-model",
            "gemini_embedding_model": "test-embedding-model",
            "gemini_embedding_dimension": 768,
        }
    )

    assert settings.google_genai_use_vertexai is True
    assert settings.gemini_embedding_dimension == 768


def test_settings_reject_missing_project() -> None:
    with pytest.raises(ValidationError, match="GOOGLE_CLOUD_PROJECT is required"):
        Settings.model_validate(
            {
                "google_cloud_project": "",
                "gemini_chat_model": "test-chat-model",
                "gemini_embedding_model": "test-embedding-model",
            }
        )


def test_settings_reject_non_vertex_mode() -> None:
    with pytest.raises(ValidationError, match="GOOGLE_GENAI_USE_VERTEXAI must be true"):
        Settings.model_validate(
            {
                "google_genai_use_vertexai": False,
                "google_cloud_project": "test-project",
                "gemini_chat_model": "test-chat-model",
                "gemini_embedding_model": "test-embedding-model",
            }
        )
