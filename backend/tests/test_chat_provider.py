from unittest.mock import Mock, patch

from langchain_core.language_models import BaseChatModel

from backend.app.config import Settings
from backend.app.providers.chat import get_chat_model


def test_chat_provider_constructs_explicit_vertex_model() -> None:
    settings = Settings.model_validate(
        {
            "google_cloud_project": "test-project",
            "google_cloud_location": "global",
            "gemini_chat_model": "test-chat-model",
            "gemini_embedding_model": "test-embedding-model",
        }
    )
    model = Mock(spec=BaseChatModel)

    with patch(
        "backend.app.providers.chat.ChatGoogleGenerativeAI",
        return_value=model,
    ) as constructor:
        result = get_chat_model(settings)

    assert result is model
    constructor.assert_called_once_with(
        model="test-chat-model",
        project="test-project",
        location="global",
        vertexai=True,
        api_key=None,
        temperature=0,
        request_timeout=60.0,
        max_retries=0,
    )
