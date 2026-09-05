from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.app.config import Settings, get_settings


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Build the deterministic Gemini chat model using Vertex AI and ADC."""
    resolved = settings or get_settings()
    return ChatGoogleGenerativeAI(
        model=resolved.gemini_chat_model,
        project=resolved.google_cloud_project,
        location=resolved.google_cloud_location,
        vertexai=True,
        api_key=None,
        temperature=0,
        request_timeout=resolved.agent_provider_timeout_seconds,
        max_retries=0,
    )
