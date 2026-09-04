from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from the environment or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
    )

    google_genai_use_vertexai: bool = True
    google_cloud_project: str = ""
    google_cloud_location: str = "global"
    google_application_credentials: str = "/var/secrets/google/adc.json"
    google_credentials_host_path: str = ""
    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_embedding_dimension: int = 768
    qdrant_url: str = "http://localhost:6333"

    @model_validator(mode="after")
    def validate_vertex_configuration(self) -> "Settings":
        """Reject incomplete or non-Vertex model configuration."""
        if not self.google_genai_use_vertexai:
            raise ValueError("GOOGLE_GENAI_USE_VERTEXAI must be true; no fallback is configured")
        if not self.google_cloud_project.strip():
            raise ValueError("GOOGLE_CLOUD_PROJECT is required")
        if not self.google_cloud_location.strip():
            raise ValueError("GOOGLE_CLOUD_LOCATION is required")
        if not self.gemini_chat_model.strip():
            raise ValueError("GEMINI_CHAT_MODEL is required")
        if not self.gemini_embedding_model.strip():
            raise ValueError("GEMINI_EMBEDDING_MODEL is required")
        if self.gemini_embedding_dimension <= 0:
            raise ValueError("GEMINI_EMBEDDING_DIMENSION must be positive")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
