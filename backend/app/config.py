from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
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
    qdrant_collection: str = "census_documents"
    data_root: Path = Path("data")
    sparse_embedding_model: str = "Qdrant/bm25"
    ingestion_version: str = "1"
    admin_ingestion_enabled: bool = False
    document_upload_enabled: bool = True
    document_upload_max_bytes: int = Field(default=50 * 1024 * 1024, ge=1024)
    document_upload_max_pages: int = Field(default=400, ge=1)
    # Empty indexes uploads inside the API process. A redis:// URL hands them to the durable
    # Celery worker (`backend/app/ingestion/worker.py`), which retries and survives restarts.
    ingestion_broker_url: str = ""
    ingestion_max_retries: int = Field(default=3, ge=0, le=10)
    workspace_root: Path = Path("workspace")
    # Empty keeps conversation state in `workspace/checkpoints.sqlite` (one API process). A
    # postgresql:// URL moves checkpoints, sessions, and session locks to Postgres for replicas.
    state_database_url: str = ""
    skills_dir: Path = Path("skills")
    # The longest legitimate path (artifact + one repair + refusal) takes 16 supersteps; the old
    # limit of exactly 16 turned it into an untyped GraphRecursionError. Also in compose/.env.
    agent_max_steps: int = 24
    agent_max_tool_calls: int = 12
    agent_memory_turn_threshold: int = 12
    agent_provider_timeout_seconds: float = 120.0
    agent_request_timeout_seconds: float = 240.0
    agent_provider_max_retries: int = 2
    agent_assessment_max_characters: int = 12_000
    agent_assessment_max_chunks: int = 12
    execution_queue_root: Path = Path("workspace/execution-queue")
    artifact_execution_timeout_seconds: float = Field(default=30, gt=0, le=60)
    artifact_max_files: int = Field(default=8, ge=1, le=8)
    artifact_max_total_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)
    # Langfuse tracing is on only when all three are set. Content (prompts, evidence, answers)
    # is withheld unless capture is explicitly enabled.
    langfuse_base_url: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: SecretStr = SecretStr("")
    langfuse_capture_content: bool = False
    langfuse_environment: str = "local"

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
        if self.state_database_url and not self.state_database_url.startswith(
            ("postgresql://", "postgres://")
        ):
            raise ValueError("STATE_DATABASE_URL must be a postgresql:// URL")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
