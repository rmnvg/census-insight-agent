from typing import Literal

from pydantic import BaseModel


class ProviderHealthCheck(BaseModel):
    """Result of a single provider verification operation."""

    check: Literal["text_generation"]
    passed: bool
    detail: str


class EmbeddingVerificationResult(BaseModel):
    """Result of verifying document and query embeddings."""

    check: Literal["embeddings"] = "embeddings"
    passed: bool
    document_dimensions: int | None = None
    query_dimensions: int | None = None
    detail: str


class ToolCallVerificationResult(BaseModel):
    """Result of inspecting a structured model tool call."""

    check: Literal["tool_calling"] = "tool_calling"
    passed: bool
    tool_name: str | None = None
    region: str | None = None
    detail: str
