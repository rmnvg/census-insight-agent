#!/usr/bin/env python3
import os
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from backend.app.config import Settings, get_settings
from backend.app.providers.chat import get_chat_model
from backend.app.providers.contracts import (
    EmbeddingVerificationResult,
    ProviderHealthCheck,
    ToolCallVerificationResult,
)
from backend.app.providers.embeddings import VertexEmbeddingProvider
from backend.app.providers.errors import sanitize_provider_error
from backend.app.providers.verification import verify_tool_call


@tool
def get_test_metric(region: str) -> str:
    """Return a synthetic test metric for a named region."""
    return f"Synthetic metric for {region}"


def _prepare_local_adc(settings: Settings) -> None:
    """Use the host ADC path when the configured container mount is unavailable."""
    container_path = Path(settings.google_application_credentials)
    host_path = Path(settings.google_credentials_host_path)
    if (
        not container_path.is_file()
        and settings.google_credentials_host_path
        and host_path.is_file()
    ):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(host_path)


def _text_is_nonempty(message: AIMessage) -> bool:
    content = message.content
    if isinstance(content, str):
        return bool(content.strip())
    return bool(content)


def run_checks() -> list[
    ProviderHealthCheck | ToolCallVerificationResult | EmbeddingVerificationResult
]:
    """Run live, potentially billable checks against the configured Vertex models."""
    settings = get_settings()
    _prepare_local_adc(settings)
    results: list[
        ProviderHealthCheck | ToolCallVerificationResult | EmbeddingVerificationResult
    ] = []

    try:
        chat_model = get_chat_model(settings)
        response = chat_model.invoke(
            "Reply with one short sentence confirming service availability."
        )
        passed = isinstance(response, AIMessage) and _text_is_nonempty(response)
        results.append(
            ProviderHealthCheck(
                check="text_generation",
                passed=passed,
                detail=(
                    "Received non-empty Gemini response" if passed else "Gemini response was empty"
                ),
            )
        )
    except Exception as error:
        results.append(
            ProviderHealthCheck(
                check="text_generation",
                passed=False,
                detail=sanitize_provider_error(error, settings),
            )
        )
        chat_model = None

    if chat_model is None:
        results.append(
            ToolCallVerificationResult(
                passed=False,
                detail="Tool check skipped because chat model construction failed",
            )
        )
    else:
        try:
            tool_model = chat_model.bind_tools([get_test_metric])
            response = tool_model.invoke(
                "Call get_test_metric for Karnataka. Do not answer the question yourself."
            )
            if not isinstance(response, AIMessage):
                raise TypeError("Gemini returned an unexpected message type")
            results.append(
                verify_tool_call(
                    response,
                    expected_name="get_test_metric",
                    expected_region="Karnataka",
                )
            )
        except Exception as error:
            results.append(
                ToolCallVerificationResult(
                    passed=False,
                    detail=sanitize_provider_error(error, settings),
                )
            )

    try:
        embeddings = VertexEmbeddingProvider(settings)
        document = embeddings.embed_documents(["Karnataka is a state in India."])[0]
        query = embeddings.embed_query("Which state is Karnataka?")
        expected = settings.gemini_embedding_dimension
        passed = bool(document) and bool(query) and len(document) == len(query) == expected
        results.append(
            EmbeddingVerificationResult(
                passed=passed,
                document_dimensions=len(document),
                query_dimensions=len(query),
                detail=(
                    "Embedding dimensions matched"
                    if passed
                    else "Embedding dimensions did not match"
                ),
            )
        )
    except Exception as error:
        results.append(
            EmbeddingVerificationResult(
                passed=False,
                detail=sanitize_provider_error(error, settings),
            )
        )
    return results


def main() -> int:
    try:
        results = run_checks()
    except Exception:
        print("FAIL configuration: invalid Vertex provider settings; review the environment")
        return 1

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.check}: {result.detail}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
