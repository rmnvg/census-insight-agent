import json
from collections.abc import Callable
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from frontend.config import UiSettings
from frontend.models import (
    ArtifactFile,
    ArtifactListing,
    ArtifactSummary,
    BackendHealth,
    ChatRequest,
    ChatResponse,
    CoverageSummary,
    DocumentSummary,
    ExecutorHealth,
    OperationalApiError,
    QdrantHealth,
    SessionResponse,
    TraceResponse,
)

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
_ALLOWED_FILES = {
    "chart.png": "image/png",
    "plotted-data.csv": "text/csv",
    "table.csv": "text/csv",
    "table.md": "text/markdown",
    "source-manifest.json": "application/json",
}


class ApiClientError(RuntimeError):
    def __init__(self, error: OperationalApiError, status_code: int | None = None) -> None:
        super().__init__(error.message)
        self.error = error
        self.status_code = status_code


class CensusApiClient:
    def __init__(
        self,
        settings: UiSettings,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.client = httpx.Client(
            base_url=settings.api_base_url,
            timeout=httpx.Timeout(
                settings.read_timeout_seconds,
                connect=settings.connect_timeout_seconds,
            ),
            transport=transport,
            follow_redirects=False,
        )

    def close(self) -> None:
        self.client.close()

    def create_session(self) -> SessionResponse:
        return self._request("POST", "/sessions", SessionResponse)

    def get_session(self, session_id: str) -> SessionResponse:
        return self._safe_get(f"/sessions/{session_id}", SessionResponse)

    def chat(self, session_id: str, message: str) -> ChatResponse:
        request = ChatRequest(session_id=session_id, message=message)
        return self._request("POST", "/chat", ChatResponse, json_body=request.model_dump())

    def trace(self, trace_id: str) -> TraceResponse:
        return self._safe_get(f"/runs/{trace_id}/trace", TraceResponse)

    def backend_health(self) -> BackendHealth:
        return self._safe_get("/health", BackendHealth)

    def executor_health(self) -> ExecutorHealth:
        return self._safe_get("/health/executor", ExecutorHealth)

    def qdrant_health(self) -> QdrantHealth:
        return self._safe_get("/health/qdrant", QdrantHealth)

    def documents(self) -> list[DocumentSummary]:
        response = self._safe_get_raw("/documents")
        try:
            return [DocumentSummary.model_validate(item) for item in response.json()]
        except (ValidationError, TypeError, ValueError) as error:
            raise self._malformed_error() from error

    def coverage(self, document_id: str) -> CoverageSummary:
        return self._safe_get(f"/documents/{document_id}/coverage", CoverageSummary)

    def artifacts(self, session_id: str) -> ArtifactListing:
        return self._safe_get(f"/sessions/{session_id}/artifacts", ArtifactListing)

    def artifact(self, session_id: str, artifact_id: str) -> ArtifactSummary:
        return self._safe_get(f"/sessions/{session_id}/artifacts/{artifact_id}", ArtifactSummary)

    def download_artifact(self, session_id: str, artifact_id: str, filename: str) -> ArtifactFile:
        expected_type = _ALLOWED_FILES.get(filename)
        if expected_type is None or filename != filename.rsplit("/", 1)[-1] or ".." in filename:
            raise ApiClientError(
                OperationalApiError(
                    error_code="UNSAFE_ARTIFACT_FILENAME",
                    message="This artifact filename is not permitted.",
                )
            )
        response = self._safe_get_raw(
            f"/sessions/{session_id}/artifacts/{artifact_id}/files/{filename}"
        )
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        declared_size = response.headers.get("content-length")
        if declared_size and int(declared_size) > self.settings.max_artifact_bytes:
            raise self._artifact_error("ARTIFACT_TOO_LARGE", "Artifact exceeds the UI size limit.")
        if len(response.content) > self.settings.max_artifact_bytes:
            raise self._artifact_error("ARTIFACT_TOO_LARGE", "Artifact exceeds the UI size limit.")
        if content_type != expected_type:
            raise self._artifact_error(
                "ARTIFACT_MIME_INVALID", "Artifact media type did not match its declared format."
            )
        return ArtifactFile(filename=filename, media_type=content_type, content=response.content)

    def _safe_get(self, path: str, model: type[ResponseModel]) -> ResponseModel:
        response = self._safe_get_raw(path)
        return self._validate(response, model)

    def _safe_get_raw(self, path: str) -> httpx.Response:
        return self._request_raw("GET", path, attempts=2)

    def _request(
        self,
        method: str,
        path: str,
        model: type[ResponseModel],
        *,
        json_body: dict[str, object] | None = None,
    ) -> ResponseModel:
        return self._validate(self._request_raw(method, path, json_body=json_body), model)

    def _request_raw(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, object] | None = None,
        attempts: int = 1,
    ) -> httpx.Response:
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                response = self.client.request(method, path, json=json_body)
            except (httpx.TimeoutException, httpx.NetworkError) as error:
                last_error = error
                continue
            if response.is_success:
                return response
            self._raise_http_error(response)
        code = (
            "BACKEND_TIMEOUT"
            if isinstance(last_error, httpx.TimeoutException)
            else "BACKEND_UNAVAILABLE"
        )
        raise ApiClientError(
            OperationalApiError(
                error_code=code,
                message="The Census service is temporarily unavailable.",
                retryable=True,
            )
        ) from last_error

    @staticmethod
    def _validate(response: httpx.Response, model: type[ResponseModel]) -> ResponseModel:
        try:
            return model.model_validate(response.json())
        except (json.JSONDecodeError, ValidationError, ValueError) as error:
            raise CensusApiClient._malformed_error() from error

    @staticmethod
    def _raise_http_error(response: httpx.Response) -> None:
        try:
            error = OperationalApiError.model_validate(response.json())
        except (json.JSONDecodeError, ValidationError, ValueError):
            error = OperationalApiError(
                error_code=f"HTTP_{response.status_code}",
                message="The Census service returned an unexpected error.",
                retryable=response.status_code >= 500,
            )
        raise ApiClientError(error, response.status_code)

    @staticmethod
    def _malformed_error() -> ApiClientError:
        return ApiClientError(
            OperationalApiError(
                error_code="MALFORMED_API_RESPONSE",
                message="The Census service returned an invalid response.",
            )
        )

    @staticmethod
    def _artifact_error(code: str, message: str) -> ApiClientError:
        return ApiClientError(OperationalApiError(error_code=code, message=message))


ClientFactory = Callable[[], CensusApiClient]
