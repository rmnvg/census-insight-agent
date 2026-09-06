import json
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import pytest

from frontend.api_client import ApiClientError, CensusApiClient
from frontend.config import UiSettings


def session_payload(identifier: str = "session-1") -> dict[str, str]:
    now = datetime.now(UTC).isoformat()
    return {"session_id": identifier, "created_at": now, "updated_at": now}


def chat_payload() -> dict[str, object]:
    return {
        "answer": "Karnataka literacy was 75.36 percent.",
        "claims": [],
        "citations": [],
        "artifacts": [],
        "limitations": [],
        "refusal": False,
        "trace_id": "trace-1",
    }


def client(
    handler: Callable[[httpx.Request], httpx.Response], *, maximum: int = 1024
) -> CensusApiClient:
    return CensusApiClient(
        UiSettings(api_base_url="http://test", max_artifact_bytes=maximum),
        transport=httpx.MockTransport(handler),
    )


def test_session_creation_reuse_and_minimal_chat_body() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/sessions":
            return httpx.Response(200, json=session_payload())
        return httpx.Response(200, json=chat_payload())

    api = client(handler)
    session = api.create_session()
    response = api.chat(session.session_id, "Current question")
    assert response.trace_id == "trace-1"
    body = json.loads(calls[-1].content)
    assert body == {"session_id": "session-1", "message": "Current question"}
    assert len([item for item in calls if item.url.path == "/chat"]) == 1


def test_post_chat_is_never_automatically_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ApiClientError, match="temporarily unavailable"):
        client(handler).chat("session", "question")
    assert calls == 1


def test_safe_get_has_one_bounded_retry() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("temporary", request=request)
        return httpx.Response(200, json={"status": "ok", "service": "backend"})

    assert client(handler).backend_health().status == "ok"
    assert calls == 2


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_typed_operational_errors(status: int) -> None:
    payload = {
        "error_code": "MODEL_OUTPUT_INVALID",
        "message": "Safe message",
        "session_id": "session",
        "trace_id": "trace",
        "retryable": status in {503, 504},
    }
    with pytest.raises(ApiClientError) as caught:
        client(lambda request: httpx.Response(status, json=payload)).chat("session", "question")
    assert caught.value.status_code == status
    assert caught.value.error.trace_id == "trace"


@pytest.mark.parametrize(
    "response",
    [httpx.Response(200, content=b"not-json"), httpx.Response(200, json={"answer": "missing"})],
)
def test_malformed_or_schema_invalid_response(response: httpx.Response) -> None:
    with pytest.raises(ApiClientError) as caught:
        client(lambda request: response).chat("session", "question")
    assert caught.value.error.error_code == "MALFORMED_API_RESPONSE"


@pytest.mark.parametrize(
    "filename",
    ["generated.py", "input.json", "../chart.png", "/tmp/chart.png", "chart.html", "x.js"],
)
def test_unsafe_artifact_names_are_rejected_without_http(filename: str) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    with pytest.raises(ApiClientError) as caught:
        client(handler).download_artifact("session", "artifact", filename)
    assert caught.value.error.error_code == "UNSAFE_ARTIFACT_FILENAME"
    assert calls == 0


def test_artifact_mime_and_size_validation() -> None:
    wrong = client(
        lambda request: httpx.Response(200, content=b"png", headers={"content-type": "text/html"})
    )
    with pytest.raises(ApiClientError) as caught:
        wrong.download_artifact("session", "artifact", "chart.png")
    assert caught.value.error.error_code == "ARTIFACT_MIME_INVALID"

    large = client(
        lambda request: httpx.Response(
            200, content=b"x" * 20, headers={"content-type": "image/png"}
        ),
        maximum=10,
    )
    with pytest.raises(ApiClientError) as caught:
        large.download_artifact("session", "artifact", "chart.png")
    assert caught.value.error.error_code == "ARTIFACT_TOO_LARGE"


def test_valid_artifact_download() -> None:
    api = client(
        lambda request: httpx.Response(
            200,
            content=b"\x89PNG\r\n\x1a\ncontent",
            headers={"content-type": "image/png"},
        )
    )
    result = api.download_artifact("session", "artifact", "chart.png")
    assert result.media_type == "image/png"
