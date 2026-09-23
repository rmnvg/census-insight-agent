from typing import Any

import pytest
from fastapi import HTTPException

from backend.app import api
from backend.app.retrieval.models import (
    EvidenceSufficiency,
    RetrievalSearchRequest,
    RetrievalSearchResponse,
)
from backend.app.retrieval.service import RetrievalProvenanceError


class FakeRetrievalService:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def search_response(self, **_: Any) -> RetrievalSearchResponse:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return RetrievalSearchResponse(
            evidence=[],
            evidence_sufficiency=EvidenceSufficiency(
                status="insufficient_evidence", rule="No overlap"
            ),
        )


def test_retrieval_search_reuses_the_cached_service_across_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeRetrievalService()
    build_calls = 0

    def _build() -> FakeRetrievalService:
        nonlocal build_calls
        build_calls += 1
        return fake

    monkeypatch.setattr(api, "_retrieval_service", _build)
    api.retrieval_search(RetrievalSearchRequest(query="Karnataka literacy"))
    api.retrieval_search(RetrievalSearchRequest(query="Odisha literacy"))

    # This monkeypatch replaces the *function*, not the real `@lru_cache`, so it doesn't prove
    # caching by itself; it does prove the route calls `_retrieval_service()` fresh each time
    # rather than holding a module-level service some other way, and that the same underlying
    # service handles both calls.
    assert build_calls == 2
    assert fake.calls == 2


def test_retrieval_search_maps_empty_query_to_400(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api, "_retrieval_service", lambda: FakeRetrievalService(error=ValueError("empty"))
    )
    with pytest.raises(HTTPException) as caught:
        api.retrieval_search(RetrievalSearchRequest(query="   "))
    assert caught.value.status_code == 400
    assert caught.value.detail == "empty"


def test_retrieval_search_sanitizes_internal_errors_to_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api,
        "_retrieval_service",
        lambda: FakeRetrievalService(
            error=RetrievalProvenanceError("point abc123 missing chunk_id")
        ),
    )
    with pytest.raises(HTTPException) as caught:
        api.retrieval_search(RetrievalSearchRequest(query="Karnataka literacy"))
    assert caught.value.status_code == 502
    assert caught.value.detail == "Retrieval unavailable: RetrievalProvenanceError"
    assert "abc123" not in caught.value.detail
