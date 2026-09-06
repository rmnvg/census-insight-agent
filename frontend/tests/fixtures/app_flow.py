from datetime import UTC, datetime
from unittest.mock import patch

import streamlit as st

from frontend.app import main
from frontend.models import (
    BackendHealth,
    ChatResponse,
    ExecutorHealth,
    QdrantHealth,
    SessionResponse,
    TraceResponse,
)


class FixtureClient:
    def close(self) -> None:
        return None

    def create_session(self) -> SessionResponse:
        now = datetime.now(UTC)
        return SessionResponse(session_id="fixture-session", created_at=now, updated_at=now)

    def backend_health(self) -> BackendHealth:
        return BackendHealth(status="ok", service="backend")

    def executor_health(self) -> ExecutorHealth:
        return ExecutorHealth(status="ok", detail="ready")

    def qdrant_health(self) -> QdrantHealth:
        return QdrantHealth(status="ok", collection="census_documents", detail="ready")

    def documents(self) -> list[object]:
        return []

    def chat(self, session_id: str, message: str) -> ChatResponse:
        assert session_id == "fixture-session"
        assert message == "Test question"
        st.session_state.fixture_chat_calls = st.session_state.get("fixture_chat_calls", 0) + 1
        return ChatResponse(answer="Rendered answer", trace_id="fixture-trace")

    def trace(self, trace_id: str) -> TraceResponse:
        assert trace_id == "fixture-trace"
        return TraceResponse(session_id="fixture-session", run_id=trace_id)


with patch("frontend.app._client", return_value=FixtureClient()):
    main()
