from collections.abc import MutableMapping
from contextlib import suppress
from typing import Any, cast

import streamlit as st

from frontend.api_client import ApiClientError, CensusApiClient
from frontend.config import UiSettings
from frontend.models import ChatResponse, OperationalApiError, TraceResponse
from frontend.renderers.answer import render_response
from frontend.renderers.sidebar import render_sidebar
from frontend.renderers.traces import render_trace
from frontend.state import (
    begin_submission,
    finish_submission,
    initialize_ui_state,
    replace_conversation,
)

_ASSISTANT_AVATAR = "📊"


def _client() -> CensusApiClient:
    return CensusApiClient(UiSettings.from_environment())


def _state() -> MutableMapping[str, Any]:
    return cast(MutableMapping[str, Any], st.session_state)


def _ensure_session(api: CensusApiClient) -> str:
    session_id = st.session_state.session_id
    if session_id is None:
        session_id = api.create_session().session_id
        replace_conversation(_state(), session_id)
    return str(session_id)


def _render_history(api: CensusApiClient) -> None:
    for item in st.session_state.messages:
        avatar = _ASSISTANT_AVATAR if item["role"] == "assistant" else None
        with st.chat_message(item["role"], avatar=avatar):
            if item["role"] == "user":
                st.markdown(item["message"])
                continue
            if item.get("error") is not None:
                error = OperationalApiError.model_validate(item["error"])
                trace = (
                    TraceResponse.model_validate(item["trace"])
                    if item.get("trace") is not None
                    else None
                )
                st.warning(error.message)
                if st.session_state.show_execution_details:
                    render_trace(trace, error=error)
                continue
            response = ChatResponse.model_validate(item["response"])
            trace = (
                TraceResponse.model_validate(item["trace"])
                if item.get("trace") is not None
                else None
            )
            render_response(
                response,
                api,
                trace,
                show_trace=bool(st.session_state.show_execution_details),
            )


def _submit(api: CensusApiClient, session_id: str, prompt: str) -> bool:
    fingerprint = begin_submission(_state(), prompt)
    if fingerprint is None:
        return False
    st.session_state.messages.append({"role": "user", "message": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        with st.status("Consulting the indexed Census evidence…", expanded=True) as status:
            response = api.chat(session_id, prompt)
            status.update(label="Validating citations and artifacts…")
            trace = None
            with suppress(ApiClientError):
                trace = api.trace(response.trace_id)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "response": response.model_dump(mode="json"),
                    "trace": trace.model_dump(mode="json") if trace else None,
                }
            )
            status.update(label="Complete", state="complete", expanded=False)
        with st.chat_message("assistant", avatar=_ASSISTANT_AVATAR):
            render_response(
                response,
                api,
                trace,
                show_trace=bool(st.session_state.show_execution_details),
            )
    except ApiClientError as error:
        trace = None
        if error.error.trace_id:
            with suppress(ApiClientError):
                trace = api.trace(error.error.trace_id)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "error": error.error.model_dump(mode="json"),
                "trace": trace.model_dump(mode="json") if trace else None,
            }
        )
        with st.chat_message("assistant", avatar=_ASSISTANT_AVATAR):
            st.warning(error.error.message)
            if st.session_state.show_execution_details:
                render_trace(trace, error=error.error)
    finally:
        finish_submission(_state(), fingerprint)
    return True


def main() -> None:
    st.set_page_config(
        page_title="Census Insight Agent",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_ui_state(_state())
    api = _client()
    try:
        try:
            session_id = _ensure_session(api)
        except ApiClientError as error:
            st.title("Census Insight Agent")
            st.error(error.error.message)
            render_trace(None, error=error.error)
            st.stop()

        if render_sidebar(api, session_id):
            try:
                replacement = api.create_session().session_id
            except ApiClientError as error:
                st.sidebar.error(error.error.message)
            else:
                replace_conversation(_state(), replacement)
                st.rerun()

        title_col, badge_col = st.columns([5, 1], vertical_alignment="center")
        with title_col:
            st.title("📊 Census Insight Agent")
        with badge_col:
            st.badge("Citation-safe", color="green", icon="✅")
        st.caption(
            "Ask questions about the supplied Census reports. Answers use physical PDF-page "
            "citations and citation-safe evidence."
        )
        st.divider()
        _render_history(api)

        queued = st.session_state.queued_prompt
        prompt = queued or st.chat_input(
            "Ask about the indexed Census reports…",
            disabled=bool(st.session_state.submission_in_progress),
        )
        if prompt:
            _submit(api, session_id, str(prompt))
    finally:
        api.close()


if __name__ == "__main__":
    main()
