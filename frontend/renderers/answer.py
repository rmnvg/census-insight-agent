import streamlit as st

from frontend.api_client import CensusApiClient
from frontend.models import ChatResponse, TraceResponse
from frontend.renderers.artifacts import render_artifacts
from frontend.renderers.citations import render_citations, render_claims
from frontend.renderers.traces import render_trace


def render_response(
    response: ChatResponse,
    api: CensusApiClient,
    trace: TraceResponse | None,
    *,
    show_trace: bool,
) -> None:
    if response.refusal:
        st.info(response.answer)
    else:
        st.markdown(response.answer)
    render_claims(response.claims, response.citations)
    render_citations(response.citations)
    if response.limitations:
        with st.expander("Coverage limitations"):
            for limitation in response.limitations:
                st.write(limitation)
    artifact_errors = render_artifacts(response.artifacts, api)
    if show_trace:
        render_trace(trace, artifact_errors=artifact_errors)
