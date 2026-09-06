from typing import Literal

import streamlit as st

from frontend.models import OperationalApiError, TraceResponse
from frontend.view_models import sanitize_trace


def render_trace(
    trace: TraceResponse | None,
    *,
    error: OperationalApiError | None = None,
    artifact_errors: list[str] | None = None,
) -> None:
    with st.expander("Execution details", icon="🔍", expanded=False):
        if error is not None:
            st.badge(error.error_code, color="red")
            if error.trace_id:
                st.caption(f"Trace ID: `{error.trace_id}`")
            st.caption(f"Retryable: {'yes' if error.retryable else 'no'}")
        if artifact_errors:
            for item in artifact_errors:
                st.badge(f"Artifact: {item}", color="orange")
        if trace is None:
            st.caption("Execution trace is unavailable. The answer above is preserved.")
            return
        status_color: Literal["green", "red"] = (
            "green" if trace.run_status == "completed" else "red"
        )
        st.caption(f"Run `{trace.run_id[:8]}…`")
        st.badge(trace.run_status, color=status_color)
        st.badge(trace.answer_status, color="blue")
        if trace.error_code:
            st.badge(f"Run error: {trace.error_code}", color="red")
        for index, event in enumerate(sanitize_trace(trace), 1):
            title = f"{index}. {event.event}"
            if event.node:
                title += f" ({event.node})"
            st.markdown(title)
            if event.details:
                st.json(event.details, expanded=False)
            if event.latency_ms is not None:
                st.caption(f"{event.latency_ms:.1f} ms")
