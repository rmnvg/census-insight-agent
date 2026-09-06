import streamlit as st

from frontend.models import OperationalApiError, TraceResponse
from frontend.view_models import sanitize_trace


def render_trace(
    trace: TraceResponse | None,
    *,
    error: OperationalApiError | None = None,
    artifact_errors: list[str] | None = None,
) -> None:
    with st.expander("Execution details", expanded=False):
        if error is not None:
            st.text(f"Error code: {error.error_code}")
            if error.trace_id:
                st.text(f"Trace ID: {error.trace_id}")
            st.text(f"Retryable: {'yes' if error.retryable else 'no'}")
        if artifact_errors:
            for item in artifact_errors:
                st.text(f"Artifact: {item}")
        if trace is None:
            st.caption("Execution trace is unavailable. The answer above is preserved.")
            return
        st.caption(f"Run {trace.run_id[:8]}… · {trace.run_status} · {trace.answer_status}")
        if trace.error_code:
            st.text(f"Run error: {trace.error_code}")
        for index, event in enumerate(sanitize_trace(trace), 1):
            title = f"{index}. {event.event}"
            if event.node:
                title += f" ({event.node})"
            st.markdown(title)
            if event.details:
                st.json(event.details, expanded=False)
            if event.latency_ms is not None:
                st.caption(f"{event.latency_ms:.1f} ms")
