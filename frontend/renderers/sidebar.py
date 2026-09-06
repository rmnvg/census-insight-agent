from collections.abc import MutableMapping
from typing import Any, Literal, cast

import streamlit as st

from frontend.api_client import ApiClientError, CensusApiClient
from frontend.models import CoverageSummary, SidebarSnapshot
from frontend.state import cached_sidebar_snapshot, clear_sidebar_snapshot
from frontend.view_models import VISUAL_LIMITATION

EXAMPLE_QUESTIONS = [
    "What was Karnataka’s literacy rate in 2011?",
    "How does that compare with Odisha?",
    "Which source pages support those values?",
    "Summarize the key population findings for Odisha.",
    "Create a bar chart comparing Karnataka and Odisha literacy rates.",
    "Create a table comparing total, rural and urban literacy rates.",
    "Check whether the population totals in this table are internally consistent.",
    "What was France’s unemployment rate in 2011?",
]


def render_sidebar(api: CensusApiClient, session_id: str) -> bool:
    st.sidebar.title("Census Insight Agent")
    st.sidebar.caption("Citation-grounded analysis of the supplied 2011 Census reports.")
    st.sidebar.text(f"Session: {session_id[:8]}…")
    new_conversation = st.sidebar.button("New conversation", use_container_width=True)
    st.sidebar.toggle("Show execution details", key="show_execution_details")
    state = cast(MutableMapping[str, Any], st.session_state)
    if st.sidebar.button("Refresh status", use_container_width=True):
        clear_sidebar_snapshot(state)
    snapshot = SidebarSnapshot.model_validate(
        cached_sidebar_snapshot(state, lambda: _load_snapshot(api))
    )
    _render_health(snapshot)
    _render_documents(snapshot)
    with st.sidebar.expander("Example questions"):
        for index, question in enumerate(EXAMPLE_QUESTIONS):
            if st.button(question, key=f"example-{index}", use_container_width=True):
                st.session_state.queued_prompt = question
    return new_conversation


def _load_snapshot(api: CensusApiClient) -> SidebarSnapshot:
    statuses: dict[str, Literal["ok", "error"]] = {}
    for name, operation in (
        ("backend_status", api.backend_health),
        ("executor_status", api.executor_health),
        ("document_status", api.qdrant_health),
    ):
        try:
            statuses[name] = operation().status
        except ApiClientError:
            statuses[name] = "error"
    try:
        documents = api.documents()
    except ApiClientError:
        documents = []
    coverage: dict[str, CoverageSummary] = {}
    for document in documents:
        try:
            coverage[document.document_id] = api.coverage(document.document_id)
        except ApiClientError:
            continue
    return SidebarSnapshot(
        backend_status=statuses["backend_status"],
        executor_status=statuses["executor_status"],
        document_status=statuses["document_status"],
        documents=documents,
        coverage_by_document=coverage,
    )


def _render_health(snapshot: SidebarSnapshot) -> None:
    st.sidebar.markdown("#### Service status")
    checks = [
        ("Backend", snapshot.backend_status),
        ("Executor", snapshot.executor_status),
        ("Documents", snapshot.document_status),
    ]
    for label, status in checks:
        icon = "✅" if status == "ok" else "⚠️"
        st.sidebar.caption(f"{icon} {label}")


def _render_documents(snapshot: SidebarSnapshot) -> None:
    if not snapshot.documents:
        st.sidebar.warning("Document metadata is temporarily unavailable.")
        return
    st.sidebar.markdown("#### Source reports")
    for document in snapshot.documents:
        coverage = snapshot.coverage_by_document.get(document.document_id)
        with st.sidebar.expander(document.title):
            st.caption(document.region)
            if coverage is None:
                st.caption("Coverage information unavailable.")
                continue
            report = coverage.coverage
            excluded = report.blank_decorative_pages + report.excluded_visual_pages
            st.metric("Indexed coverage", f"{report.percentage_pages_indexed:.2f}%")
            st.caption(
                f"{report.indexed_pages}/{report.pdf_page_count} physical PDF pages indexed · "
                f"{excluded} excluded"
            )
            if report.excluded_visual_pages:
                st.info(VISUAL_LIMITATION)
