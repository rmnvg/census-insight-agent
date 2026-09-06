import streamlit as st

from frontend.models import Citation, Claim
from frontend.view_models import citations_for_claim, ordered_citations


def render_claims(claims: list[Claim], citations: list[Citation]) -> None:
    if not claims:
        return
    st.markdown("#### ✅ Verified claims")
    numbered = {
        item.citation_id: index for index, item in enumerate(ordered_citations(citations), 1)
    }
    for claim in claims:
        references = [
            f"`[{numbered[item.citation_id]}]`" for item in citations_for_claim(claim, citations)
        ]
        suffix = f" {' '.join(references)}" if references else ""
        st.markdown(f"- {claim.text}{suffix}")
        if claim.citation_ids and not references:
            st.caption("A referenced citation was unavailable in the public response.")


def render_citations(citations: list[Citation]) -> None:
    values = ordered_citations(citations)
    if not values:
        return
    st.markdown("#### 📖 Source citations")
    for index, citation in enumerate(values, 1):
        label = f"[{index}] {citation.document_title} — physical PDF page {citation.page_number}"
        with st.expander(label, expanded=len(citation.snippet) <= 280, icon="📄"):
            if citation.section_path:
                st.caption(" › ".join(citation.section_path))
            st.code(citation.snippet, language=None, wrap_lines=True)
            st.caption(f"Document ID: {citation.document_id} · Chunk ID: {citation.chunk_id}")
