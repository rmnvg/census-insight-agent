import streamlit as st

from frontend.api_client import ApiClientError, CensusApiClient
from frontend.models import ArtifactSummary, PendingArtifact
from frontend.view_models import parse_csv, validate_png, validate_source_manifest


def render_artifacts(
    artifacts: list[ArtifactSummary | PendingArtifact], api: CensusApiClient
) -> list[str]:
    errors: list[str] = []
    for artifact in artifacts:
        if isinstance(artifact, PendingArtifact):
            st.info(artifact.message)
            continue
        try:
            _render_artifact(artifact, api)
        except (ApiClientError, ValueError) as error:
            errors.append(f"{artifact.title}: {type(error).__name__}")
            st.warning("Artifact could not be displayed.")
    return errors


_TYPE_ICON = {"chart": "📈", "table": "📋"}


def _render_artifact(artifact: ArtifactSummary, api: CensusApiClient) -> None:
    icon = _TYPE_ICON.get(artifact.artifact_type, "🗂️")
    with st.container(border=True):
        st.markdown(f"#### {icon} {artifact.title}")
        st.caption(
            f"Validated {artifact.artifact_type} artifact generated from "
            "citation-bound source data."
        )
        if artifact.artifact_type == "chart":
            chart = api.download_artifact(artifact.session_id, artifact.artifact_id, "chart.png")
            validate_png(chart.content)
            st.image(chart.content, caption=artifact.title, use_container_width=True)
            st.download_button(
                "Download chart PNG",
                chart.content,
                file_name="chart.png",
                mime="image/png",
                key=f"png-{artifact.artifact_id}",
                icon="⬇️",
            )
            csv_file = api.download_artifact(
                artifact.session_id, artifact.artifact_id, "plotted-data.csv"
            )
            with st.expander("Plotted data", icon="🔢"):
                st.dataframe(parse_csv(csv_file.content), use_container_width=True, hide_index=True)
            st.download_button(
                "Download plotted CSV",
                csv_file.content,
                file_name="plotted-data.csv",
                mime="text/csv",
                key=f"csv-{artifact.artifact_id}",
                icon="⬇️",
            )
        elif artifact.artifact_type == "table":
            csv_file = api.download_artifact(artifact.session_id, artifact.artifact_id, "table.csv")
            st.dataframe(parse_csv(csv_file.content), use_container_width=True, hide_index=True)
            st.download_button(
                "Download table CSV",
                csv_file.content,
                file_name="table.csv",
                mime="text/csv",
                key=f"table-csv-{artifact.artifact_id}",
                icon="⬇️",
            )
            markdown = api.download_artifact(artifact.session_id, artifact.artifact_id, "table.md")
            with st.expander("Plain Markdown table", icon="📝"):
                st.code(markdown.content.decode("utf-8"), language=None, wrap_lines=True)
            st.download_button(
                "Download table Markdown",
                markdown.content,
                file_name="table.md",
                mime="text/markdown",
                key=f"table-md-{artifact.artifact_id}",
                icon="⬇️",
            )
        else:
            raise ValueError("Unsupported artifact type")
        manifest = api.download_artifact(
            artifact.session_id, artifact.artifact_id, "source-manifest.json"
        )
        validate_source_manifest(manifest.content)
        st.download_button(
            "Download source manifest",
            manifest.content,
            file_name="source-manifest.json",
            mime="application/json",
            key=f"manifest-{artifact.artifact_id}",
            icon="⬇️",
        )
        with st.expander("Artifact metadata", icon="🏷️"):
            st.json(
                {
                    "artifact_id": artifact.artifact_id,
                    "type": artifact.artifact_type,
                    "size_bytes": artifact.byte_size,
                    "source_manifest": "Validated and available for download",
                },
                expanded=False,
            )
