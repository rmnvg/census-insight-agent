from datetime import UTC, datetime
from unittest.mock import patch

from frontend.app import main
from frontend.models import (
    ArtifactFile,
    ArtifactSummary,
    BackendHealth,
    ChatResponse,
    Citation,
    Claim,
    CoverageReport,
    CoverageSummary,
    DocumentSummary,
    EvidenceSpan,
    ExecutorHealth,
    PageCoverageEntry,
    QdrantHealth,
    SessionResponse,
    TraceResponse,
)


def _document() -> DocumentSummary:
    return DocumentSummary(
        document_id="doc-mp",
        title="Madhya Pradesh PCA Highlights",
        region="Madhya Pradesh",
        source_checksum="a" * 64,
    )


def _coverage() -> CoverageSummary:
    return CoverageSummary(
        document=_document(),
        coverage=CoverageReport(
            document_id="doc-mp",
            pdf_page_count=100,
            indexed_pages=88,
            blank_decorative_pages=6,
            excluded_visual_pages=6,
            failed_mappings=0,
            percentage_pages_indexed=88.0,
            page_lists_by_status={"indexed_provided_markdown": [1, 2, 3]},
            pages=[PageCoverageEntry(page_number=1, status="indexed_provided_markdown")],
        ),
        limitations=[],
    )


def _citation() -> Citation:
    return Citation(
        citation_id="citation-1",
        document_id="doc-mp",
        document_title="Madhya Pradesh PCA Highlights",
        page_number=33,
        snippet="Balaghat | 1021",
        chunk_id="chunk-1",
        section_path=["Statement 6", "Sex Ratio"],
        evidence_span=EvidenceSpan(evidence_id="chunk-1", start_offset=0, end_offset=15),
    )


def _claim() -> Claim:
    return Claim(
        claim_id="claim-1",
        text="Balaghat recorded the highest sex ratio (1,021).",
        citation_ids=["citation-1"],
        metric="sex ratio",
        region="Balaghat",
        year=2011,
        value=1021.0,
        unit="other",
    )


def _artifact() -> ArtifactSummary:
    return ArtifactSummary(
        artifact_id="artifact-1",
        artifact_type="table",
        title="Sex Ratio in Districts of Madhya Pradesh, 2011",
        filename="table.md",
        media_type="text/markdown",
        byte_size=100,
        sha256="b" * 64,
        session_id="fixture-session",
        run_id="fixture-run",
        source_manifest_path="source-manifest.json",
        download_url="/artifacts/artifact-1",
    )


class RichFixtureClient:
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

    def documents(self) -> list[DocumentSummary]:
        return [_document()]

    def coverage(self, document_id: str) -> CoverageSummary:
        assert document_id == "doc-mp"
        return _coverage()

    def chat(self, session_id: str, message: str) -> ChatResponse:
        assert session_id == "fixture-session"
        assert message == "Test question"
        return ChatResponse(
            answer="Balaghat recorded the highest sex ratio (1,021).",
            claims=[_claim()],
            citations=[_citation()],
            artifacts=[_artifact()],
            trace_id="fixture-trace",
        )

    def trace(self, trace_id: str) -> TraceResponse:
        assert trace_id == "fixture-trace"
        return TraceResponse(session_id="fixture-session", run_id=trace_id)

    def download_artifact(self, session_id: str, artifact_id: str, filename: str) -> ArtifactFile:
        assert session_id == "fixture-session"
        assert artifact_id == "artifact-1"
        if filename == "table.csv":
            content = b"label,value\nBalaghat,1021\n"
            media_type = "text/csv"
        elif filename == "table.md":
            content = b"| District | Sex Ratio |\n|---|---|\n| Balaghat | 1021 |\n"
            media_type = "text/markdown"
        elif filename == "source-manifest.json":
            content = b'{"dataset_title": "t", "source_records": [], "computed_values": []}'
            media_type = "application/json"
        else:
            raise AssertionError(f"Unexpected artifact filename: {filename}")
        return ArtifactFile(filename=filename, media_type=media_type, content=content)


with patch("frontend.app._client", return_value=RichFixtureClient()):
    main()
