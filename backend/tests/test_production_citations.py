from pathlib import Path

import pytest

from backend.app.ingestion.chunking import chunk_pages
from backend.app.ingestion.coverage import (
    load_coverage_decisions,
    load_manual_transcriptions,
)
from backend.app.ingestion.discovery import discover_documents, source_checksum
from backend.app.ingestion.errors import DocumentPairingError
from backend.app.ingestion.pages import map_document_pages
from backend.app.ingestion.safety import validate_chunks_for_indexing


def test_all_production_dry_run_chunks_have_verbatim_citations_without_external_calls() -> None:
    data_root = Path("data").resolve()
    source_dir = data_root / "source"
    manifest_dir = data_root / "manifests"
    try:
        pairs = discover_documents(source_dir, manifest_dir)
    except DocumentPairingError:
        # The committed override manifest names the production census PDFs, but the PDFs
        # themselves are gitignored (supplied separately, not checked in). A clean checkout
        # without them is exactly the "documents are not present" case this test already
        # skips for below; discovery just raises before reaching that check in that case.
        pytest.skip("Production census source documents are not present")
    if len(pairs) != 3:
        pytest.skip("Production census source documents are not present")

    chunks = []
    for pair in pairs:
        checksum = source_checksum(pair.pdf_path)
        mapping = map_document_pages(
            document_id=pair.document_id,
            pdf_path=pair.pdf_path,
            markdown_path=pair.markdown_path,
            coverage_decisions=load_coverage_decisions(manifest_dir, pair.document_id, checksum),
            manual_transcriptions=load_manual_transcriptions(
                manifest_dir, pair.document_id, checksum
            ),
        )
        chunks.extend(
            chunk_pages(
                mapping.pages,
                document_title=pair.title,
                region=pair.region,
                source_checksum=checksum,
            )
        )

    validate_chunks_for_indexing(chunks)
    invalid = [chunk for chunk in chunks if chunk.metadata.citation_snippet not in chunk.text]
    assert len(chunks) == 2058
    assert invalid == []
