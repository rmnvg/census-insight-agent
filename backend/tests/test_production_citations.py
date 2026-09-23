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
    # 2058 -> 2112 after `_leading_header_line_count` (chunking.py) started repeating a table's
    # sub-header row (e.g. "Total | Rural | Urban") on every split fragment, not just the first —
    # verified live against this real corpus: 600 chunks across 65 pages now carry that row, where
    # only the first fragment of each such page used to. Repeating more header text per fragment
    # leaves less room for data rows before `max_characters`, so some tables now split into a few
    # more fragments than before; this is the expected shape of the fix, not drift. The live
    # Qdrant collection still holds 2058 points from before this fix — `make verify-offline`
    # (`--expected-points 2058` in the Makefile) intentionally still checks against that until a
    # deliberate, billable re-ingest (`scripts/initialize_corpus.py --allow-paid-calls --rebuild`)
    # is run to replace it; do not "fix" that number to 2112 without also actually re-ingesting.
    assert len(chunks) == 2112
    assert invalid == []
