"""Build table stores for documents that are already indexed, without re-embedding them.

Ingestion writes a document's table store itself. This command covers a corpus indexed before
table stores existed: it re-derives the indexed pages from the source files, exactly as
ingestion does, and binds rows to the chunks currently in Qdrant, which may predate the current
chunker. No model or embedding calls are made.
"""

import argparse
import json

from qdrant_client import QdrantClient, models

from backend.app.config import Settings, get_settings
from backend.app.ingestion.coverage import load_coverage_decisions, load_manual_transcriptions
from backend.app.ingestion.discovery import discover_documents, source_checksum
from backend.app.ingestion.pages import map_document_pages
from backend.app.tables.extraction import IndexedChunk, extract_document_tables
from backend.app.tables.store import write_document_tables


def _indexed_chunks(client: QdrantClient, collection: str, document_id: str) -> list[dict]:
    payloads: list[dict] = []
    offset = None
    while True:
        page, offset = client.scroll(
            collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            ),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        payloads.extend(record.payload or {} for record in page)
        if offset is None:
            return payloads


def build_from_index(settings: Settings, document_id: str | None = None) -> list[dict[str, object]]:
    manifest_dir = (settings.data_root / "manifests").resolve()
    client = QdrantClient(url=settings.qdrant_url)
    summaries: list[dict[str, object]] = []
    for pair in discover_documents(settings.data_root / "source", manifest_dir):
        if document_id and pair.document_id != document_id:
            continue
        checksum = source_checksum(pair.pdf_path)
        payloads = _indexed_chunks(client, settings.qdrant_collection, pair.document_id)
        if not payloads:
            summaries.append({"document_id": pair.document_id, "status": "not_indexed"})
            continue
        if {payload.get("source_checksum") for payload in payloads} != {checksum}:
            # Rows would be bound to evidence from a different version of the PDF.
            summaries.append({"document_id": pair.document_id, "status": "index_out_of_date"})
            continue
        mapping = map_document_pages(
            document_id=pair.document_id,
            pdf_path=pair.pdf_path,
            markdown_path=pair.markdown_path,
            coverage_decisions=load_coverage_decisions(manifest_dir, pair.document_id, checksum),
            manual_transcriptions=load_manual_transcriptions(
                manifest_dir, pair.document_id, checksum
            ),
        )
        tables = extract_document_tables(
            document_id=pair.document_id,
            region=pair.region,
            source_checksum=checksum,
            pages=mapping.pages,
            chunks=[
                IndexedChunk(
                    chunk_id=str(payload["chunk_id"]),
                    page_number=int(payload["page_number"]),
                    text=str(payload["text"]),
                )
                for payload in payloads
            ],
        )
        write_document_tables(settings.data_root, tables)
        summaries.append(
            {
                "document_id": pair.document_id,
                "status": "written",
                "tables": len(tables.tables),
                "rows": sum(len(table.rows) for table in tables.tables),
                "uncitable_rows": sum(
                    1 for table in tables.tables for row in table.rows if row.chunk_id is None
                ),
                "malformed_rows": sum(table.malformed_rows for table in tables.tables),
            }
        )
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description="Build structured table stores")
    parser.add_argument("--document-id")
    args = parser.parse_args()
    summaries = build_from_index(get_settings(), args.document_id)
    print(json.dumps(summaries, indent=2))
    return 0 if all(item["status"] == "written" for item in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
