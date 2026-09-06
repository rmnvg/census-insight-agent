#!/usr/bin/env python3
"""Safely verify or intentionally initialize the assignment corpus."""

import argparse
from pathlib import Path

from qdrant_client import QdrantClient

from backend.app.config import get_settings
from backend.app.ingestion.discovery import discover_documents
from backend.app.ingestion.service import IngestionService
from backend.app.retrieval.qdrant_store import QdrantStore
from backend.app.retrieval.validation import validate_collection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-paid-calls", action="store_true")
    parser.add_argument("--expected-points", type=int, default=2058)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()
    settings = get_settings()
    source_dir = args.source_dir or settings.data_root / "source"
    report_output = (
        args.report_output or settings.data_root / "processed/initialization-report.json"
    )
    client = QdrantClient(url=settings.qdrant_url)
    store = QdrantStore(
        client,
        settings.qdrant_collection,
        dense_dimensions=settings.gemini_embedding_dimension,
        dense_model=settings.gemini_embedding_model,
        sparse_model=settings.sparse_embedding_model,
    )
    exists = client.collection_exists(settings.qdrant_collection)
    if exists:
        store.validate_schema()
        count = client.count(settings.qdrant_collection, exact=True).count
        if count == args.expected_points:
            validation = validate_collection(
                client,
                settings.qdrant_collection,
                expected_points=args.expected_points,
                dense_dimensions=settings.gemini_embedding_dimension,
            )
            if validation.errors:
                print("FAIL: collection count matches, but read-only validation failed.")
                return 1
            print(f"PASS: existing collection is valid with {count} points; ingestion skipped.")
            return 0
        if count:
            print(
                f"FAIL: existing collection has {count} points, expected {args.expected_points}; "
                "it was not modified. Investigate before any explicit rebuild."
            )
            return 1
    pairs = discover_documents(source_dir.resolve(), (settings.data_root / "manifests").resolve())
    if len(pairs) != 3:
        print(f"FAIL: expected three source PDF/Markdown pairs, discovered {len(pairs)}.")
        return 1
    if not args.allow_paid_calls:
        print(
            "SAFE STOP: the collection needs initialization. Review a dry run, then rerun this "
            "command with --allow-paid-calls; Vertex embedding charges will apply."
        )
        return 2
    ingestion_report = IngestionService.live(settings).ingest(
        source_dir=source_dir,
        dry_run=False,
        rebuild=False,
        report_output=report_output,
    )
    if ingestion_report.failures:
        print(
            f"FAIL: ingestion reported {len(ingestion_report.failures)} failure(s); "
            "collection was not rebuilt."
        )
        return 1
    count = client.count(settings.qdrant_collection, exact=True).count
    print(f"PASS: initialization completed with {count} points. Report: {report_output}")
    return int(count != args.expected_points)


if __name__ == "__main__":
    raise SystemExit(main())
