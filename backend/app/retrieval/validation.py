import argparse
import re
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from backend.app.config import get_settings


class CollectionValidationReport(BaseModel):
    collection: str
    expected_points: int
    exact_points: int
    scanned_points: int
    valid_points: int
    invalid_points: int
    duplicate_point_ids: int
    duplicate_chunk_ids: int
    dense_vectors_768: int
    non_empty_sparse_vectors: int
    excluded_pages_indexed: int
    unverified_ocr_points: int
    dense_query_results: int = 0
    sparse_query_results: int = 0
    client_server_compatible: bool = False
    invalid_by_field: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


REQUIRED_PAYLOAD_FIELDS = {
    "chunk_id",
    "text",
    "document_id",
    "document_title",
    "region",
    "page_number",
    "section_path",
    "citation_snippet",
    "extraction_method",
    "coverage_status",
    "source_checksum",
}


def validate_collection(
    client: QdrantClient,
    collection: str,
    *,
    expected_points: int,
    dense_dimensions: int = 768,
    excluded_pages: set[tuple[str, int]] | None = None,
) -> CollectionValidationReport:
    """Exhaustively validate stored vectors and citation payloads without mutation."""
    excluded_pages = excluded_pages or set()
    exact_points = client.count(collection, exact=True).count
    report = CollectionValidationReport(
        collection=collection,
        expected_points=expected_points,
        exact_points=exact_points,
        scanned_points=0,
        valid_points=0,
        invalid_points=0,
        duplicate_point_ids=0,
        duplicate_chunk_ids=0,
        dense_vectors_768=0,
        non_empty_sparse_vectors=0,
        excluded_pages_indexed=0,
        unverified_ocr_points=0,
    )
    if exact_points != expected_points:
        report.errors.append(f"exact point count {exact_points} != expected {expected_points}")

    point_ids: set[str] = set()
    chunk_ids: set[str] = set()
    sample_dense: list[float] | None = None
    sample_sparse: models.SparseVector | None = None
    offset: Any = None
    while True:
        points, offset = client.scroll(
            collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for point in points:
            report.scanned_points += 1
            errors: list[str] = []
            invalid_fields: set[str] = set()
            point_id = str(point.id)
            payload = point.payload or {}
            missing = REQUIRED_PAYLOAD_FIELDS - payload.keys()
            if missing:
                errors.append(f"missing payload fields {sorted(missing)}")
                invalid_fields.update(missing)
            chunk_id = payload.get("chunk_id")
            text = payload.get("text")
            snippet = payload.get("citation_snippet")
            page = payload.get("page_number")
            document_id = payload.get("document_id")
            if point_id in point_ids:
                report.duplicate_point_ids += 1
                errors.append("duplicate point id")
            point_ids.add(point_id)
            if not isinstance(chunk_id, str) or chunk_id in chunk_ids:
                report.duplicate_chunk_ids += 1
                errors.append("missing or duplicate chunk id")
                invalid_fields.add("chunk_id")
            else:
                chunk_ids.add(chunk_id)
            if not isinstance(page, int) or page < 1:
                errors.append("invalid one-based page number")
                invalid_fields.add("page_number")
            if not isinstance(document_id, str) or not document_id.strip():
                errors.append("missing document id")
                invalid_fields.add("document_id")
            checksum = payload.get("source_checksum")
            if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
                errors.append("missing or invalid source checksum")
                invalid_fields.add("source_checksum")
            if payload.get("coverage_status") not in {
                "indexed_provided_markdown",
                "indexed_pymupdf4llm_fallback",
                "approved_manual_transcription",
            }:
                errors.append("unsafe coverage status")
                invalid_fields.add("coverage_status")
            if not isinstance(text, str) or not isinstance(snippet, str) or snippet not in text:
                errors.append("citation snippet is not verbatim chunk text")
                invalid_fields.add("citation_snippet")
            if payload.get("extraction_method") == "unverified_ocr":
                report.unverified_ocr_points += 1
                errors.append("unverified OCR is indexed")
            if (
                isinstance(document_id, str)
                and isinstance(page, int)
                and (document_id, page) in excluded_pages
            ):
                report.excluded_pages_indexed += 1
                errors.append("excluded page is indexed")
            vectors = point.vector
            dense = vectors.get("dense") if isinstance(vectors, dict) else None
            sparse = vectors.get("sparse") if isinstance(vectors, dict) else None
            if (
                isinstance(dense, list)
                and len(dense) == dense_dimensions
                and all(isinstance(value, (float, int)) for value in dense)
            ):
                report.dense_vectors_768 += 1
                sample_dense = sample_dense or cast(list[float], dense)
            else:
                errors.append("missing or invalid dense vector")
            if isinstance(sparse, models.SparseVector) and sparse.indices and sparse.values:
                report.non_empty_sparse_vectors += 1
                sample_sparse = sample_sparse or sparse
            else:
                errors.append("missing or empty sparse vector")
            if errors:
                report.invalid_points += 1
                for field in invalid_fields:
                    report.invalid_by_field[field] = report.invalid_by_field.get(field, 0) + 1
                if len(report.errors) < 25:
                    report.errors.append(f"point {point_id}: {'; '.join(errors)}")
            else:
                report.valid_points += 1
        if offset is None:
            break
    if report.scanned_points != exact_points:
        report.errors.append(
            f"scanned point count {report.scanned_points} != exact count {exact_points}"
        )
    if sample_dense is not None and sample_sparse is not None:
        try:
            report.dense_query_results = len(
                client.query_points(
                    collection_name=collection,
                    query=sample_dense,
                    using="dense",
                    limit=1,
                    with_payload=False,
                    with_vectors=False,
                ).points
            )
            report.sparse_query_results = len(
                client.query_points(
                    collection_name=collection,
                    query=sample_sparse,
                    using="sparse",
                    limit=1,
                    with_payload=False,
                    with_vectors=False,
                ).points
            )
            report.client_server_compatible = bool(
                report.dense_query_results and report.sparse_query_results
            )
            if not report.client_server_compatible:
                report.errors.append("read-only dense or sparse compatibility query was empty")
        except Exception as error:
            report.errors.append(
                f"read-only client/server compatibility query failed: {type(error).__name__}"
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Qdrant payload/vector audit")
    parser.add_argument("--expected-points", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = get_settings()
    excluded = {
        ("census-2011-karnataka-pca-highlights", page)
        for page in (1, 13, 19, 21, 24, 28, 31, 33, 37, 43, 48, 51, 56)
    } | {
        ("census-2011-odisha-pca-highlights", 134),
        *(("census-2011-madhya-pradesh-pca-highlights", page) for page in (7, 19, 47, 63)),
    }
    report = validate_collection(
        QdrantClient(url=settings.qdrant_url),
        settings.qdrant_collection,
        expected_points=args.expected_points,
        dense_dimensions=settings.gemini_embedding_dimension,
        excluded_pages=excluded,
    )
    rendered = report.model_dump_json(indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return int(bool(report.errors or report.invalid_points))


if __name__ == "__main__":
    raise SystemExit(main())
