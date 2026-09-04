import argparse
from pathlib import Path

from backend.app.config import get_settings
from backend.app.ingestion.service import IngestionService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Citation-safe census document ingestion")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("--source-dir", type=Path, required=True)
    ingest.add_argument("--document-id")
    ingest.add_argument("--dry-run", action="store_true")
    ingest.add_argument("--rebuild", action="store_true")
    ingest.add_argument("--review-override", action="store_true")
    ingest.add_argument("--report-output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = get_settings()
    service = IngestionService(settings) if args.dry_run else IngestionService.live(settings)
    report = service.ingest(
        source_dir=args.source_dir,
        document_id=args.document_id,
        dry_run=args.dry_run,
        rebuild=args.rebuild,
        review_override=args.review_override,
        report_output=args.report_output,
    )
    print(report.model_dump_json(indent=2))
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
