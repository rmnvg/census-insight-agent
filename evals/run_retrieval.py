import argparse
import json
from pathlib import Path

from qdrant_client import QdrantClient

from backend.app.config import get_settings
from backend.app.evaluation import RetrievalEvaluationCase, evaluate_retrieval
from backend.app.providers.embeddings import VertexEmbeddingProvider
from backend.app.retrieval.service import HybridRetrievalService
from backend.app.retrieval.sparse import BM25SparseEncoder


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real-corpus hybrid retrieval evaluation")
    parser.add_argument("--cases", type=Path, default=Path("evals/real_corpus_cases.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/retrieval-evaluation-report.json")
    )
    # Defaults set just below the last known-good real run against this corpus
    # (data/processed/retrieval-evaluation-report-4c.json: recall@5=0.8125, recall@10=1.0,
    # mean_reciprocal_rank=0.7574), with margin for the residual retrieval nondeterminism
    # FAILURE_ANALYSIS.md documents (near-duplicate chunks occasionally rank differently between
    # runs) rather than exact-value flakiness. Previously this script only failed on unsafe
    # evidence or citation-page accuracy — a real recall/ranking regression produced a report
    # nobody was required to read, with an exit code that stayed 0.
    parser.add_argument("--min-recall-at-5", type=float, default=0.7)
    parser.add_argument("--min-recall-at-10", type=float, default=0.9)
    parser.add_argument("--min-mean-reciprocal-rank", type=float, default=0.6)
    args = parser.parse_args()
    settings = get_settings()
    cases = [
        RetrievalEvaluationCase.model_validate(item)
        for item in json.loads(args.cases.read_text(encoding="utf-8"))
    ]
    service = HybridRetrievalService(
        client=QdrantClient(url=settings.qdrant_url),
        collection_name=settings.qdrant_collection,
        dense_provider=VertexEmbeddingProvider(settings),
        sparse_encoder=BM25SparseEncoder(
            settings.sparse_embedding_model,
            cache_dir=settings.data_root / "processed" / "fastembed-cache",
        ),
    )
    metrics = evaluate_retrieval(
        cases,
        lambda case: service.search_response(
            query=case.query,
            document_ids=case.document_ids,
            regions=case.regions,
            top_k=case.top_k,
            debug=True,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(metrics.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(metrics.model_dump_json(indent=2))
    failures = {
        "unsafe_evidence_count > 0": metrics.unsafe_evidence_count > 0,
        "citation_page_accuracy < 1": metrics.citation_page_accuracy < 1,
        f"recall_at_5 < {args.min_recall_at_5}": metrics.recall_at_5 < args.min_recall_at_5,
        f"recall_at_10 < {args.min_recall_at_10}": metrics.recall_at_10 < args.min_recall_at_10,
        f"mean_reciprocal_rank < {args.min_mean_reciprocal_rank}": (
            metrics.mean_reciprocal_rank < args.min_mean_reciprocal_rank
        ),
    }
    failed = [reason for reason, triggered in failures.items() if triggered]
    if failed:
        print(f"FAIL: {'; '.join(failed)}")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
