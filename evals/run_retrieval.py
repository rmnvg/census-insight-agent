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
    return int(metrics.unsafe_evidence_count > 0 or metrics.citation_page_accuracy < 1)


if __name__ == "__main__":
    raise SystemExit(main())
