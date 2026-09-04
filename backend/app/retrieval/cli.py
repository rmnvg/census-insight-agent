import argparse

from qdrant_client import QdrantClient

from backend.app.config import get_settings
from backend.app.providers.embeddings import VertexEmbeddingProvider
from backend.app.retrieval.service import HybridRetrievalService
from backend.app.retrieval.sparse import BM25SparseEncoder


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect hybrid retrieval evidence")
    parser.add_argument("search", nargs="?")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--document-id", action="append")
    parser.add_argument("--region", action="append")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    service = HybridRetrievalService(
        client=QdrantClient(url=settings.qdrant_url),
        collection_name=settings.qdrant_collection,
        dense_provider=VertexEmbeddingProvider(settings),
        sparse_encoder=BM25SparseEncoder(
            settings.sparse_embedding_model,
            cache_dir=settings.data_root / "processed" / "fastembed-cache",
        ),
    )
    response = service.search_response(
        query=args.query,
        document_ids=args.document_id,
        regions=args.region,
        top_k=args.top_k,
        debug=args.debug,
    )
    for rank, item in enumerate(response.evidence, start=1):
        section = " > ".join(item.section_path) or "(none)"
        print(
            f"{rank}. {item.document_id} p.{item.page_number} score={item.retrieval_score:.6f}\n"
            f"   section={section}\n   citation={item.citation_snippet!r}"
        )
    print(f"sufficiency={response.evidence_sufficiency.model_dump_json()}")
    if args.debug and response.debug:
        print(f"dense={response.debug.dense_candidates}")
        print(f"sparse={response.debug.sparse_candidates}")
        print(f"fused={response.debug.fused_ranking}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
