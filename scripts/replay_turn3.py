import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, cast

from qdrant_client import QdrantClient

from backend.app.agent.citations import validate_and_materialize_citations
from backend.app.agent.memory import (
    evidence_references,
    select_source_support_turn,
    source_support_draft,
)
from backend.app.agent.models import ValidatedClaimTurn
from backend.app.agent.persistence import TraceStore
from backend.app.agent.skills import SkillRegistry
from backend.app.agent.tools import AgentTools
from backend.app.config import get_settings
from backend.app.retrieval.models import RetrievedEvidence
from backend.app.retrieval.qdrant_store import QdrantStore
from scripts.smoke_agent_memory import assert_turn_3, assert_turn_4


def replay(turn: ValidatedClaimTurn, evidence: list[RetrievedEvidence]) -> dict[str, Any]:
    draft, calculations = source_support_draft(turn.claims, evidence)
    validation = validate_and_materialize_citations(draft, evidence, calculations)
    public = {
        "answer": draft.answer_markdown,
        "claims": [claim.model_dump(mode="json") for claim in validation.claims],
        "citations": [item.model_dump(mode="json") for item in validation.citations],
        "refusal": not validation.valid,
    }
    if validation.valid:
        assert_turn_3(public)
    out_of_scope = {"answer": "out of scope", "claims": [], "citations": [], "refusal": True}
    assert_turn_4(out_of_scope)
    return {
        "referenced_run_id": turn.run_id,
        "gemini_calls": 0,
        "embedding_calls": 0,
        "qdrant_writes": 0,
        "citation_validation_passed": validation.valid,
        "turn_3_smoke_assertion_passed": validation.valid,
        "turn_4_assertion_passed": True,
        "resolved_claims": [
            {
                "claim_id": claim.claim_id,
                "region": claim.region,
                "metric": claim.metric,
                "year": claim.year,
                "displayed_value": claim.displayed_value,
                "unit": claim.unit,
            }
            for claim in turn.claims
        ],
        "citations": [
            {
                "document_id": item.document_id,
                "page_number": item.page_number,
                "chunk_id": item.chunk_id,
                "source_checksum": next(
                    value.source_checksum for value in evidence if value.chunk_id == item.chunk_id
                ),
                "exact_substring": item.snippet
                == next(value.text for value in evidence if value.chunk_id == item.chunk_id)[
                    item.evidence_span.start_offset : item.evidence_span.end_offset
                ],
            }
            for item in validation.citations
        ],
        "errors": validation.errors,
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    history = TraceStore(args.workspace).recover_validated_claim_history(args.session_id)
    turn, clarification = select_source_support_turn(
        "Which source pages support those values?", history
    )
    if turn is None:
        raise RuntimeError(clarification or "No validated comparison exists")
    references = evidence_references(turn.claims)
    store = QdrantStore(
        QdrantClient(url=args.qdrant_url),
        args.collection,
        dense_dimensions=settings.gemini_embedding_dimension,
        dense_model=settings.gemini_embedding_model,
        sparse_model=settings.sparse_embedding_model,
    )
    tools = AgentTools(
        cast(Any, None),
        store,
        SkillRegistry(settings.skills_dir),
        settings.data_root,
    )
    evidence = await tools.get_evidence_by_ids(list(references), references)
    return replay(turn, evidence)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay Turn 3 without Gemini or embeddings")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))
    parser.add_argument("--qdrant-url", default="http://qdrant:6333")
    parser.add_argument("--collection", default="census_documents")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["citation_validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
