import argparse
import json
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from backend.app.agent.calculations import (
    add_deterministic_derived_claims,
    enrich_source_claims,
    validated_calculation,
)
from backend.app.agent.citations import validate_and_materialize_citations
from backend.app.agent.models import CalculationRequest, DraftAnswer, DraftClaim
from backend.app.retrieval.models import RetrievedEvidence
from scripts.smoke_agent_memory import assert_turn_2


def replay(trace: dict[str, Any], evidence: list[RetrievedEvidence]) -> dict[str, Any]:
    resolved = next(
        event["details"]["query"] for event in trace["events"] if event["event"] == "resolved_query"
    )
    draft_event = next(event for event in trace["events"] if event["event"] == "structured_draft")
    draft = DraftAnswer(
        answer_markdown=draft_event["details"]["answer_preview"],
        claims=[DraftClaim.model_validate(item) for item in draft_event["details"]["claims"]],
    )
    calculations = []
    for call in trace["tool_calls"]:
        if call["tool_name"] != "calculate" or call["status"] != "ok":
            continue
        arguments = dict(call["arguments"])
        arguments.pop("unit", None)
        calculation = validated_calculation(
            resolved, CalculationRequest.model_validate(arguments), evidence
        )
        if calculation is not None:
            calculations.append(calculation)
    draft = enrich_source_claims(draft, evidence, resolved)
    draft = add_deterministic_derived_claims(draft, calculations)
    validation = validate_and_materialize_citations(draft, evidence, calculations)
    public = {
        "answer": "\n\n".join(claim.text for claim in validation.claims),
        "claims": [claim.model_dump(mode="json") for claim in validation.claims],
        "citations": [item.model_dump(mode="json") for item in validation.citations],
        "refusal": not validation.valid,
    }
    if validation.valid:
        assert_turn_2(public, trace)
    return {
        "run_id": trace["run_id"],
        "gemini_calls": 0,
        "qdrant_writes": 0,
        "citation_validation_passed": validation.valid,
        "smoke_assertion_passed": validation.valid,
        "calculation": [item.model_dump(mode="json") for item in calculations],
        "claims": [claim.model_dump(mode="json") for claim in validation.claims],
        "citations": [
            {
                "document_id": item.document_id,
                "page_number": item.page_number,
                "chunk_id": item.chunk_id,
                "start_offset": item.evidence_span.start_offset,
                "end_offset": item.evidence_span.end_offset,
                "span_length": len(item.snippet),
                "exact_substring": item.snippet
                == evidence_by_id(evidence)[item.chunk_id].text[
                    item.evidence_span.start_offset : item.evidence_span.end_offset
                ],
            }
            for item in validation.citations
        ],
        "quote_diagnostics": [
            item.model_dump(mode="json") for item in validation.quote_diagnostics
        ],
        "errors": validation.errors,
    }


def evidence_by_id(evidence: list[RetrievedEvidence]) -> dict[str, RetrievedEvidence]:
    return {item.chunk_id: item for item in evidence}


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a saved Turn 2 without Gemini")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--qdrant-url", default="http://qdrant:6333")
    parser.add_argument("--collection", default="census_documents")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    assessment = next(event for event in trace["events"] if event["event"] == "evidence_assessment")
    evidence_ids = assessment["details"]["selected_evidence_ids"]
    points = QdrantClient(url=args.qdrant_url).retrieve(
        args.collection, ids=evidence_ids, with_payload=True, with_vectors=False
    )
    points_by_id = {str(point.id): point for point in points}
    evidence = [
        RetrievedEvidence.model_validate(
            {**(points_by_id[evidence_id].payload or {}), "retrieval_score": 0.0}
        )
        for evidence_id in evidence_ids
    ]
    report = replay(trace, evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["citation_validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
