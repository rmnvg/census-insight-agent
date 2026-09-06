from typing import Literal, cast

from backend.app.agent.models import AgentResponse, AnswerClaim, Citation, ClaimUnit, EvidenceSpan
from backend.app.execution.contracts import ArtifactDataset, ArtifactDescriptor, SourceRecord
from backend.app.retrieval.models import RetrievedEvidence


def artifact_response(
    dataset: ArtifactDataset,
    descriptor: ArtifactDescriptor,
    evidence: list[RetrievedEvidence],
    trace_id: str,
    *,
    ranking_winner: SourceRecord | None = None,
    rank_direction: Literal["max", "min"] | None = None,
) -> AgentResponse:
    by_id = {item.chunk_id: item for item in evidence}
    citations: list[Citation] = []
    citation_by_source: dict[str, str] = {}
    for source in dataset.source_records:
        item = by_id[source.chunk_id]
        start = item.text.index(source.exact_supporting_quote)
        citation_id = f"artifact-citation-{len(citations) + 1}"
        citation_by_source[source.source_record_id] = citation_id
        citations.append(
            Citation(
                citation_id=citation_id,
                document_id=item.document_id,
                document_title=item.document_title,
                page_number=item.page_number,
                snippet=source.exact_supporting_quote,
                chunk_id=item.chunk_id,
                section_path=item.section_path,
                evidence_span=EvidenceSpan(
                    evidence_id=item.chunk_id,
                    start_offset=start,
                    end_offset=start + len(source.exact_supporting_quote),
                ),
            )
        )
    claims = [
        AnswerClaim(
            claim_id=f"artifact-source-{index}",
            text=f"{source.row_id} {source.field}: {source.raw_value} {source.unit or ''}".strip(),
            citation_ids=[citation_by_source[source.source_record_id]],
            value=source.normalized_numeric_value,
            unit=cast(
                ClaimUnit,
                source.unit if source.unit in {"percent", "count", "ratio"} else "other",
            ),
        )
        for index, source in enumerate(dataset.source_records, start=1)
    ]
    for index, computed in enumerate(dataset.computed_values, start=1):
        claims.append(
            AnswerClaim(
                claim_id=f"artifact-computed-{index}",
                text=f"{computed.field}: {computed.result}",
                citation_ids=list(
                    dict.fromkeys(
                        citation_by_source[source_id]
                        for source_id in computed.input_source_record_ids
                    )
                ),
                document_derived=True,
                value=computed.result,
                unit="other",
            )
        )
    lead = ""
    if ranking_winner is not None:
        direction_word = "lowest" if rank_direction == "min" else "highest"
        unit_suffix = f" {ranking_winner.unit}" if ranking_winner.unit else ""
        metric = ranking_winner.metric or "value"
        lead = (
            f"{ranking_winner.region} recorded the {direction_word} {metric} "
            f"({ranking_winner.raw_value}{unit_suffix}) among {dataset.title}. "
        )
    return AgentResponse(
        answer_markdown=f"{lead}Created {dataset.title}. Download: {descriptor.download_url}",
        claims=claims,
        citations=citations,
        artifacts=[descriptor],
        trace_id=trace_id,
    )
