import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from qdrant_client import models

from backend.app.agent.calculations import rounded_subtraction
from backend.app.agent.models import EvidenceReference
from backend.app.agent.skills import RuntimeSkill, SkillMetadata, SkillRegistry
from backend.app.ingestion.models import CoverageLimitation, DocumentCoverageReport
from backend.app.retrieval.models import DocumentSummary, RetrievedEvidence
from backend.app.retrieval.qdrant_store import QdrantStore
from backend.app.retrieval.service import HybridRetrievalService


class SearchDocumentsInput(BaseModel):
    query: str = Field(min_length=1)
    document_ids: list[str] | None = None
    regions: list[str] | None = None
    top_k: int = Field(default=10, ge=1, le=20)


class ArithmeticInput(BaseModel):
    operation: Literal["sum", "difference", "percentage_difference"]
    values: list[float] = Field(min_length=2, max_length=20)


class StaleEvidenceError(RuntimeError):
    """Previously validated evidence no longer matches the current Qdrant payload."""


class AgentTools:
    """Narrow validated tools; none exposes vectors or arbitrary filesystem access."""

    def __init__(
        self,
        retrieval: HybridRetrievalService,
        store: QdrantStore,
        skills: SkillRegistry,
        data_root: Path,
        *,
        timeout_seconds: float = 60,
    ) -> None:
        self.retrieval = retrieval
        self.store = store
        self.skills = skills
        self.data_root = data_root
        self.timeout_seconds = timeout_seconds

    async def search_documents(self, value: SearchDocumentsInput) -> list[RetrievedEvidence]:
        return await asyncio.wait_for(
            asyncio.to_thread(
                self.retrieval.search,
                value.query,
                value.document_ids,
                value.regions,
                value.top_k,
            ),
            timeout=self.timeout_seconds,
        )

    async def get_evidence_by_ids(
        self,
        evidence_ids: list[str],
        expected: dict[str, EvidenceReference],
    ) -> list[RetrievedEvidence]:
        """Rehydrate trusted chunks by point ID without embeddings or semantic search."""
        unique_ids = list(dict.fromkeys(evidence_ids))
        records = await asyncio.wait_for(
            asyncio.to_thread(
                self.store.client.retrieve,
                self.store.collection_name,
                ids=unique_ids,
                with_payload=True,
                with_vectors=False,
            ),
            timeout=self.timeout_seconds,
        )
        by_id = {str(record.id): record.payload or {} for record in records}
        if missing := [evidence_id for evidence_id in unique_ids if evidence_id not in by_id]:
            raise StaleEvidenceError(f"Missing previously validated evidence: {', '.join(missing)}")
        evidence: list[RetrievedEvidence] = []
        for evidence_id in unique_ids:
            payload = by_id[evidence_id]
            reference = expected[evidence_id]
            actual_chunk_id = str(payload.get("chunk_id", ""))
            actual_checksum = str(payload.get("source_checksum", ""))
            mismatch = (
                actual_chunk_id != evidence_id
                or payload.get("document_id") != reference.document_id
                or payload.get("page_number") != reference.page_number
                or (
                    bool(reference.source_checksum) and actual_checksum != reference.source_checksum
                )
            )
            if mismatch:
                raise StaleEvidenceError(
                    f"Previously validated evidence provenance changed: {evidence_id}"
                )
            evidence.append(RetrievedEvidence.model_validate({**payload, "retrieval_score": 0.0}))
        return evidence

    async def list_documents(self) -> list[DocumentSummary]:
        payloads = await asyncio.wait_for(
            asyncio.to_thread(self.store.list_documents), timeout=self.timeout_seconds
        )
        return [
            DocumentSummary(
                document_id=str(item["document_id"]),
                title=str(item["document_title"]),
                region=str(item["region"]),
                source_checksum=str(item["source_checksum"]),
            )
            for item in payloads
        ]

    async def collect_summary_evidence(
        self, document_id: str, *, max_sections: int = 24
    ) -> list[RetrievedEvidence]:
        """Collect page-distributed section representatives without top-k summary bias."""
        records: list[Any] = []
        offset: Any = None
        while True:
            page, offset = await asyncio.wait_for(
                asyncio.to_thread(
                    self.store.client.scroll,
                    self.store.collection_name,
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
                ),
                timeout=self.timeout_seconds,
            )
            records.extend(page)
            if offset is None:
                break
        selected: list[RetrievedEvidence] = []
        sections: set[tuple[str, ...]] = set()
        for record in sorted(records, key=lambda item: int((item.payload or {})["page_number"])):
            payload = record.payload or {}
            section = tuple(str(value) for value in payload.get("section_path", []))
            key = section[:2] or (f"page-{payload['page_number']}",)
            if key in sections:
                continue
            sections.add(key)
            selected.append(
                RetrievedEvidence.model_validate(
                    {
                        **payload,
                        "retrieval_score": 0.0,
                    }
                )
            )
            if len(selected) >= max_sections:
                break
        return selected

    async def expand_candidate_pages(
        self, candidates: list[RetrievedEvidence], *, max_pages_per_document: int = 3
    ) -> list[RetrievedEvidence]:
        """Fetch sibling chunks from the best candidate pages without another model/API call."""
        pages: dict[str, list[int]] = {}
        for candidate in candidates:
            selected = pages.setdefault(candidate.document_id, [])
            if candidate.page_number not in selected and len(selected) < max_pages_per_document:
                selected.append(candidate.page_number)
        expanded: list[RetrievedEvidence] = []
        for document_id, page_numbers in pages.items():
            records, offset = await asyncio.wait_for(
                asyncio.to_thread(
                    self.store.client.scroll,
                    self.store.collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id", match=models.MatchValue(value=document_id)
                            ),
                            models.FieldCondition(
                                key="page_number", match=models.MatchAny(any=page_numbers)
                            ),
                        ]
                    ),
                    limit=128,
                    with_payload=True,
                    with_vectors=False,
                ),
                timeout=self.timeout_seconds,
            )
            while True:
                expanded.extend(
                    RetrievedEvidence.model_validate(
                        {**(record.payload or {}), "retrieval_score": 0.0}
                    )
                    for record in records
                )
                if offset is None:
                    break
                records, offset = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.store.client.scroll,
                        self.store.collection_name,
                        scroll_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="document_id",
                                    match=models.MatchValue(value=document_id),
                                ),
                                models.FieldCondition(
                                    key="page_number", match=models.MatchAny(any=page_numbers)
                                ),
                            ]
                        ),
                        limit=128,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    ),
                    timeout=self.timeout_seconds,
                )
        return expanded

    async def get_document_coverage(
        self, document_id: str
    ) -> tuple[DocumentCoverageReport | None, list[CoverageLimitation]]:
        report_path = self.data_root / "processed" / "dry-run-report.json"
        if not report_path.is_file():
            report_path = self.data_root / "processed" / "idempotency-report.json"
        raw = await asyncio.to_thread(report_path.read_text, encoding="utf-8")
        report = json.loads(raw)
        for document in report.get("documents", []):
            if document.get("document_id") == document_id:
                coverage = DocumentCoverageReport.model_validate(document["coverage"])
                limitations = [
                    CoverageLimitation.model_validate(item)
                    for item in document.get("limitations", [])
                ]
                return coverage, limitations
        return None, []

    async def list_skills(self) -> list[SkillMetadata]:
        return await asyncio.to_thread(self.skills.list_skills)

    async def read_skill(self, skill_name: str) -> RuntimeSkill:
        return await asyncio.to_thread(self.skills.read_skill, skill_name)

    @staticmethod
    def calculate(value: ArithmeticInput) -> float:
        if value.operation == "sum":
            return sum(value.values)
        if value.operation == "difference":
            return rounded_subtraction(value.values[0], value.values[1])
        baseline = value.values[1]
        if baseline == 0:
            raise ValueError("Cannot calculate percentage difference from zero")
        return (value.values[0] - baseline) / abs(baseline) * 100


ToolFactory = Callable[[], AgentTools]
