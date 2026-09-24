import asyncio
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from qdrant_client import models

from backend.app.agent.calculations import rounded_subtraction
from backend.app.agent.models import EvidenceReference
from backend.app.agent.skills import RuntimeSkill, SkillMetadata, SkillRegistry
from backend.app.ingestion.models import (
    CoverageLimitation,
    DocumentCoverageReport,
    DocumentManifest,
)
from backend.app.retrieval.models import DocumentSummary, RetrievedEvidence
from backend.app.retrieval.qdrant_store import QdrantStore
from backend.app.retrieval.service import HybridRetrievalService

_WORD = re.compile(r"[a-z0-9]+")

# Statement-title qualifiers that mark a genuinely different table from the plain metric
# (e.g. "Child Sex Ratio", "Sex Ratio among Scheduled Castes/Tribes" vs plain "Sex Ratio").
_RANKING_QUALIFIER_WORDS = {"child", "children", "scheduled", "caste", "castes", "tribe", "tribes"}


def _words(value: str) -> set[str]:
    return set(_WORD.findall(value.casefold()))


def _table_title(payload: dict[str, Any]) -> str | None:
    """The one-line statement title a heading-only chunk carries, if this chunk is one.

    Census Markdown writes "### Statement 17" followed by a plain title paragraph ("Sex Ratio
    ... among Scheduled Tribes by residence"). The chunker keeps that paragraph as its own chunk,
    so every table fragment's breadcrumb says only "Statement 17".
    """
    text = str(payload.get("text", ""))
    if "|" in text:
        return None
    section = " > ".join(str(value) for value in payload.get("section_path", []))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and section and lines[0] == section:
        lines = lines[1:]
    if len(lines) != 1:
        return None
    title = lines[0].strip("*# ").strip()
    if not 12 <= len(title) <= 220 or title.endswith("."):
        return None
    return title


def _title_index(payloads: list[dict[str, Any]]) -> dict[tuple[str, ...], str]:
    candidates: dict[tuple[str, ...], set[str]] = {}
    for payload in payloads:
        title = _table_title(payload)
        if title:
            key = tuple(str(value) for value in payload.get("section_path", []))
            candidates.setdefault(key, set()).add(title)
    return {key: next(iter(titles)) for key, titles in candidates.items() if len(titles) == 1}


def with_table_title(
    item: RetrievedEvidence, titles: dict[tuple[str, ...], str]
) -> RetrievedEvidence:
    """Append a table fragment's statement title to its section path (text is untouched)."""
    if "|" not in item.text:
        return item
    title = titles.get(tuple(item.section_path))
    if not title or title in item.section_path:
        return item
    return item.model_copy(update={"section_path": [*item.section_path, title]})


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
        self._titles: dict[tuple[str, str], dict[tuple[str, ...], str]] = {}

    async def search_documents(self, value: SearchDocumentsInput) -> list[RetrievedEvidence]:
        results = await asyncio.wait_for(
            asyncio.to_thread(
                self.retrieval.search,
                value.query,
                value.document_ids,
                value.regions,
                value.top_k,
            ),
            timeout=self.timeout_seconds,
        )
        return await self.with_table_titles(results)

    async def with_table_titles(self, evidence: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
        """Attach each table fragment's statement title (cached per document version)."""
        enriched: list[RetrievedEvidence] = []
        for item in evidence:
            key = (item.document_id, item.source_checksum)
            if key not in self._titles:
                records = await self._scroll_all(
                    models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id", match=models.MatchValue(value=item.document_id)
                            )
                        ]
                    )
                )
                self._titles[key] = _title_index([record.payload or {} for record in records])
            enriched.append(with_table_title(item, self._titles[key]))
        return enriched

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
            actual_checksum = payload.get("source_checksum")
            mismatch = (
                actual_chunk_id != evidence_id
                or payload.get("document_id") != reference.document_id
                or payload.get("page_number") != reference.page_number
                or not isinstance(actual_checksum, str)
                or re.fullmatch(r"[0-9a-f]{64}", actual_checksum) is None
                or (
                    reference.source_checksum is not None
                    and actual_checksum != reference.source_checksum
                )
            )
            if mismatch:
                raise StaleEvidenceError(
                    f"Previously validated evidence provenance changed: {evidence_id}"
                )
            evidence.append(RetrievedEvidence.model_validate({**payload, "retrieval_score": 0.0}))
        return evidence

    async def list_documents(self) -> list[DocumentSummary]:
        """Read portable document identity from the generated manifests, not Qdrant.

        `search_documents`/`collect_*` already scroll or query Qdrant when they need chunk
        payloads; this only needs the small, stable per-document identity fields (id, title,
        region, checksum) that `api.py`'s `_public_documents()` already reads the same way for
        the same reason — a full collection scroll (all points, with payload) is unnecessary
        network and deserialization cost just to name the 2-3 documents in this corpus, and it
        was repeated on every `summary`/`rank_all` request.
        """

        def _read() -> list[DocumentSummary]:
            documents: list[DocumentSummary] = []
            for path in sorted((self.data_root / "manifests" / "generated").glob("*.json")):
                manifest = DocumentManifest.model_validate_json(path.read_text(encoding="utf-8"))
                documents.append(
                    DocumentSummary(
                        document_id=manifest.document_id,
                        title=manifest.title,
                        region=manifest.region,
                        source_checksum=manifest.source_checksum,
                    )
                )
            return documents

        return await asyncio.to_thread(_read)

    async def _scroll_all(self, scroll_filter: models.Filter, *, limit: int = 256) -> list[Any]:
        """Page through every matching Qdrant point, deduplicating this loop across callers."""
        records: list[Any] = []
        offset: Any = None
        while True:
            page, offset = await asyncio.wait_for(
                asyncio.to_thread(
                    self.store.client.scroll,
                    self.store.collection_name,
                    scroll_filter=scroll_filter,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                ),
                timeout=self.timeout_seconds,
            )
            records.extend(page)
            if offset is None:
                break
        return records

    async def collect_summary_evidence(
        self, document_id: str, *, max_sections: int = 24
    ) -> list[RetrievedEvidence]:
        """Collect page-distributed section representatives without top-k summary bias."""
        records = await self._scroll_all(
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            )
        )
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
        titles = _title_index([record.payload or {} for record in records])
        return [with_table_title(item, titles) for item in selected]

    async def collect_metric_table_rows(
        self, document_id: str, metric: str
    ) -> list[RetrievedEvidence]:
        """Scan every chunk of a document for the full table backing a ranking request.

        Unlike search_documents, this never truncates to a top-k semantic ranking: a "which
        district had the highest X" question needs every row of the matching table, not the
        chunks that best embed near the query text.
        """
        metric_words = _words(metric)
        records = await self._scroll_all(
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            )
        )
        disqualifying_modifiers = _RANKING_QUALIFIER_WORDS - metric_words
        titles = _title_index([record.payload or {} for record in records])
        matched: list[RetrievedEvidence] = []
        for record in records:
            payload = record.payload or {}
            path = [str(value) for value in payload.get("section_path", [])]
            section = " ".join([*path, titles.get(tuple(path), "")])
            context = f"{section} {str(payload.get('text', ''))[:400]}"
            context_words = _words(context)
            if not metric_words or not metric_words <= context_words:
                continue
            if disqualifying_modifiers & context_words:
                # e.g. a plain "sex ratio" request must not pull in the distinct "Child Sex
                # Ratio" or "Sex Ratio among Scheduled Castes/Tribes" statement tables, which
                # otherwise match on the "sex ratio" word subset alone.
                continue
            matched.append(
                with_table_title(
                    RetrievedEvidence.model_validate({**payload, "retrieval_score": 0.0}), titles
                )
            )
        return sorted(matched, key=lambda item: item.page_number)

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
            records = await self._scroll_all(
                models.Filter(
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
            )
            expanded.extend(
                RetrievedEvidence.model_validate({**(record.payload or {}), "retrieval_score": 0.0})
                for record in records
            )
        return await self.with_table_titles(expanded)

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
