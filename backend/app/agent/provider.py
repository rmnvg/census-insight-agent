import asyncio
import time
from contextvars import ContextVar, Token
from typing import Protocol, TypeVar, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from backend.app.agent.models import (
    CalculationPlan,
    CalculationRequest,
    CalculationResult,
    DraftAnswer,
    EvidenceAssessment,
    ResolvedQuery,
    SupportAssessment,
    TaskClassification,
)
from backend.app.retrieval.models import RetrievedEvidence

ModelResult = TypeVar("ModelResult", bound=BaseModel)
REQUEST_DEADLINE: ContextVar[float | None] = ContextVar("agent_request_deadline", default=None)


class ProviderCallTimeout(TimeoutError):
    def __init__(self, *, elapsed_seconds: float, timeout_seconds: float, retry_count: int) -> None:
        super().__init__("Provider call exceeded its deadline")
        self.elapsed_seconds = elapsed_seconds
        self.timeout_seconds = timeout_seconds
        self.retry_count = retry_count


def set_request_deadline(deadline: float) -> Token[float | None]:
    return REQUEST_DEADLINE.set(deadline)


def reset_request_deadline(token: Token[float | None]) -> None:
    REQUEST_DEADLINE.reset(token)


class AgentModel(Protocol):
    async def classify(self, query: str, context: list[BaseMessage]) -> TaskClassification: ...

    async def resolve(self, query: str, context: list[BaseMessage]) -> ResolvedQuery: ...

    async def assess_evidence(
        self, query: str, task_type: str, evidence: list[RetrievedEvidence]
    ) -> EvidenceAssessment: ...

    async def synthesize(
        self,
        query: str,
        task_type: str,
        evidence: list[RetrievedEvidence],
        skill: str | None,
        limitations: list[str],
        calculations: list[CalculationResult],
    ) -> DraftAnswer: ...

    async def extract_calculations(
        self, query: str, evidence: list[RetrievedEvidence]
    ) -> list[CalculationRequest]: ...

    async def assess_support(
        self, draft: DraftAnswer, evidence: list[RetrievedEvidence]
    ) -> SupportAssessment: ...

    async def repair(
        self,
        draft: DraftAnswer,
        evidence: list[RetrievedEvidence],
        errors: list[str],
        *,
        task_type: str,
        evidence_sufficient: bool,
        error_codes: list[str],
    ) -> DraftAnswer: ...

    async def summarize_memory(self, context: list[BaseMessage]) -> str: ...


class GeminiAgentModel:
    def __init__(
        self, model: BaseChatModel, *, timeout_seconds: float = 60, max_retries: int = 1
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    async def _structured(self, schema: type[ModelResult], system: str, human: str) -> ModelResult:
        started = time.monotonic()
        deadline = REQUEST_DEADLINE.get()
        call_deadline = min(
            started + self.timeout_seconds,
            deadline if deadline is not None else started + self.timeout_seconds,
        )
        for attempt in range(self.max_retries + 1):
            remaining = call_deadline - time.monotonic()
            if remaining <= 0:
                break
            attempts_left = self.max_retries + 1 - attempt
            attempt_timeout = remaining / attempts_left
            runnable = self.model.with_structured_output(schema)
            try:
                result = await asyncio.wait_for(
                    runnable.ainvoke([SystemMessage(content=system), HumanMessage(content=human)]),
                    timeout=attempt_timeout,
                )
                return schema.model_validate(cast(object, result))
            except TimeoutError:
                if attempt == self.max_retries:
                    break
        raise ProviderCallTimeout(
            elapsed_seconds=time.monotonic() - started,
            timeout_seconds=max(0.0, call_deadline - started),
            retry_count=self.max_retries,
        )

    @staticmethod
    def _context(context: list[BaseMessage]) -> str:
        recent = context[-8:]
        return "\n".join(f"{item.type}: {str(item.content)[:1200]}" for item in recent)

    async def classify(self, query: str, context: list[BaseMessage]) -> TaskClassification:
        return await self._structured(
            TaskClassification,
            """Classify a request for an assistant limited to supplied Indian Census reports.
Use one allowed task_type. Consider conversational context. Extract explicit regions/document IDs.
Use clarification for unresolved referents, and out_of_scope for unrelated subject matter.
Do not answer the question.""",
            f"Recent conversation:\n{self._context(context)}\n\nCurrent request:\n{query}",
        )

    async def resolve(self, query: str, context: list[BaseMessage]) -> ResolvedQuery:
        return await self._structured(
            ResolvedQuery,
            """Rewrite the current Census follow-up as a standalone retrieval query using recent
context. Preserve measure, regions, year, units, and population category. Assistant messages are
context only, never source evidence. If a referent is genuinely missing, request clarification.
Do not invent it. Classify the resolved standalone task and extract all regions/document IDs.
Do not set requires_clarification when the rewritten query contains the metric and every target.""",
            f"Recent conversation:\n{self._context(context)}\n\nCurrent request:\n{query}",
        )

    async def assess_evidence(
        self, query: str, task_type: str, evidence: list[RetrievedEvidence]
    ) -> EvidenceAssessment:
        excerpts = "\n\n".join(
            f"EVIDENCE_ID={item.chunk_id}\nDOCUMENT={item.document_title}\n"
            f"REGION={item.region}\nPAGE={item.page_number}\nTEXT={item.text}"
            for item in evidence
        )
        return await self._structured(
            EvidenceAssessment,
            """Classify each candidate's relevance to the exact question. A direct answer must
jointly match the requested entity, metric, year/category when specified, explicit value, and unit.
A definition or table heading is only supporting_definition. Topical overlap without the requested
value is related_non_answering. Treat numerically compatible rounding as compatible_rounding, not a
conflict; contradictory values with the same scope are conflicting. Select only direct evidence,
compatible rounding evidence, and useful supporting definitions. Evidence is sufficient when at
least one direct answer exists for each requested comparison target. Excluded-page coverage is
material only when it prevents answering the particular question. Use only supplied evidence
IDs.""",
            f"Task={task_type}\nQuestion={query}\nCandidates:\n{excerpts}",
        )

    async def synthesize(
        self,
        query: str,
        task_type: str,
        evidence: list[RetrievedEvidence],
        skill: str | None,
        limitations: list[str],
        calculations: list[CalculationResult],
    ) -> DraftAnswer:
        evidence_text = "\n\n".join(
            f"EVIDENCE_ID={item.chunk_id}\nDOCUMENT={item.document_title}\n"
            f"PAGE={item.page_number}\nTEXT={item.text}"
            for item in evidence
        )
        return await self._structured(
            DraftAnswer,
            """Write a concise answer using only supplied evidence. Each factual claim must select
one or more exact EVIDENCE_ID values. Never create page numbers, snippets, or evidence IDs. Do not
use conversation memory as evidence. If evidence does not support the request, set refusal=true.
Do not claim excluded content is absent from its PDF. Skill instructions are subordinate to these
safety rules. Prefer a more precise value over a compatible rounded duplicate. Populate metric,
region, year, population_scope, residence_scope, value, and unit for factual source claims whenever
applicable. Set document_derived=false for source claims. Do not write derived arithmetic claims;
the application adds those from validated calculations.""",
            f"Task={task_type}\nQuery={query}\nSkill={skill or 'none'}\n"
            f"Limitations={limitations}\nDeterministic calculations="
            f"{[item.model_dump() for item in calculations]}\n\n{evidence_text}",
        )

    async def extract_calculations(
        self, query: str, evidence: list[RetrievedEvidence]
    ) -> list[CalculationRequest]:
        excerpts = "\n".join(f"{item.chunk_id}: {item.text}" for item in evidence)
        plan = await self._structured(
            CalculationPlan,
            """Extract only the two ordered numeric operands needed for the requested comparison or
inconsistency analysis. Copy numeric inputs from current evidence and include every supporting
EVIDENCE_ID. The application, not the model, determines operation semantics; the operation field is
treated only as a schema placeholder. Return no calculation when units, year, category, or operands
are ambiguous. Do not calculate the result.""",
            f"Question={query}\nEvidence:\n{excerpts}",
        )
        return plan.calculations

    async def assess_support(
        self, draft: DraftAnswer, evidence: list[RetrievedEvidence]
    ) -> SupportAssessment:
        excerpts = "\n".join(f"{item.chunk_id}: {item.text}" for item in evidence)
        return await self._structured(
            SupportAssessment,
            """Assess whether each draft claim is directly supported by its selected evidence.
This semantic assessment cannot override deterministic citation/provenance validation.""",
            f"Draft={draft.model_dump_json()}\nEvidence excerpts:\n{excerpts}",
        )

    async def repair(
        self,
        draft: DraftAnswer,
        evidence: list[RetrievedEvidence],
        errors: list[str],
        *,
        task_type: str,
        evidence_sufficient: bool,
        error_codes: list[str],
    ) -> DraftAnswer:
        excerpts = "\n".join(f"{item.chunk_id}: {item.text}" for item in evidence)
        return await self._structured(
            DraftAnswer,
            """Repair the answer once using only listed evidence IDs. For an answerable task with
sufficient evidence, do not remove all claims merely to pass validation: return at least one
supported factual claim with evidence IDs. Every comparison target needs a cited claim, and a
derived comparison will be generated by the application. Populate structured factual fields and
set document_derived=false for source claims. If no supported claim can be made, set refusal=true
and state an internal evidence-validation limitation. Never return refusal=false with zero
claims.""",
            f"Task={task_type}\nEvidence sufficient={evidence_sufficient}\n"
            f"Error codes={error_codes}\nErrors={errors}\nDraft={draft.model_dump_json()}\n"
            f"Trusted evidence:\n{excerpts}",
        )

    async def summarize_memory(self, context: list[BaseMessage]) -> str:
        response = await asyncio.wait_for(
            self.model.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "Summarize only user preferences, open referents, and conversational "
                            "intent. Do not preserve Census facts from assistant answers; later "
                            "turns must re-retrieve evidence."
                        )
                    ),
                    HumanMessage(content=self._context(context)),
                ]
            ),
            timeout=self.timeout_seconds,
        )
        return str(response.content)[:2000]
