import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar, Token
from typing import Literal, Protocol, TypeVar, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from backend.app.agent.models import (
    CalculationPlan,
    CalculationRequest,
    CalculationResult,
    DraftAnswer,
    EvidenceAssessment,
    ResearchPlan,
    ResolvedQuery,
    SupportAssessment,
    TaskClassification,
)
from backend.app.execution.contracts import (
    ArtifactDataset,
    ArtifactDatasetProposal,
    GeneratedProgram,
    SourceManifest,
)
from backend.app.retrieval.models import DocumentSummary, RetrievedEvidence

DocumentCatalog = Callable[[], Awaitable[list[DocumentSummary]]]
_CATALOG_LIMIT = 50

ModelResult = TypeVar("ModelResult", bound=BaseModel)
REQUEST_DEADLINE: ContextVar[float | None] = ContextVar("agent_request_deadline", default=None)


class ProviderCallTimeout(TimeoutError):
    def __init__(self, *, elapsed_seconds: float, timeout_seconds: float, retry_count: int) -> None:
        super().__init__("Provider call exceeded its deadline")
        self.elapsed_seconds = elapsed_seconds
        self.timeout_seconds = timeout_seconds
        self.retry_count = retry_count


ProviderErrorCode = Literal[
    "MODEL_SCHEMA_REJECTED",
    "MODEL_RATE_LIMITED",
    "MODEL_AUTHENTICATION_FAILED",
    "MODEL_OUTPUT_INVALID",
    "MODEL_UNAVAILABLE",
]


class ProviderOperationalError(RuntimeError):
    def __init__(
        self,
        code: ProviderErrorCode,
        *,
        retryable: bool,
        provider_exception: str,
        status_code: int | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.provider_exception = provider_exception
        self.status_code = status_code


def map_provider_error(error: Exception) -> ProviderOperationalError:
    """Map provider/structured-output failures without exposing provider text publicly."""
    name = type(error).__name__
    folded = f"{name} {error}".casefold()
    raw_status = getattr(error, "status_code", None) or getattr(error, "code", None)
    status = raw_status if isinstance(raw_status, int) else None
    if isinstance(error, ValidationError):
        return ProviderOperationalError(
            "MODEL_OUTPUT_INVALID", retryable=False, provider_exception=name, status_code=status
        )
    if "schema" in folded and (
        "invalid_argument" in folded
        or "invalid argument" in folded
        or "too many states" in folded
        or "invalidrequest" in folded
    ):
        return ProviderOperationalError(
            "MODEL_SCHEMA_REJECTED", retryable=False, provider_exception=name, status_code=status
        )
    if status == 429 or any(value in folded for value in ("resourceexhausted", "rate limit")):
        return ProviderOperationalError(
            "MODEL_RATE_LIMITED", retryable=True, provider_exception=name, status_code=status
        )
    if status in {401, 403} or any(
        value in folded
        for value in ("unauthenticated", "authentication", "permissiondenied", "permission denied")
    ):
        return ProviderOperationalError(
            "MODEL_AUTHENTICATION_FAILED",
            retryable=False,
            provider_exception=name,
            status_code=status,
        )
    return ProviderOperationalError(
        "MODEL_UNAVAILABLE", retryable=True, provider_exception=name, status_code=status
    )


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

    async def propose_artifact_dataset(
        self,
        query: str,
        task_type: str,
        evidence: list[RetrievedEvidence],
        *,
        rank_all: bool = False,
    ) -> ArtifactDatasetProposal: ...

    async def generate_artifact_code(
        self, dataset: ArtifactDataset, skill: str
    ) -> GeneratedProgram: ...

    async def repair_artifact_code(
        self,
        dataset: ArtifactDataset,
        code: str,
        skill: str,
        error_code: str,
        stderr: str,
    ) -> GeneratedProgram: ...


class GeminiAgentModel:
    def __init__(
        self,
        model: BaseChatModel,
        *,
        timeout_seconds: float = 60,
        max_retries: int = 1,
        catalog: DocumentCatalog | None = None,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.catalog = catalog

    async def _catalog_note(self) -> str:
        """Name the supplied documents so scope and region names come from the library itself.

        Without this, the classifier only knew the corpus as "Indian Census reports" and relied
        on world knowledge to recognise regions, so an uploaded report for an unfamiliar region
        was treated as an unresolved referent. Identity metadata only; never document text.
        """
        if self.catalog is None:
            return ""
        try:
            documents = await self.catalog()
        except Exception:
            return ""
        if not documents:
            return ""
        lines = "\n".join(
            f"- {item.document_id}: {item.title} (region: {item.region})"
            for item in documents[:_CATALOG_LIMIT]
        )
        return f"Supplied documents (use these exact region names and document IDs):\n{lines}\n\n"

    async def _structured(self, schema: type[ModelResult], system: str, human: str) -> ModelResult:
        started = time.monotonic()
        deadline = REQUEST_DEADLINE.get()
        call_deadline = min(
            started + self.timeout_seconds,
            deadline if deadline is not None else started + self.timeout_seconds,
        )
        last_error: ProviderOperationalError | None = None
        for attempt in range(self.max_retries + 1):
            remaining = call_deadline - time.monotonic()
            if remaining <= 0:
                break
            # Every attempt but the last gets two-thirds of whatever remains, not an equal
            # up-front share of the original budget: with an equal split, a legitimately slow but
            # otherwise healthy call (large evidence, code generation) got cut off on every single
            # attempt merely because retries it never needed were reserved in advance. The last
            # attempt always gets 100% of what's left, since there is no future attempt left to
            # preserve budget for — this also guarantees a retry always gets a real chance to run
            # (never zero time) even after an earlier attempt used its whole share on a hard
            # timeout, unlike simply giving attempt 0 the entire remaining budget outright would.
            is_last_attempt = attempt == self.max_retries
            attempt_timeout = remaining if is_last_attempt else remaining * 2 / 3
            try:
                runnable = self.model.with_structured_output(schema)
                result = await asyncio.wait_for(
                    runnable.ainvoke([SystemMessage(content=system), HumanMessage(content=human)]),
                    timeout=attempt_timeout,
                )
                return schema.model_validate(cast(object, result))
            except TimeoutError:
                if attempt == self.max_retries:
                    break
            except Exception as error:
                mapped = map_provider_error(error)
                if mapped.retryable and attempt < self.max_retries:
                    last_error = mapped
                    continue
                raise mapped from error
        if last_error is not None:
            raise last_error
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
Any region or document in the supplied documents list is in scope, including user-uploaded ones;
use its exact region name.
For artifact tasks, populate artifact_requirement with presentation type and the independent source
data need: metric, year, regions, population scope, residence scope, and comparison flag. A request
to create a chart requires numeric source data; it does not require a chart in the source PDF.
A request asking which unnamed entity (e.g. district) has the highest/lowest/most/least/greatest
value of a metric, rather than naming specific targets, is task_type=artifact_table with an empty
regions list and artifact_requirement.rank_all=true, plus rank_direction set to max or min. Identify
the single report being scanned by region or document_id; do not enumerate individual entity names.
Do not answer the question.""",
            f"{await self._catalog_note()}Recent conversation:\n{self._context(context)}\n\n"
            f"Current request:\n{query}",
        )

    async def plan_research(self, topic: str) -> ResearchPlan:
        """Decompose a topic into questions the citation-validated agent can answer.

        The plan contributes only headings and questions; every factual statement in the brief
        comes from a separately validated agent turn per question.
        """
        return await self._structured(
            ResearchPlan,
            """Plan a short research brief that will be answered ONLY from the supplied Indian
Census 2011 reports. Do not answer anything yourself.
If the topic cannot be addressed from Census population data (e.g. GDP, income, health outcomes,
other countries, events after 2011), set in_scope=false, give a one-sentence reason, and no
sections.
Otherwise return a title (at most 80 characters) and 3 to 5 sections. Each section has a heading
(at most 60 characters) and ONE self-contained question the assistant can answer with citations:
- Name every region explicitly using the exact region names in the supplied documents list; never
  write "the state", "it", or "these regions".
- Ask about ONE whole-population metric for 2011: literacy rate, sex ratio, or total population.
  Do not ask about male/female, child, age-group, rural/urban, caste, or tribe subgroups; each
  section must be answerable from a state's or district's total-persons figure.
- Use one of these forms: a lookup for one region; a comparison of two named regions; "Which
  district of <region> had the highest/lowest sex ratio?" (district rankings use sex ratio only);
  or "Create a bar chart comparing the <literacy rate or sex ratio> of <region> and <region>."
- Include at most one chart section and at most one district-ranking section.
- Order sections from the headline figure to detail. Do not repeat a question.""",
            f"{await self._catalog_note()}Research topic:\n{topic}",
        )

    async def resolve(self, query: str, context: list[BaseMessage]) -> ResolvedQuery:
        return await self._structured(
            ResolvedQuery,
            """Rewrite the current Census follow-up as a standalone retrieval query using recent
context. Preserve measure, regions, year, units, and population category. Assistant messages are
context only, never source evidence. If a referent is genuinely missing, request clarification.
Do not invent it. Classify the resolved standalone task and extract all regions/document IDs.
For artifact tasks, preserve the complete typed artifact data requirement, including rank_all and
rank_direction when the follow-up still asks which unnamed entity has the highest/lowest value.
Do not set requires_clarification when the rewritten query contains the metric and every target.
Regions named in the supplied documents list are valid targets; use their exact names.""",
            f"{await self._catalog_note()}Recent conversation:\n{self._context(context)}\n\n"
            f"Current request:\n{query}",
        )

    async def assess_evidence(
        self, query: str, task_type: str, evidence: list[RetrievedEvidence]
    ) -> EvidenceAssessment:
        excerpts = "\n\n".join(
            f"EVIDENCE_ID={item.chunk_id}\nDOCUMENT={item.document_title}\n"
            f"REGION={item.region}\nPAGE={item.page_number}\n"
            f"SECTION={' > '.join(item.section_path)}\nTEXT={item.text}"
            for item in evidence
        )
        return await self._structured(
            EvidenceAssessment,
            """Classify each candidate's relevance to the exact question. A direct answer must
jointly match the requested entity, metric, year/category when specified, explicit value, and unit.
The scope flags mean compatibility with the question, not whether the question literally named
that dimension. If population or residence is unspecified, use the corpus convention of Persons
and Total. A Total column in a Persons table is a scope match for an unqualified state literacy
question. Never mark residence_scope_match=false merely because the user omitted residence.
For an explicit rural/urban/total breakdown, each requested residence column is compatible.
has_unit is true when the unit is stated anywhere in the candidate, including its table title or
column heading (e.g. "Sex Ratio (number of females per 1000 males)", "%", "per cent"); population
counts are in persons.
SECTION names the table a candidate comes from. A table for a population subgroup (Scheduled
Castes, Scheduled Tribes, children aged 0-6) does not match a question about the whole population:
set population_scope_match=false for it unless the question asks about that subgroup.
A definition or table heading is only supporting_definition. Topical overlap without the requested
value is related_non_answering. Treat numerically compatible rounding as compatible_rounding, not a
conflict; contradictory values with the same scope are conflicting. Select only direct evidence,
compatible rounding evidence, and useful supporting definitions. Evidence is sufficient when at
least one direct answer exists for each requested comparison target. For comparison artifacts,
verify entity, metric, year, population category, residence scope, explicit value, and compatible
units for every target. For an artifact_chart or artifact_table task specifically, prefer a
tabular pipe-delimited row over narrative prose stating the same fact: only a table row can later
be bound to an exact cell, so a narrative sentence is at most compatible_rounding even when its
value matches exactly, whenever an equivalent tabular row is also present among the candidates.
Excluded-page coverage is
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
            f"PAGE={item.page_number}\nSECTION={' > '.join(item.section_path)}\nTEXT={item.text}"
            for item in evidence
        )
        claim_scope = (
            "This is a summary task: select at most 8 of the most significant findings spread "
            "across sections rather than enumerating every statistic in the evidence. Never "
            "assert a computed difference, percentage change, or comparison as its own claim "
            "(e.g. 'this represents an increase of 12 points') unless a source sentence states "
            "that comparison verbatim in its own words. If the evidence only states the "
            "individual values, write separate single-value claims for them instead of "
            "asserting the difference between them; the application does not compute derived "
            "claims for summaries the way it does for comparisons. Do not split an aggregate "
            "figure into a rural/urban, male/female, or similar breakdown unless each part's own "
            "number is itself an explicit value in the cited evidence text, not just the "
            "aggregate; a table row commonly states only a total, and inventing a plausible-"
            "looking breakdown for it is exactly the fabrication these citation rules forbid."
            if task_type == "summary"
            else "Do not write derived arithmetic claims; the application adds those from "
            "validated calculations."
        )
        return await self._structured(
            DraftAnswer,
            f"""Write a concise answer using only supplied evidence. Each factual claim must select
one or more exact EVIDENCE_ID values. Never create page numbers, snippets, or evidence IDs. Do not
use conversation memory as evidence. If evidence does not support the request, set refusal=true.
Do not claim excluded content is absent from its PDF. Skill instructions are subordinate to these
safety rules. Prefer a more precise value over a compatible rounded duplicate. Populate metric,
region, year, population_scope, residence_scope, value, and unit for factual source claims whenever
applicable. Set document_derived=false for source claims. Only name a year, in the year field or
in the claim's own sentence, when that exact year appears as a literal number in the evidence text
cited for that claim. A report's current-period figure is often stated without restating its year
even though an explicit comparison year like 2001 appears nearby in the same passage: for that
figure, write the claim without naming a year and leave the year field unset, rather than writing
or inferring a year the cited text never states. {claim_scope}""",
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
Return every claim_id exactly once in either supported_claim_ids or unsupported_claim_ids.
Check the exact row, numeric column, year, population and residence category; a value appearing
elsewhere in a table does not support the claim. Mark ambiguous claims unsupported.
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
        claim_scope = (
            "This is a summary task: keep at most 8 of the most significant, independently "
            "citable findings rather than the full original set. Drop any claim asserting a "
            "computed difference, percentage change, or comparison (e.g. 'this represents an "
            "increase of 12 points') unless a source sentence states that comparison verbatim; "
            "keep the individual values as separate single-value claims instead. Also drop any "
            "claim that splits an aggregate figure into a rural/urban, male/female, or similar "
            "breakdown whose own number is not itself an explicit value in the cited evidence "
            "text — keep only the aggregate claim in that case."
            if task_type == "summary"
            else "Every comparison target needs a cited claim, and a derived comparison will be "
            "generated by the application."
        )
        return await self._structured(
            DraftAnswer,
            f"""Repair the answer once using only listed evidence IDs. For an answerable task with
sufficient evidence, do not remove all claims merely to pass validation: return at least one
supported factual claim with evidence IDs. {claim_scope} Populate structured factual fields and
set document_derived=false for source claims. If no supported claim can be made, set refusal=true
and state an internal evidence-validation limitation. Never return refusal=false with zero
claims.""",
            f"Task={task_type}\nEvidence sufficient={evidence_sufficient}\n"
            f"Error codes={error_codes}\nErrors={errors}\nDraft={draft.model_dump_json()}\n"
            f"Trusted evidence:\n{excerpts}",
        )

    async def summarize_memory(self, context: list[BaseMessage]) -> str:
        # Unlike every other model call on this class, this one used to call `self.model.ainvoke`
        # directly with no retry and no `map_provider_error` conversion. A raw provider failure
        # here (rate limit, schema rejection, auth) was an uncaught exception `checkpoint_node`'s
        # `except ProviderCallTimeout`/`except ProviderOperationalError` clauses could not catch
        # (neither type), since only real instances of those types are handled — it would escape
        # as a bare 500 instead of the sanitized typed-error contract every other operational
        # failure in this system uses. A plain `asyncio.TimeoutError` here was the one exception:
        # `service._chat_turn` catches `TimeoutError` too, just without the more specific
        # `MODEL_TIMEOUT` code or a `prepare_artifact`-style per-call diagnostic. Routing this
        # through `_structured`-equivalent error handling (without its retry loop, since a
        # conversation-memory summary is a best-effort convenience, not required for the turn to
        # answer) closes that gap the same way `prepare_artifact` already closes it for its own
        # calls.
        started = time.monotonic()
        deadline = REQUEST_DEADLINE.get()
        timeout = min(
            self.timeout_seconds,
            deadline - started if deadline is not None else self.timeout_seconds,
        )
        try:
            response = await asyncio.wait_for(
                self.model.ainvoke(
                    [
                        SystemMessage(
                            content=(
                                "Summarize only user preferences, open referents, and "
                                "conversational intent. Do not preserve Census facts from "
                                "assistant answers; later turns must re-retrieve evidence."
                            )
                        ),
                        HumanMessage(content=self._context(context)),
                    ]
                ),
                timeout=max(0.0, timeout),
            )
        except TimeoutError as error:
            raise ProviderCallTimeout(
                elapsed_seconds=time.monotonic() - started,
                timeout_seconds=timeout,
                retry_count=0,
            ) from error
        except Exception as error:
            raise map_provider_error(error) from error
        return str(response.content)[:2000]

    async def propose_artifact_dataset(
        self,
        query: str,
        task_type: str,
        evidence: list[RetrievedEvidence],
        *,
        rank_all: bool = False,
    ) -> ArtifactDatasetProposal:
        excerpts = "\n\n".join(
            f"EVIDENCE_ID={item.chunk_id}\nREGION={item.region}\n"
            f"SECTION={' > '.join(item.section_path)}\nTEXT={item.text}"
            for item in evidence
        )
        row_instruction = (
            "Propose one row for every distinct individual entity (e.g. every district) found in "
            "the evidence table, not just the extreme value. Exclude the state/UT aggregate total "
            "row; only individual entities are eligible."
            if rank_all
            else "Propose only presentation preferences and the minimal rows needed for this "
            "artifact."
        )
        return await self._structured(
            ArtifactDatasetProposal,
            f"""{row_instruction}
Do not choose the authoritative chart/table artifact type. Copy each numeric value from
the correct trusted evidence table cell. For every row identify its evidence ID, label, optional
series, unit, year, population scope, and residence scope. Do not provide provenance, checksums,
paths, citations, manifests, internal row IDs, or computed values. Do not infer or invent
values. The label must be the exact entity name from the table, without category suffixes.
For a total/rural/urban breakdown, emit one row per entity and requested residence category;
set series and residence_scope to that category. Populate year and population_scope from the
table header or section heading. Never omit a requested entity/category combination.""",
            f"Task={task_type}\nRequest={query}\nEvidence:\n{excerpts}",
        )

    async def generate_artifact_code(
        self, dataset: ArtifactDataset, skill: str
    ) -> GeneratedProgram:
        outputs = (
            "output/chart.png, output/plotted-data.csv, output/source-manifest.json"
            if dataset.task_type == "artifact_chart"
            else "output/table.csv, output/table.md, output/source-manifest.json"
        )
        input_sample = {
            "dataset": dataset.model_dump(mode="json"),
            "source_manifest": SourceManifest(
                dataset_title=dataset.title,
                source_records=dataset.source_records,
                computed_values=dataset.computed_values,
            ).model_dump(mode="json"),
        }
        return await self._structured(
            GeneratedProgram,
            """Write task-specific Python that reads only Path('input.json') and writes only the
declared relative output paths. Use only json, math, statistics, decimal, pathlib, pandas, numpy,
matplotlib, and seaborn. Do not use open, eval, exec, compile, networking, processes, threads,
dynamic imports, absolute paths, or parent traversal. Never reference any double-underscore name
(no `__name__`, `__file__`, `__main__`, `__class__`): write plain top-level statements, not an
`if __name__ == "__main__":` block or a `main()` guard. The JSON root is an envelope with `dataset`
and `source_manifest` keys: read rows and display metadata from `payload["dataset"]`, and write
`payload["source_manifest"]` unchanged to the declared manifest output. The output directory already
exists; do not create directories. Use literal relative output paths or variables assigned directly
from those literal Path values. Every `to_csv`, `savefig`, and `write_text` call must take that
same literal declared path directly as its argument, not a variable built through renaming,
joining, or reformatting it. Never build an output path with the `/` operator (e.g.
`Path("output") / "table.csv"`); write it as one inline literal instead, either the bare string
`"output/table.csv"` or `Path("output/table.csv")`. Always call `to_csv` with that literal output
path as its own first argument, e.g. `df.to_csv("output/table.csv", index=False)`; never call
`to_csv()` with no path argument and write its returned string separately. Prefer
`Path("input.json").read_text()` and
`Path("output/...").write_text()` for JSON I/O. The same literal-path discipline required for
output paths above applies to `Path("input.json")` too: call `.read_text()`/`.read_bytes()` on
that literal directly, or on a variable assigned from it exactly once and never reassigned to
anything else afterward — reusing a generic variable name (e.g. `path`) for both the input file and
an output file makes it impossible to verify which literal path a later `.read_text()` call
actually reads. Preserve dataset rows exactly in CSV, including
column names: do not call DataFrame/Series `.rename()`; if friendlier column headers are wanted
for the Markdown table, build that string directly rather than mutating a DataFrame's columns.
Charts need labeled axes, units, deterministic style,
tight_layout, at least 120 DPI, and a zero numeric baseline unless explicitly justified. Do not use
a line chart for unordered categories. Return only the complete program in the code field.""",
            f"Applicable skill:\n{skill}\n\nDeclared outputs: {outputs}\n"
            f"Exact input.json envelope and sample:\n{json.dumps(input_sample, indent=2)}",
        )

    async def repair_artifact_code(
        self,
        dataset: ArtifactDataset,
        code: str,
        skill: str,
        error_code: str,
        stderr: str,
    ) -> GeneratedProgram:
        input_sample = {
            "dataset": dataset.model_dump(mode="json"),
            "source_manifest": SourceManifest(
                dataset_title=dataset.title,
                source_records=dataset.source_records,
                computed_values=dataset.computed_values,
            ).model_dump(mode="json"),
        }
        return await self._structured(
            GeneratedProgram,
            """Repair this artifact program once. Preserve the same restricted Python policy,
input dataset, declared outputs, exact CSV values, and exact source manifest. The JSON root has
`dataset` and `source_manifest` keys. Read data from `payload["dataset"]` and write
`payload["source_manifest"]` unchanged. The output directory already exists; do not create it.
Address only the reported runtime or artifact-validation failure. Return only the complete repaired
program.""",
            f"Skill:\n{skill}\nError={error_code}\nBounded stderr:\n{stderr[:8000]}\n"
            f"Exact input.json envelope:\n{json.dumps(input_sample, indent=2)}\nProgram:\n{code}",
        )
