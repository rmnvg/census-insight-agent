import asyncio
import hashlib
import re
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any, Literal, cast

from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from backend.app.agent.calculations import (
    add_deterministic_derived_claims,
    enrich_source_claims,
    validated_calculation,
)
from backend.app.agent.checkpoint import checkpoint_node
from backend.app.agent.citations import render_citation, validate_and_materialize_citations
from backend.app.agent.errors import AgentOperationalError
from backend.app.agent.evidence import (
    EvidenceBudgetInsufficient,
    bound_assessment_evidence,
    pack_targeted_assessment_evidence,
)
from backend.app.agent.memory import (
    build_validated_claim_turn,
    evidence_references,
    is_source_support_query,
    select_source_support_turn,
    source_support_draft,
    source_support_query,
)
from backend.app.agent.models import (
    AgentPlan,
    AgentResponse,
    AgentState,
    ArtifactDataRequirement,
    ArtifactResult,
    CalculationResult,
    DraftAnswer,
    EvidenceAssessment,
    SupportAssessment,
    TaskClassification,
    ToolCallRecord,
    TraceEvent,
)
from backend.app.agent.provider import AgentModel, ProviderCallTimeout, ProviderOperationalError
from backend.app.agent.scopes import canonicalize_artifact_requirement
from backend.app.agent.tools import AgentTools, ArithmeticInput, SearchDocumentsInput
from backend.app.execution.client import (
    ArtifactStore,
    ExecutionQueueClient,
    ExecutorUnavailableError,
    new_request,
)
from backend.app.execution.contracts import ExpectedArtifact, SourceManifest, SourceRecord
from backend.app.execution.hydration import (
    ProposalHydrationError,
    TrustedEvidenceProvenanceError,
    count_table_entity_rows,
    hydrate_artifact_dataset,
    resolved_chart_kind,
    validate_trusted_artifact_evidence,
)
from backend.app.execution.lineage import (
    DatasetValidationError,
    validate_artifact_lineage,
    validate_dataset,
)
from backend.app.execution.presentation import artifact_response
from backend.app.retrieval.models import DocumentSummary, RetrievedEvidence
from executor.policy import validate_code

_ARTIFACT_BOILERPLATE = re.compile(
    r"\b(?:create|make|generate|show|plot|a|an|the|bar|line|chart|table|comparing?|for|and)\b",
    re.IGNORECASE,
)


def build_artifact_requirement(
    classification: TaskClassification, resolved_query: str, task_type: str
) -> ArtifactDataRequirement | None:
    if task_type not in {"artifact_chart", "artifact_table"}:
        return None
    if classification.artifact_requirement is not None:
        rank_all = classification.artifact_requirement.rank_all
        target_regions = (
            []
            if rank_all
            else classification.artifact_requirement.regions or classification.regions
        )
        requirement = classification.artifact_requirement.model_copy(
            update={
                "artifact_type": "chart" if task_type == "artifact_chart" else "table",
                "regions": target_regions,
                "comparison": bool(
                    not rank_all
                    and (classification.artifact_requirement.comparison or len(target_regions) > 1)
                ),
            }
        )
        return canonicalize_artifact_requirement(requirement, resolved_query)
    metric = resolved_query
    for region in classification.regions:
        metric = re.sub(re.escape(region), " ", metric, flags=re.IGNORECASE)
    metric = re.sub(r"\b(?:18|19|20|21)\d{2}\b", " ", metric)
    metric = re.sub(r"\b(?:persons?|total|rural|urban)\b", " ", metric, flags=re.IGNORECASE)
    metric = _ARTIFACT_BOILERPLATE.sub(" ", metric)
    metric = " ".join(metric.split()) or "requested metric"
    year_match = re.search(r"\b((?:18|19|20|21)\d{2})\b", resolved_query)
    folded = resolved_query.casefold()
    return canonicalize_artifact_requirement(
        ArtifactDataRequirement(
            artifact_type="chart" if task_type == "artifact_chart" else "table",
            metric=metric,
            year=int(year_match.group(1)) if year_match else None,
            regions=classification.regions,
            population_scope="persons" if "person" in folded else None,
            residence_scope=next(
                (scope for scope in ("total", "rural", "urban") if scope in folded), None
            ),
            comparison=len(classification.regions) > 1,
        ),
        resolved_query,
    )


def format_artifact_target_query(requirement: ArtifactDataRequirement, target: str) -> str:
    """Build a normalized data query; presentation words never enter retrieval."""
    parts: list[str] = []
    if requirement.year is not None:
        parts.append(str(requirement.year))
    if requirement.residence_scope:
        parts.append(requirement.residence_scope)
    if requirement.population_scope:
        parts.append(requirement.population_scope)
    parts.extend((requirement.metric, target))
    return " ".join(" ".join(parts).split())


class AgentGraph:
    def __init__(
        self,
        model: AgentModel,
        tools: AgentTools | Any,
        *,
        max_tool_calls: int = 6,
        memory_turn_threshold: int = 12,
        assessment_max_characters: int = 24_000,
        assessment_max_chunks: int = 24,
        execution_queue: ExecutionQueueClient | None = None,
        artifact_store: ArtifactStore | None = None,
        execution_timeout_seconds: float = 30,
        artifact_max_files: int = 8,
        artifact_max_total_bytes: int = 20 * 1024 * 1024,
    ) -> None:
        self.model = model
        self.tools = tools
        self.max_tool_calls = max_tool_calls
        self.memory_turn_threshold = memory_turn_threshold
        self.assessment_max_characters = assessment_max_characters
        self.assessment_max_chunks = assessment_max_chunks
        self.execution_queue = execution_queue
        self.artifact_store = artifact_store
        self.execution_timeout_seconds = execution_timeout_seconds
        self.artifact_max_files = artifact_max_files
        self.artifact_max_total_bytes = artifact_max_total_bytes

    def build(self, checkpointer: Any) -> object:
        graph = StateGraph(AgentState)
        graph.add_node("load_memory", checkpoint_node(self.load_memory))
        graph.add_node("classify_task", checkpoint_node(self.classify_task))
        graph.add_node("resolve_query", checkpoint_node(self.resolve_query))
        graph.add_node("plan", checkpoint_node(self.plan))
        graph.add_node("load_skill", checkpoint_node(self.load_skill))
        graph.add_node("call_tools", checkpoint_node(self.call_tools))
        graph.add_node("assess_evidence", checkpoint_node(self.assess_evidence))
        graph.add_node("synthesize", checkpoint_node(self.synthesize))
        graph.add_node("prepare_artifact", checkpoint_node(self.prepare_artifact))
        graph.add_node("generate_artifact_code", checkpoint_node(self.generate_artifact_code))
        graph.add_node("execute_artifact", checkpoint_node(self.execute_artifact))
        graph.add_node("inspect_artifact", checkpoint_node(self.inspect_artifact))
        graph.add_node("repair_artifact", checkpoint_node(self.repair_artifact))
        graph.add_node("validate_citations", checkpoint_node(self.validate_citations))
        graph.add_node("repair", checkpoint_node(self.repair))
        graph.add_node("graceful_response", checkpoint_node(self.graceful_response))
        graph.add_node("persist_result", checkpoint_node(self.persist_result))
        graph.add_edge(START, "load_memory")
        graph.add_edge("load_memory", "classify_task")
        graph.add_conditional_edges(
            "classify_task",
            self._classification_route,
            {"continue": "resolve_query", "stop": "graceful_response"},
        )
        graph.add_conditional_edges(
            "resolve_query",
            self._resolution_route,
            {"continue": "plan", "stop": "graceful_response"},
        )
        graph.add_conditional_edges(
            "plan", self._plan_route, {"continue": "load_skill", "stop": "graceful_response"}
        )
        graph.add_edge("load_skill", "call_tools")
        graph.add_edge("call_tools", "assess_evidence")
        graph.add_conditional_edges(
            "assess_evidence",
            self._evidence_route,
            {
                "continue": "synthesize",
                "artifact": "prepare_artifact",
                "stop": "graceful_response",
            },
        )
        graph.add_conditional_edges(
            "prepare_artifact",
            self._artifact_prepare_route,
            {"continue": "generate_artifact_code", "stop": "graceful_response"},
        )
        graph.add_edge("generate_artifact_code", "execute_artifact")
        graph.add_edge("execute_artifact", "inspect_artifact")
        graph.add_conditional_edges(
            "inspect_artifact",
            self._artifact_result_route,
            {"done": "persist_result", "repair": "repair_artifact", "stop": "graceful_response"},
        )
        graph.add_edge("repair_artifact", "execute_artifact")
        graph.add_edge("synthesize", "validate_citations")
        graph.add_conditional_edges(
            "validate_citations",
            self._validation_route,
            {"done": "persist_result", "repair": "repair", "stop": "graceful_response"},
        )
        graph.add_edge("repair", "validate_citations")
        graph.add_edge("graceful_response", "persist_result")
        graph.add_edge("persist_result", END)
        return graph.compile(checkpointer=checkpointer)

    @staticmethod
    def _event(event: str, node: str, **details: object) -> TraceEvent:
        return TraceEvent(event=event, node=node, details=details)

    async def load_memory(self, state: AgentState) -> dict[str, object]:
        messages = state.get("messages", [])
        summary = state.get("conversation_summary", "")
        removals: list[RemoveMessage] = []
        if len(messages) > self.memory_turn_threshold * 2:
            summary = await self.model.summarize_memory(list(messages[:-8]))
            removals = [RemoveMessage(id=item.id) for item in messages[:-8] if item.id]
        updates: dict[str, object] = {
            "conversation_summary": summary,
            "trace_events": [
                *state.get("trace_events", []),
                self._event(
                    "node_end",
                    "load_memory",
                    message_count=len(messages),
                    pruned_message_count=len(removals),
                ),
            ],
        }
        if removals:
            updates["messages"] = removals
        return updates

    async def classify_task(self, state: AgentState) -> dict[str, object]:
        started = time.monotonic()
        context = cast(list[BaseMessage], list(state.get("messages", [])[:-1]))
        deterministic = is_source_support_query(state["user_query"])
        classification = (
            TaskClassification(
                task_type="source_support",
                referent="previous validated claims",
                reason="Explicit source or citation follow-up",
            )
            if deterministic
            else await self.model.classify(state["user_query"], context)
        )
        return {
            "classification": classification,
            "task_type": classification.task_type,
            "trace_events": [
                *state.get("trace_events", []),
                *(
                    []
                    if deterministic
                    else [
                        self._event("model_invocation", "classify_task", operation="classification")
                    ]
                ),
                TraceEvent(
                    event="task_classification",
                    node="classify_task",
                    details={
                        "task_type": classification.task_type,
                        "deterministic_safeguard": deterministic,
                        "conversation_message_count": len(context),
                        "validated_history_turn_count": len(
                            state.get("validated_claim_history", [])
                        ),
                        "artifact_data_requirement": (
                            classification.artifact_requirement.model_dump(mode="json")
                            if classification.artifact_requirement
                            else None
                        ),
                    },
                    latency_ms=(time.monotonic() - started) * 1000,
                ),
            ],
        }

    async def resolve_query(self, state: AgentState) -> dict[str, object]:
        started = time.monotonic()
        context = cast(list[BaseMessage], list(state.get("messages", [])[:-1]))
        if state["task_type"] == "source_support":
            turn, clarification = select_source_support_turn(
                state["user_query"], state.get("validated_claim_history", [])
            )
            if turn is None:
                classification = state["classification"].model_copy(
                    update={
                        "task_type": "clarification",
                        "clarification_question": clarification,
                    }
                )
                return {
                    "task_type": "clarification",
                    "classification": classification,
                    "resolved_query": state["user_query"],
                    "source_support_claims": [],
                    "trace_events": [
                        *state.get("trace_events", []),
                        self._event(
                            "resolved_query",
                            "resolve_query",
                            requires_clarification=True,
                            query=state["user_query"],
                            resolution_source="validated_claim_history",
                            latency_ms=(time.monotonic() - started) * 1000,
                        ),
                    ],
                }
            resolved_query = source_support_query(turn)
            return {
                "resolved_query": resolved_query,
                "source_support_claims": turn.claims,
                "trace_events": [
                    *state.get("trace_events", []),
                    self._event(
                        "resolved_query",
                        "resolve_query",
                        requires_clarification=False,
                        query=resolved_query,
                        resolution_source="validated_claim_history",
                        referenced_run_id=turn.run_id,
                        referenced_claim_ids=[claim.claim_id for claim in turn.claims],
                        latency_ms=(time.monotonic() - started) * 1000,
                    ),
                ],
            }
        resolved = await self.model.resolve(state["user_query"], context)
        classification = state["classification"]
        target_names = classification.regions or classification.document_ids
        missing_targets = [
            target for target in target_names if target.casefold() not in resolved.query.casefold()
        ]
        resolved_query = resolved.query
        target_guard_applied = bool(
            state["task_type"] == "comparison" and target_names and missing_targets
        )
        if target_guard_applied:
            resolved_query = (
                f"{resolved.query.rstrip()} Compare the same metric, year, population category, "
                f"and units across these targets: {', '.join(target_names)}."
            )
        resolved_has_targets = bool(target_names) and all(
            target.casefold() in resolved_query.casefold() for target in target_names
        )
        requires_clarification = resolved.requires_clarification and not resolved_has_targets
        resolved_requirement = resolved.artifact_requirement
        classified_requirement = classification.artifact_requirement
        if resolved_requirement is not None and classified_requirement is not None:
            resolved_requirement = resolved_requirement.model_copy(
                update={
                    "regions": resolved_requirement.regions or classified_requirement.regions,
                    "year": resolved_requirement.year or classified_requirement.year,
                    "population_scope": resolved_requirement.population_scope
                    or classified_requirement.population_scope,
                    "residence_scope": resolved_requirement.residence_scope
                    or classified_requirement.residence_scope,
                    "comparison": (
                        resolved_requirement.comparison or classified_requirement.comparison
                    ),
                }
            )
        updates: dict[str, object] = {
            "resolved_query": resolved_query,
            "artifact_requirement": resolved_requirement or classified_requirement,
            "trace_events": [
                *state.get("trace_events", []),
                self._event("model_invocation", "resolve_query", operation="query_resolution"),
                self._event(
                    "resolved_query",
                    "resolve_query",
                    requires_clarification=requires_clarification,
                    query=resolved_query,
                    target_guard_applied=target_guard_applied,
                    missing_targets=missing_targets,
                    latency_ms=(time.monotonic() - started) * 1000,
                ),
            ],
        }
        if requires_clarification:
            classification = state["classification"].model_copy(
                update={
                    "task_type": "clarification",
                    "clarification_question": resolved.clarification_question,
                }
            )
            updates.update(task_type="clarification", classification=classification)
        elif state["task_type"] == "clarification" and resolved.task_type not in {
            None,
            "clarification",
            "out_of_scope",
        }:
            classification = classification.model_copy(
                update={
                    "task_type": resolved.task_type,
                    "regions": resolved.regions,
                    "document_ids": resolved.document_ids,
                    "clarification_question": None,
                }
            )
            updates.update(task_type=resolved.task_type, classification=classification)
        return updates

    async def plan(self, state: AgentState) -> dict[str, object]:
        classification = state["classification"]
        task_type = state["task_type"]
        if task_type == "summary" and not (
            classification.regions or classification.document_ids or classification.referent
        ):
            classification = classification.model_copy(
                update={
                    "task_type": "clarification",
                    "clarification_question": "Which Census report should I summarize?",
                }
            )
            return {"task_type": "clarification", "classification": classification}
        requirement_hint = state.get("artifact_requirement") or classification.artifact_requirement
        if (
            task_type == "artifact_table"
            and requirement_hint is not None
            and requirement_hint.rank_all
            and not classification.document_ids
            and len(classification.regions) != 1
        ):
            classification = classification.model_copy(
                update={
                    "task_type": "clarification",
                    "clarification_question": "Which Census report's districts should I rank?",
                }
            )
            return {"task_type": "clarification", "classification": classification}
        artifact = task_type in {"artifact_chart", "artifact_table"}
        steps = ["retrieve citation-safe evidence", "validate evidence"]
        if artifact:
            steps.extend(
                [
                    "construct and validate cited dataset",
                    "generate restricted Python",
                    "execute in isolated worker",
                    "validate and persist artifact",
                ]
            )
        else:
            steps.extend(["synthesize cited claims", "validate claim citations"])
        requirement = build_artifact_requirement(
            classification.model_copy(
                update={
                    "artifact_requirement": state.get("artifact_requirement")
                    or classification.artifact_requirement
                }
            ),
            state["resolved_query"],
            task_type,
        )
        return {
            "plan": AgentPlan(steps=steps, retrieval_top_k=10, capability_pending=artifact),
            "artifact_requirement": requirement,
            "trace_events": [
                *state.get("trace_events", []),
                *(
                    [
                        self._event(
                            "artifact_data_requirement",
                            "plan",
                            requirement=requirement.model_dump(mode="json"),
                        )
                    ]
                    if requirement is not None
                    else []
                ),
                self._event("node_end", "plan", step_count=len(steps)),
            ],
        }

    async def load_skill(self, state: AgentState) -> dict[str, object]:
        if state["task_type"] not in {
            "summary",
            "inconsistency_analysis",
            "artifact_chart",
            "artifact_table",
        }:
            return {"selected_skill": None, "skill_instructions": None}
        started = time.monotonic()
        metadata = await self.tools.list_skills()
        listing = ToolCallRecord(
            tool_name="list_skills",
            arguments={},
            status="ok",
            result_count=len(metadata),
            latency_ms=(time.monotonic() - started) * 1000,
        )
        selected = next((item for item in metadata if state["task_type"] in item.task_types), None)
        if selected is None:
            return {"selected_skill": None, "skill_instructions": None, "tool_calls": [listing]}
        started = time.monotonic()
        skill = await self.tools.read_skill(selected.name)
        reading = ToolCallRecord(
            tool_name="read_skill",
            arguments={"skill_name": selected.name},
            status="ok",
            result_count=1,
            latency_ms=(time.monotonic() - started) * 1000,
        )
        return {
            "selected_skill": skill.metadata.name,
            "skill_instructions": skill.instructions,
            "tool_calls": [*state.get("tool_calls", []), listing, reading],
            "trace_events": [
                *state.get("trace_events", []),
                self._event("selected_skill", "load_skill", skill=skill.metadata.name),
            ],
        }

    async def _recorded_tool(
        self,
        state: AgentState,
        name: str,
        arguments: dict[str, object],
        call: Callable[[], Awaitable[object]],
    ) -> tuple[object | None, ToolCallRecord]:
        if len(state.get("tool_calls", [])) >= self.max_tool_calls:
            return None, ToolCallRecord(
                tool_name=name, arguments=arguments, status="limit_exceeded"
            )
        started = time.monotonic()
        try:
            result = await call()
            count = len(result) if isinstance(result, list) else int(result is not None)
            return result, ToolCallRecord(
                tool_name=name,
                arguments=arguments,
                status="ok",
                result_count=count,
                latency_ms=(time.monotonic() - started) * 1000,
            )
        except TimeoutError:
            return None, ToolCallRecord(
                tool_name=name,
                arguments=arguments,
                status="timeout",
                error_type="TimeoutError",
                latency_ms=(time.monotonic() - started) * 1000,
            )
        except Exception as error:
            return None, ToolCallRecord(
                tool_name=name,
                arguments=arguments,
                status="error",
                error_type=type(error).__name__,
                latency_ms=(time.monotonic() - started) * 1000,
            )

    async def call_tools(self, state: AgentState) -> dict[str, object]:
        classification = state["classification"]
        evidence: list[RetrievedEvidence] = []
        calls = list(state.get("tool_calls", []))
        query = state["resolved_query"]
        retrieval_events: list[TraceEvent] = []
        if state["task_type"] == "source_support":
            references = evidence_references(state.get("source_support_claims", []))
            result, record = await self._recorded_tool(
                state,
                "get_evidence_by_ids",
                {"evidence_ids": list(references)},
                partial(self.tools.get_evidence_by_ids, list(references), references),
            )
            evidence = cast(list[RetrievedEvidence], result or [])
            return {
                "retrieved_evidence": evidence,
                "tool_calls": [*state.get("tool_calls", []), record],
                "errors": (
                    state.get("errors", [])
                    if record.status == "ok"
                    else [*state.get("errors", []), f"STALE_EVIDENCE: {record.error_type}"]
                ),
                "trace_events": [
                    *state.get("trace_events", []),
                    self._event(
                        "evidence_rehydration",
                        "call_tools",
                        requested_evidence_ids=list(references),
                        rehydrated_evidence_ids=[item.chunk_id for item in evidence],
                        embedding_calls=0,
                        status=record.status,
                        error_type=record.error_type,
                    ),
                ],
            }
        targets: list[tuple[list[str] | None, list[str] | None]] = []
        artifact_requirement = state.get("artifact_requirement")
        artifact_comparison = bool(
            state["task_type"] in {"artifact_chart", "artifact_table"}
            and artifact_requirement
            and artifact_requirement.comparison
        )
        if artifact_comparison and artifact_requirement and artifact_requirement.regions:
            targets.extend((None, [region]) for region in artifact_requirement.regions)
        elif artifact_comparison and classification.document_ids:
            targets.extend(([document], None) for document in classification.document_ids)
        elif state["task_type"] == "comparison" and classification.regions:
            targets.extend((None, [region]) for region in classification.regions)
        elif state["task_type"] == "comparison" and classification.document_ids:
            targets.extend(([document], None) for document in classification.document_ids)
        else:
            targets.append((classification.document_ids or None, classification.regions or None))
        if artifact_requirement is not None and artifact_requirement.rank_all:
            documents_result, listing = await self._recorded_tool(
                {**state, "tool_calls": calls},
                "list_documents",
                {},
                self.tools.list_documents,
            )
            calls.append(listing)
            documents = cast(list[DocumentSummary], documents_result or [])
            document_id = next(iter(classification.document_ids), None)
            if document_id is None and classification.regions:
                region = classification.regions[0]
                document_id = next(
                    (item.document_id for item in documents if item.region == region), None
                )
            if document_id is not None:
                result, record = await self._recorded_tool(
                    {**state, "tool_calls": calls},
                    "collect_metric_table_rows",
                    {"document_id": document_id, "metric": artifact_requirement.metric},
                    partial(
                        self.tools.collect_metric_table_rows,
                        document_id,
                        artifact_requirement.metric,
                    ),
                )
                calls.append(record)
                evidence.extend(cast(list[RetrievedEvidence], result or []))
        elif state["task_type"] == "summary":
            documents_result, listing = await self._recorded_tool(
                {**state, "tool_calls": calls},
                "list_documents",
                {},
                self.tools.list_documents,
            )
            calls.append(listing)
            documents = cast(list[DocumentSummary], documents_result or [])
            wanted = classification.document_ids
            if not wanted and classification.regions:
                wanted = [
                    item.document_id for item in documents if item.region in classification.regions
                ]
            if not wanted and classification.referent:
                referent = classification.referent.casefold()
                wanted = [
                    item.document_id
                    for item in documents
                    if referent in item.title.casefold() or referent in item.region.casefold()
                ]
            for document_id in wanted:
                result, record = await self._recorded_tool(
                    {**state, "tool_calls": calls},
                    "collect_summary_evidence",
                    {"document_id": document_id},
                    partial(self.tools.collect_summary_evidence, document_id),
                )
                calls.append(record)
                evidence.extend(cast(list[RetrievedEvidence], result or []))
        else:
            for document_ids, regions in targets:
                target_name: str | None = None
                if regions:
                    target_name = regions[0]
                elif document_ids:
                    target_name = document_ids[0]
                target_query = query
                if artifact_requirement is not None and target_name is not None:
                    target_query = format_artifact_target_query(artifact_requirement, target_name)
                value = SearchDocumentsInput(
                    query=target_query,
                    document_ids=document_ids,
                    regions=regions,
                    top_k=state["plan"].retrieval_top_k,
                )
                result, record = await self._recorded_tool(
                    {**state, "tool_calls": calls},
                    "search_documents",
                    value.model_dump(exclude_none=True),
                    partial(self.tools.search_documents, value),
                )
                calls.append(record)
                target_evidence = cast(list[RetrievedEvidence], result or [])
                evidence.extend(target_evidence)
                retrieval_events.append(
                    self._event(
                        "target_retrieval",
                        "call_tools",
                        target=target_name,
                        query=target_query,
                        document_ids=document_ids,
                        regions=regions,
                        candidate_count=len(target_evidence),
                        candidate_ids=[item.chunk_id for item in target_evidence],
                    )
                )
        deduplicated = {item.chunk_id: item for item in evidence}
        if deduplicated and state["task_type"] in {"lookup", "comparison"}:
            expansion, record = await self._recorded_tool(
                {**state, "tool_calls": calls},
                "expand_candidate_pages",
                {"candidate_count": len(deduplicated), "max_pages_per_document": 3},
                partial(self.tools.expand_candidate_pages, list(deduplicated.values())),
            )
            calls.append(record)
            for item in cast(list[RetrievedEvidence], expansion or []):
                deduplicated.setdefault(item.chunk_id, item)
        if state["task_type"] in {"artifact_chart", "artifact_table"}:
            try:
                validate_trusted_artifact_evidence(list(deduplicated.values()))
            except TrustedEvidenceProvenanceError as error:
                raise AgentOperationalError(
                    code="INTERNAL_PROVENANCE_INVALID",
                    node="call_tools",
                    message=(
                        "Artifact preparation failed because trusted source provenance was invalid."
                    ),
                    retryable=False,
                    elapsed_seconds=None,
                    configured_timeout_seconds=None,
                    retry_count=0,
                    diagnostics={
                        "failure_stage": "retrieval_provenance_validation",
                        "reason_code": error.code,
                        "affected_evidence_id": error.evidence_id,
                        "checksum_present": error.checksum_present,
                        "executor_submitted": False,
                    },
                ) from error
        calculations: list[CalculationResult] = []
        calculation_event: list[TraceEvent] = []
        if state["task_type"] in {"comparison", "inconsistency_analysis"} and deduplicated:
            calculation_evidence = bound_assessment_evidence(
                list(deduplicated.values()),
                max_characters=self.assessment_max_characters,
                max_chunks=self.assessment_max_chunks,
            )
            requests = await self.model.extract_calculations(query, calculation_evidence)
            calculation_event.append(
                self._event(
                    "model_invocation",
                    "call_tools",
                    operation="calculation_extraction",
                    input_chunk_count=len(calculation_evidence),
                    input_characters=sum(len(item.text) for item in calculation_evidence),
                    input_counts_by_document=dict(
                        Counter(item.document_id for item in calculation_evidence)
                    ),
                )
            )
            for request in requests:
                arguments = request.model_dump()
                normalized = validated_calculation(query, request, list(deduplicated.values()))
                if normalized is None:
                    calls.append(
                        ToolCallRecord(
                            tool_name="calculate",
                            arguments=arguments,
                            status="error",
                            error_type="InvalidCalculationOperands",
                        )
                    )
                    continue
                arithmetic_input = ArithmeticInput(
                    operation=normalized.operation, values=normalized.values
                )
                arguments["operation"] = normalized.operation
                arguments["unit"] = normalized.unit
                result, record = await self._recorded_tool(
                    {**state, "tool_calls": calls},
                    "calculate",
                    arguments,
                    partial(asyncio.to_thread, self.tools.calculate, arithmetic_input),
                )
                calls.append(record)
                if isinstance(result, int | float):
                    calculations.append(normalized.model_copy(update={"result": float(result)}))
        available_limitations: list[str] = []
        for document_id in sorted({item.document_id for item in deduplicated.values()}):
            coverage_result, record = await self._recorded_tool(
                {**state, "tool_calls": calls},
                "get_document_coverage",
                {"document_id": document_id},
                partial(self.tools.get_document_coverage, document_id),
            )
            calls.append(record)
            if coverage_result is None:
                continue
            coverage, values = cast(tuple[object, list[Any]], coverage_result)
            del coverage
            available_limitations.extend(
                item.message for item in values if item.message not in available_limitations
            )
        errors = [f"{item.tool_name}: {item.status}" for item in calls if item.status != "ok"]
        return {
            "retrieved_evidence": list(deduplicated.values()),
            "tool_calls": calls,
            "available_limitations": available_limitations,
            "calculations": calculations,
            "errors": [*state.get("errors", []), *errors],
            "trace_events": [
                *state.get("trace_events", []),
                *retrieval_events,
                *calculation_event,
                self._event(
                    "tool_result_summary",
                    "call_tools",
                    candidate_count=len(deduplicated),
                    tool_call_count=len(calls),
                ),
                self._event(
                    "retrieval_candidates",
                    "call_tools",
                    candidates=[
                        {
                            "chunk_id": item.chunk_id,
                            "document_id": item.document_id,
                            "page_number": item.page_number,
                            "citation_snippet": item.citation_snippet,
                            "retrieval_score": item.retrieval_score,
                            "coverage_status": item.coverage_status,
                        }
                        for item in deduplicated.values()
                    ],
                ),
            ],
        }

    async def assess_evidence(self, state: AgentState) -> dict[str, object]:
        evidence = state.get("retrieved_evidence", [])
        if state["task_type"] == "source_support":
            expected = set(evidence_references(state.get("source_support_claims", [])))
            actual = {item.chunk_id for item in evidence}
            sufficient = bool(expected and expected == actual)
            assessment = EvidenceAssessment(
                selected_evidence_ids=[item.chunk_id for item in evidence] if sufficient else [],
                sufficient=sufficient,
                explanation=(
                    "Previously validated evidence was rehydrated and provenance checked"
                    if sufficient
                    else "Previously validated evidence is missing or stale"
                ),
            )
            return {
                "evidence_assessment": assessment,
                "selected_evidence": evidence if sufficient else [],
                "evidence_sufficient": sufficient,
                "trace_events": [
                    *state.get("trace_events", []),
                    self._event(
                        "evidence_assessment",
                        "assess_evidence",
                        sufficient=sufficient,
                        candidate_count=len(evidence),
                        selected_evidence_ids=assessment.selected_evidence_ids,
                        assessment_method="deterministic_rehydration",
                        explanation=assessment.explanation,
                    ),
                ],
            }
        rank_requirement = state.get("artifact_requirement")
        if rank_requirement is not None and rank_requirement.rank_all:
            sufficient = bool(evidence)
            assessment = EvidenceAssessment(
                selected_evidence_ids=[item.chunk_id for item in evidence] if sufficient else [],
                sufficient=sufficient,
                explanation=(
                    "A full-table scan of the requested document returned candidate rows"
                    if sufficient
                    else "No matching table rows were found for the requested metric"
                ),
            )
            return {
                "evidence_assessment": assessment,
                "selected_evidence": evidence if sufficient else [],
                "evidence_sufficient": sufficient,
                "trace_events": [
                    *state.get("trace_events", []),
                    self._event(
                        "evidence_assessment",
                        "assess_evidence",
                        sufficient=sufficient,
                        candidate_count=len(evidence),
                        selected_evidence_ids=assessment.selected_evidence_ids,
                        assessment_method="deterministic_full_table_scan",
                        explanation=assessment.explanation,
                    ),
                ],
            }
        classification = state["classification"]
        requirement = state.get("artifact_requirement")
        required_targets = (
            requirement.regions or classification.document_ids
            if requirement is not None and requirement.comparison
            else (
                classification.regions or classification.document_ids
                if state["task_type"] == "comparison"
                else []
            )
        )
        try:
            packing = pack_targeted_assessment_evidence(
                evidence,
                required_targets=required_targets,
                max_characters=self.assessment_max_characters,
                max_chunks=self.assessment_max_chunks,
            )
        except EvidenceBudgetInsufficient as error:
            raise AgentOperationalError(
                code="EVIDENCE_BUDGET_INSUFFICIENT",
                node="assess_evidence",
                message=(
                    "Direct evidence for every requested target cannot fit within the safe "
                    "assessment budget."
                ),
                retryable=False,
                elapsed_seconds=0,
                configured_timeout_seconds=0,
                retry_count=0,
                diagnostics={
                    "candidate_count": len(evidence),
                    "required_targets": required_targets,
                    "character_budget": self.assessment_max_characters,
                    "chunk_budget": self.assessment_max_chunks,
                    "reason": str(error),
                },
            ) from error
        assessment_evidence = packing.included
        input_characters = sum(len(item.text) for item in assessment_evidence)
        counts = dict(Counter(item.document_id for item in assessment_evidence))
        counts_by_target = {
            target: sum(
                item.region.casefold() == target.casefold()
                or item.document_id.casefold() == target.casefold()
                for item in assessment_evidence
            )
            for target in required_targets
        }
        if not evidence:
            assessment = EvidenceAssessment(
                sufficient=False,
                coverage_limitation_material=bool(state.get("available_limitations")),
                explanation="No candidate evidence was retrieved",
            )
        else:
            started = time.monotonic()
            try:
                assessment = await self.model.assess_evidence(
                    state["resolved_query"], state["task_type"], assessment_evidence
                )
            except (ProviderCallTimeout, TimeoutError) as error:
                elapsed = time.monotonic() - started
                raise AgentOperationalError(
                    code="EVIDENCE_ASSESSMENT_TIMEOUT",
                    node="assess_evidence",
                    message=(
                        "Evidence assessment timed out before citation-safe validation completed."
                    ),
                    retryable=True,
                    elapsed_seconds=getattr(error, "elapsed_seconds", elapsed),
                    configured_timeout_seconds=getattr(error, "timeout_seconds", elapsed),
                    retry_count=getattr(error, "retry_count", 0),
                    diagnostics={
                        "candidate_count": len(evidence),
                        "candidate_counts_by_document": dict(
                            Counter(item.document_id for item in evidence)
                        ),
                        "assessment_chunk_count": len(assessment_evidence),
                        "assessment_characters": input_characters,
                        "assessment_counts_by_document": counts,
                    },
                ) from error
        by_id = {item.chunk_id: item for item in evidence}
        permitted = {
            item.evidence_id
            for item in assessment.items
            if item.relevance in {"direct_answer", "compatible_rounding", "supporting_definition"}
        }
        valid_selected_ids = [
            evidence_id
            for evidence_id in assessment.selected_evidence_ids
            if evidence_id in by_id and evidence_id in permitted
        ]
        selected = [by_id[evidence_id] for evidence_id in valid_selected_ids]
        direct = {
            item.evidence_id
            for item in assessment.items
            if item.relevance in {"direct_answer", "compatible_rounding"}
            and item.evidence_id in by_id
            and item.entity_match
            and item.metric_match
            and item.year_match
            and item.population_scope_match
            and item.residence_scope_match
            and item.has_explicit_value
            and item.has_unit
            and item.unit_compatible
        }
        if state["task_type"] in {"artifact_chart", "artifact_table"}:
            valid_selected_ids = [item_id for item_id in valid_selected_ids if item_id in direct]
            selected = [by_id[evidence_id] for evidence_id in valid_selected_ids]
        if not valid_selected_ids:
            valid_selected_ids = [item.chunk_id for item in evidence if item.chunk_id in direct]
            selected = [by_id[evidence_id] for evidence_id in valid_selected_ids]
        direct_by_target = {
            target: [
                evidence_id
                for evidence_id in direct
                if (
                    by_id[evidence_id].region.casefold() == target.casefold()
                    or by_id[evidence_id].document_id.casefold() == target.casefold()
                )
            ]
            for target in required_targets
        }
        target_complete = not required_targets or all(direct_by_target.values())
        sufficient = bool(assessment.sufficient and direct and target_complete)
        limitations = (
            state.get("available_limitations", [])
            if assessment.coverage_limitation_material and not target_complete
            else []
        )
        return {
            "evidence_assessment": assessment,
            "selected_evidence": selected,
            "evidence_sufficient": sufficient,
            "limitations": limitations,
            "trace_events": [
                *state.get("trace_events", []),
                self._event(
                    "evidence_budget_packing",
                    "assess_evidence",
                    required_targets=required_targets,
                    character_budget=self.assessment_max_characters,
                    chunk_budget=self.assessment_max_chunks,
                    reserved_by_target=packing.reserved_by_target,
                    included_evidence_ids=[item.chunk_id for item in assessment_evidence],
                    excluded_evidence=packing.excluded_reasons,
                    counts_by_target=counts_by_target,
                ),
                self._event("model_invocation", "assess_evidence", operation="evidence_selection"),
                self._event(
                    "evidence_assessment",
                    "assess_evidence",
                    sufficient=sufficient,
                    candidate_count=len(evidence),
                    assessment_chunk_count=len(assessment_evidence),
                    assessment_characters=input_characters,
                    assessment_counts_by_document=counts,
                    assessment_counts_by_target=counts_by_target,
                    selected_evidence_by_target=direct_by_target,
                    selected_evidence_ids=valid_selected_ids,
                    categories={item.evidence_id: item.relevance for item in assessment.items},
                    explanation=assessment.explanation,
                ),
            ],
        }

    async def synthesize(self, state: AgentState) -> dict[str, object]:
        if state["task_type"] == "source_support":
            draft, calculations = source_support_draft(
                state.get("source_support_claims", []), state.get("selected_evidence", [])
            )
            return {
                "draft_answer": draft,
                "calculations": calculations,
                "trace_events": [
                    *state.get("trace_events", []),
                    self._event(
                        "structured_draft",
                        "synthesize",
                        claim_count=len(draft.claims),
                        refusal=False,
                        synthesis_method="validated_claim_history",
                    ),
                ],
            }
        if state["task_type"] in {"artifact_chart", "artifact_table"}:
            artifact_type: Literal["chart", "table"] = (
                "chart" if state["task_type"] == "artifact_chart" else "table"
            )
            response = DraftAnswer(
                answer_markdown=(
                    f"The {artifact_type} request was understood, but artifact execution is "
                    "pending implementation in Prompt 5."
                ),
                claims=[],
                refusal=False,
            )
            return {
                "draft_answer": response,
                "artifacts": [
                    ArtifactResult(
                        status="capability_pending",
                        artifact_type=artifact_type,
                        message="Artifact execution is pending Prompt 5.",
                    )
                ],
            }
        draft = await self.model.synthesize(
            state["resolved_query"],
            state["task_type"],
            state.get("selected_evidence", []),
            state.get("skill_instructions"),
            state.get("limitations", []),
            state.get("calculations", []),
        )
        draft = enrich_source_claims(
            draft, state.get("selected_evidence", []), state["resolved_query"]
        )
        if state["task_type"] == "comparison":
            draft = add_deterministic_derived_claims(draft, state.get("calculations", []))
        return {
            "draft_answer": draft,
            "trace_events": [
                *state.get("trace_events", []),
                self._event("model_invocation", "synthesize", operation="structured_answer"),
                self._event(
                    "structured_draft",
                    "synthesize",
                    claim_count=len(draft.claims),
                    refusal=draft.refusal,
                    claims=[
                        {
                            "claim_id": item.claim_id,
                            "text": item.text[:500],
                            "evidence_ids": item.evidence_ids,
                            "document_derived": item.document_derived,
                            "metric": item.metric,
                            "region": item.region,
                            "year": item.year,
                            "population_scope": item.population_scope,
                            "residence_scope": item.residence_scope,
                            "value": item.value,
                            "unit": item.unit,
                            "derivation": (
                                item.derivation.model_dump(mode="json")
                                if item.derivation is not None
                                else None
                            ),
                        }
                        for item in draft.claims
                    ],
                    answer_preview=draft.answer_markdown[:500],
                ),
            ],
        }

    @staticmethod
    def _expected_artifacts(task_type: str, columns: list[str]) -> list[ExpectedArtifact]:
        manifest = ExpectedArtifact(
            artifact_type="manifest",
            title="Source manifest",
            filename="source-manifest.json",
            media_type="application/json",
        )
        if task_type == "artifact_chart":
            return [
                ExpectedArtifact(
                    artifact_type="chart",
                    title="Chart",
                    filename="chart.png",
                    media_type="image/png",
                ),
                ExpectedArtifact(
                    artifact_type="data",
                    title="Plotted data",
                    filename="plotted-data.csv",
                    media_type="text/csv",
                    expected_columns=columns,
                ),
                manifest,
            ]
        return [
            ExpectedArtifact(
                artifact_type="data",
                title="Table data",
                filename="table.csv",
                media_type="text/csv",
                expected_columns=columns,
            ),
            ExpectedArtifact(
                artifact_type="table",
                title="Readable table",
                filename="table.md",
                media_type="text/markdown",
            ),
            manifest,
        ]

    async def prepare_artifact(self, state: AgentState) -> dict[str, object]:
        requirement = state.get("artifact_requirement")
        if requirement is None:
            raise AgentOperationalError(
                code="MODEL_OUTPUT_INVALID",
                node="prepare_artifact",
                message="Artifact preparation failed because its data requirements were invalid.",
                retryable=False,
                elapsed_seconds=None,
                configured_timeout_seconds=None,
                retry_count=0,
                diagnostics={"failure_stage": "requirement_validation"},
            )
        selected_evidence = state.get("selected_evidence", [])
        try:
            validate_trusted_artifact_evidence(selected_evidence)
        except TrustedEvidenceProvenanceError as error:
            raise AgentOperationalError(
                code="INTERNAL_PROVENANCE_INVALID",
                node="prepare_artifact",
                message=(
                    "Artifact preparation failed because trusted source provenance was invalid."
                ),
                retryable=False,
                elapsed_seconds=None,
                configured_timeout_seconds=None,
                retry_count=0,
                diagnostics={
                    "failure_stage": "trusted_evidence_validation",
                    "reason_code": error.code,
                    "affected_evidence_id": error.evidence_id,
                    "checksum_present": error.checksum_present,
                    "executor_submitted": False,
                },
            ) from error
        proposal = None
        try:
            proposal = await self.model.propose_artifact_dataset(
                state["resolved_query"],
                state["task_type"],
                selected_evidence,
                rank_all=requirement.rank_all,
            )
            dataset = hydrate_artifact_dataset(
                proposal,
                requirement,
                selected_evidence,
                state["resolved_query"],
            )
            validate_dataset(
                dataset,
                selected_evidence,
                requirement,
            )
        except ProviderCallTimeout as error:
            raise AgentOperationalError(
                code="MODEL_TIMEOUT",
                node="prepare_artifact",
                message="Artifact preparation timed out before a trusted dataset was created.",
                retryable=True,
                elapsed_seconds=error.elapsed_seconds,
                configured_timeout_seconds=error.timeout_seconds,
                retry_count=error.retry_count,
                diagnostics={"failure_stage": "proposal_generation"},
            ) from error
        except ProviderOperationalError as error:
            messages = {
                "MODEL_SCHEMA_REJECTED": (
                    "Artifact preparation failed because the model could not accept its response "
                    "schema."
                ),
                "MODEL_RATE_LIMITED": (
                    "Artifact preparation is temporarily unavailable because the model is busy."
                ),
                "MODEL_AUTHENTICATION_FAILED": (
                    "Artifact preparation could not authenticate with the configured model."
                ),
                "MODEL_OUTPUT_INVALID": (
                    "Artifact preparation returned an invalid structured response."
                ),
                "MODEL_UNAVAILABLE": "Artifact preparation is temporarily unavailable.",
            }
            raise AgentOperationalError(
                code=error.code,
                node="prepare_artifact",
                message=messages[error.code],
                retryable=error.retryable,
                elapsed_seconds=None,
                configured_timeout_seconds=None,
                retry_count=0,
                diagnostics={
                    "failure_stage": "proposal_generation",
                    "provider_exception": error.provider_exception,
                    "provider_status_code": error.status_code,
                },
            ) from error
        except ValidationError as error:
            proposed_id = proposal.rows[0].evidence_id if proposal and proposal.rows else "unknown"
            trusted = next(
                (item for item in selected_evidence if item.chunk_id == proposed_id), None
            )
            raise AgentOperationalError(
                code="INTERNAL_PROVENANCE_INVALID",
                node="prepare_artifact",
                message=(
                    "Artifact preparation failed because trusted source provenance was invalid."
                ),
                retryable=False,
                elapsed_seconds=None,
                configured_timeout_seconds=None,
                retry_count=0,
                diagnostics={
                    "failure_stage": "internal_contract_validation",
                    "reason_code": "INTERNAL_MODEL_VALIDATION_FAILED",
                    "affected_evidence_id": proposed_id,
                    "checksum_present": bool(trusted and trusted.source_checksum),
                    "executor_submitted": False,
                },
            ) from error
        except (ProposalHydrationError, DatasetValidationError) as error:
            reason_code = (
                error.code if isinstance(error, ProposalHydrationError) else type(error).__name__
            )
            raise AgentOperationalError(
                code="MODEL_OUTPUT_INVALID",
                node="prepare_artifact",
                message=(
                    "Artifact preparation failed because proposed values could not be verified "
                    "against trusted evidence."
                ),
                retryable=False,
                elapsed_seconds=None,
                configured_timeout_seconds=None,
                retry_count=0,
                diagnostics={
                    "failure_stage": "deterministic_hydration",
                    "reason_code": reason_code,
                    "expected_artifact_type": requirement.artifact_type,
                    "authoritative_artifact_type": requirement.artifact_type,
                    "expected_population_scope": requirement.population_scope,
                    "expected_residence_scope": requirement.residence_scope,
                    "proposed_row_scopes": [
                        {
                            "population_scope": row.population_scope,
                            "residence_scope": row.residence_scope,
                        }
                        for row in proposal.rows
                    ]
                    if proposal is not None
                    else [],
                    "executor_submitted": False,
                },
            ) from error
        ranking_winner: SourceRecord | None = None
        if requirement.rank_all:
            detected_rows = count_table_entity_rows(selected_evidence)
            if detected_rows and len(dataset.rows) < detected_rows:
                raise AgentOperationalError(
                    code="MODEL_OUTPUT_INVALID",
                    node="prepare_artifact",
                    message=(
                        "Artifact preparation failed because the proposed dataset covered fewer "
                        "rows than the source table appears to contain."
                    ),
                    retryable=False,
                    elapsed_seconds=None,
                    configured_timeout_seconds=None,
                    retry_count=0,
                    diagnostics={
                        "failure_stage": "ranking_row_completeness",
                        "reason_code": "RANKING_COVERAGE_INCOMPLETE",
                        "proposed_row_count": len(dataset.rows),
                        "detected_row_count": detected_rows,
                        "executor_submitted": False,
                    },
                )
            reverse = requirement.rank_direction != "min"
            sorted_rows = sorted(
                dataset.rows, key=lambda row: cast(float, row["value"]), reverse=reverse
            )
            sorted_row_ids = [str(row["row_id"]) for row in sorted_rows]
            sorted_sources = sorted(
                dataset.source_records, key=lambda source: sorted_row_ids.index(source.row_id)
            )
            dataset = dataset.model_copy(
                update={"rows": sorted_rows, "source_records": sorted_sources}
            )
            winner_row = sorted_rows[0]
            state_region = selected_evidence[0].region if selected_evidence else None
            if state_region and str(winner_row["label"]).casefold() == state_region.casefold():
                raise AgentOperationalError(
                    code="MODEL_OUTPUT_INVALID",
                    node="prepare_artifact",
                    message=(
                        "Artifact preparation failed because the ranked row was the state/UT "
                        "total rather than an individual entity."
                    ),
                    retryable=False,
                    elapsed_seconds=None,
                    configured_timeout_seconds=None,
                    retry_count=0,
                    diagnostics={
                        "failure_stage": "ranking_aggregate_row_guard",
                        "reason_code": "RANKING_INCLUDES_AGGREGATE_ROW",
                        "winning_label": str(winner_row["label"]),
                        "executor_submitted": False,
                    },
                )
            ranking_winner = next(
                source for source in dataset.source_records if source.row_id == winner_row["row_id"]
            )
        return {
            "artifact_dataset": dataset,
            "ranking_winner": ranking_winner,
            "trace_events": [
                *state.get("trace_events", []),
                *(
                    [
                        self._event(
                            "ranking_winner_selected",
                            "prepare_artifact",
                            winning_row_id=ranking_winner.row_id,
                            winning_label=ranking_winner.region,
                            winning_value=ranking_winner.normalized_numeric_value,
                            direction=requirement.rank_direction or "max",
                            detected_row_count=detected_rows,
                            proposed_row_count=len(dataset.rows),
                        )
                    ]
                    if ranking_winner is not None
                    else []
                ),
                self._event(
                    "model_invocation", "prepare_artifact", operation="artifact_dataset_proposal"
                ),
                self._event(
                    "artifact_proposal_validation",
                    "prepare_artifact",
                    valid=True,
                    proposed_row_count=len(proposal.rows),
                    proposed_evidence_ids=[row.evidence_id for row in proposal.rows],
                    authoritative_artifact_type=requirement.artifact_type,
                    presentation_chart_kind=resolved_chart_kind(proposal, requirement),
                ),
                self._event(
                    "artifact_dataset_validation",
                    "prepare_artifact",
                    valid=True,
                    row_count=len(dataset.rows),
                    column_count=len(dataset.columns),
                    source_record_count=len(dataset.source_records),
                    evidence_ids=list(dict.fromkeys(x.chunk_id for x in dataset.source_records)),
                    required_targets=requirement.regions if requirement else [],
                    represented_targets=list(
                        dict.fromkeys(
                            source.region
                            for source in dataset.source_records
                            if source.region is not None
                        )
                    ),
                ),
            ],
        }

    async def generate_artifact_code(self, state: AgentState) -> dict[str, object]:
        dataset = state["artifact_dataset"]
        if dataset is None:
            return {"artifact_errors": ["Missing validated artifact dataset"]}
        generated = await self.model.generate_artifact_code(
            dataset, state.get("skill_instructions") or ""
        )
        policy = validate_code(generated.code)
        return {
            "generated_code": generated.code,
            "artifact_errors": [] if policy.valid else ["CODE_POLICY_VIOLATION"],
            "trace_events": [
                *state.get("trace_events", []),
                self._event(
                    "model_invocation", "generate_artifact_code", operation="code_generation"
                ),
                self._event(
                    "code_policy_validation",
                    "generate_artifact_code",
                    valid=policy.valid,
                    code_sha256=hashlib.sha256(generated.code.encode()).hexdigest(),
                    error_count=len(policy.errors),
                    errors=list(policy.errors),
                ),
            ],
        }

    async def execute_artifact(self, state: AgentState) -> dict[str, object]:
        dataset = state.get("artifact_dataset")
        code = state.get("generated_code")
        attempt = state.get("artifact_attempt", 0) + 1
        if dataset is None or not code or self.execution_queue is None:
            return {
                "artifact_attempt": attempt,
                "artifact_errors": ["EXECUTOR_UNAVAILABLE"],
                "execution_result": None,
            }
        policy = validate_code(code)
        if not policy.valid:
            return {
                "artifact_attempt": attempt,
                "artifact_errors": ["CODE_POLICY_VIOLATION"],
                "execution_result": None,
            }
        manifest = SourceManifest(
            dataset_title=dataset.title,
            source_records=dataset.source_records,
            computed_values=dataset.computed_values,
        )
        request = new_request(
            state["session_id"],
            state["run_id"],
            code,
            {
                "dataset": dataset.model_dump(mode="json"),
                "source_manifest": manifest.model_dump(mode="json"),
            },
            self._expected_artifacts(state["task_type"], dataset.columns),
            self.execution_timeout_seconds,
        )
        try:
            await asyncio.to_thread(self.execution_queue.submit, request)
            result = await self.execution_queue.wait(
                request.job_id, self.execution_timeout_seconds + 5
            )
            errors = (
                [] if result.status == "succeeded" else [result.error_code or "EXECUTION_FAILED"]
            )
        except ExecutorUnavailableError:
            result = None
            errors = ["EXECUTOR_UNAVAILABLE"]
        return {
            "artifact_attempt": attempt,
            "execution_request": request,
            "execution_result": result,
            "artifact_errors": errors,
            "trace_events": [
                *state.get("trace_events", []),
                self._event(
                    "executor_submission",
                    "execute_artifact",
                    job_id=request.job_id,
                    attempt=attempt,
                    code_sha256=request.code_sha256,
                    dataset_row_count=len(dataset.rows),
                    source_record_count=len(dataset.source_records),
                    expected_filenames=[item.filename for item in request.expected_artifacts],
                ),
                self._event(
                    "executor_result",
                    "execute_artifact",
                    job_id=request.job_id,
                    status=result.status if result else "unavailable",
                    exit_code=result.exit_code if result else None,
                    timed_out=result.timed_out if result else False,
                    generated_filenames=(
                        [item.filename for item in result.artifacts] if result else []
                    ),
                    error_code=result.error_code if result else "EXECUTOR_UNAVAILABLE",
                ),
                self._event(
                    "artifact_execution",
                    "execute_artifact",
                    job_id=request.job_id,
                    attempt=attempt,
                    code_sha256=request.code_sha256,
                    status=result.status if result else "unavailable",
                    exit_code=result.exit_code if result else None,
                    timed_out=result.timed_out if result else False,
                    stdout_bytes=len(result.stdout.encode()) if result else 0,
                    stderr_bytes=len(result.stderr.encode()) if result else 0,
                    error_code=result.error_code if result else "EXECUTOR_UNAVAILABLE",
                ),
            ],
        }

    async def inspect_artifact(self, state: AgentState) -> dict[str, object]:
        result = state.get("execution_result")
        request = state.get("execution_request")
        dataset = state.get("artifact_dataset")
        if result is None or request is None or dataset is None:
            return {}
        if result.status != "succeeded":
            if self.artifact_store is not None:
                self.artifact_store.discard(request.job_id)
            return {}
        if (
            len(result.artifacts) > self.artifact_max_files
            or sum(item.byte_size for item in result.artifacts) > self.artifact_max_total_bytes
        ):
            if self.artifact_store is not None:
                self.artifact_store.discard(request.job_id)
            return {"artifact_errors": ["OUTPUT_LIMIT_EXCEEDED"]}
        if self.execution_queue is None or self.artifact_store is None:
            return {"artifact_errors": ["EXECUTOR_UNAVAILABLE"]}
        staged = self.execution_queue.root / "jobs" / request.job_id / "output"
        try:
            validate_artifact_lineage(dataset, staged)
            primary = "chart.png" if state["task_type"] == "artifact_chart" else "table.md"
            descriptor = self.artifact_store.accept(
                request,
                result,
                artifact_type="chart" if state["task_type"] == "artifact_chart" else "table",
                title=dataset.title,
                primary_filename=primary,
            )
            requirement = state.get("artifact_requirement")
            response = artifact_response(
                dataset,
                descriptor,
                state.get("selected_evidence", []),
                state["run_id"],
                ranking_winner=state.get("ranking_winner"),
                rank_direction=requirement.rank_direction if requirement else None,
            )
        except (ValueError, DatasetValidationError) as error:
            self.artifact_store.discard(request.job_id)
            return {
                "artifact_errors": ["INVALID_ARTIFACT"],
                "trace_events": [
                    *state.get("trace_events", []),
                    self._event(
                        "artifact_validation",
                        "inspect_artifact",
                        valid=False,
                        error_type=type(error).__name__,
                    ),
                ],
            }
        return {
            "artifact_descriptors": [descriptor],
            "artifacts": [descriptor],
            "final_response": response,
            "artifact_errors": [],
            "trace_events": [
                *state.get("trace_events", []),
                self._event(
                    "artifact_validation",
                    "inspect_artifact",
                    valid=True,
                    generated_filenames=[item.filename for item in result.artifacts],
                    artifact_checksums={item.filename: item.sha256 for item in result.artifacts},
                ),
            ],
        }

    async def repair_artifact(self, state: AgentState) -> dict[str, object]:
        dataset = state["artifact_dataset"]
        if dataset is None:
            return {"artifact_errors": ["Missing validated artifact dataset"]}
        result = state.get("execution_result")
        generated = await self.model.repair_artifact_code(
            dataset,
            state.get("generated_code") or "",
            state.get("skill_instructions") or "",
            (result.error_code if result else None) or state.get("artifact_errors", [""])[0],
            result.stderr[:8000] if result else "",
        )
        policy = validate_code(generated.code)
        return {
            "generated_code": generated.code,
            "artifact_errors": [] if policy.valid else ["CODE_POLICY_VIOLATION"],
            "trace_events": [
                *state.get("trace_events", []),
                self._event("artifact_repair_decision", "repair_artifact", repair=True, attempt=2),
                self._event("model_invocation", "repair_artifact", operation="code_repair"),
                self._event(
                    "code_policy_validation",
                    "repair_artifact",
                    valid=policy.valid,
                    code_sha256=hashlib.sha256(generated.code.encode()).hexdigest(),
                    error_count=len(policy.errors),
                    errors=list(policy.errors),
                ),
            ],
        }

    @staticmethod
    def _response_invariant_errors(
        state: AgentState, draft: DraftAnswer, uncited_claim_count: int
    ) -> list[str]:
        task = state["task_type"]
        answerable = task in {"lookup", "comparison", "summary", "inconsistency_analysis"}
        codes: list[str] = []
        if not draft.answer_markdown.strip():
            codes.append("EMPTY_ANSWER_TEXT")
        if draft.refusal and draft.claims:
            codes.append("ANSWER_REFUSAL_FLAG_CONFLICT")
        if answerable and state.get("evidence_sufficient") and not draft.refusal:
            if not draft.claims:
                codes.extend(
                    ["EMPTY_CLAIMS_FOR_ANSWERABLE_TASK", "ANSWER_WITHOUT_STRUCTURED_CLAIM"]
                )
            if uncited_claim_count:
                codes.append("CLAIM_WITHOUT_EVIDENCE")
            if task == "comparison":
                evidence = {item.chunk_id: item for item in state.get("selected_evidence", [])}
                cited_regions = {
                    evidence[evidence_id].region.casefold()
                    for claim in draft.claims
                    for evidence_id in claim.evidence_ids
                    if evidence_id in evidence
                }
                targets = {item.casefold() for item in state["classification"].regions}
                if not targets <= cited_regions:
                    codes.append("MISSING_COMPARISON_TARGET")
                calculations = state.get("calculations", [])
                has_derived = any(
                    set(calculation.evidence_ids) <= set(claim.evidence_ids)
                    for calculation in calculations
                    for claim in draft.claims
                )
                if not calculations or not has_derived:
                    codes.append("DERIVED_CLAIM_MISSING_INPUT_CITATIONS")
        return list(dict.fromkeys(codes))

    async def validate_citations(self, state: AgentState) -> dict[str, object]:
        draft = state.get("draft_answer")
        if draft is None:
            return {"errors": [*state.get("errors", []), "Missing draft answer"]}
        evidence = state.get("selected_evidence", [])
        result = validate_and_materialize_citations(
            draft,
            evidence,
            state.get("calculations", []),
        )
        source_draft = draft.model_copy(
            update={"claims": [claim for claim in draft.claims if not claim.document_derived]}
        )
        support = (
            SupportAssessment(
                supported_claim_ids=[claim.claim_id for claim in source_draft.claims],
                explanation="Previously validated claims were rehydrated from current evidence",
            )
            if state["task_type"] == "source_support"
            else await self.model.assess_support(source_draft, evidence)
            if source_draft.claims
            else SupportAssessment(explanation="No factual claims require semantic assessment")
        )
        errors = list(result.errors)
        error_codes = list(result.error_codes)
        if support.unsupported_claim_ids:
            errors.append(f"Unsupported claims: {', '.join(support.unsupported_claim_ids)}")
            error_codes.append("UNSUPPORTED_CLAIM")
        invariant_codes = self._response_invariant_errors(
            state, draft, sum(not item.citation_ids for item in result.claims)
        )
        error_codes.extend(invariant_codes)
        if invariant_codes:
            errors.extend(f"Response invariant failed: {code}" for code in invariant_codes)
        error_codes = list(dict.fromkeys(error_codes))
        if errors:
            return {
                "support_assessment": support,
                "answer_claims": result.claims,
                "citations": result.citations,
                "errors": [*state.get("errors", []), *errors],
                "validation_error_codes": error_codes,
                "trace_events": [
                    *state.get("trace_events", []),
                    *(
                        [
                            self._event(
                                "model_invocation",
                                "validate_citations",
                                operation="support_assessment",
                            )
                        ]
                        if draft.claims
                        else []
                    ),
                    self._event(
                        "citation_validation",
                        "validate_citations",
                        valid=False,
                        claim_count=len(draft.claims),
                        citation_count=len(result.citations),
                        error_codes=error_codes,
                        quote_diagnostics=[
                            item.model_dump(mode="json") for item in result.quote_diagnostics
                        ],
                    ),
                ],
            }
        sources = "\n".join(f"- {render_citation(item)}" for item in result.citations)
        rendered_claims = "\n\n".join(item.text for item in result.claims)
        answer_body = (
            draft.answer_markdown
            if state["task_type"] == "source_support"
            else rendered_claims
            if result.claims and not draft.refusal
            else draft.answer_markdown
        )
        answer = answer_body + (
            f"\n\nSources:\n{sources}" if sources and state["task_type"] != "source_support" else ""
        )
        response = AgentResponse(
            answer_markdown=answer,
            claims=result.claims,
            citations=result.citations,
            artifacts=state.get("artifacts", []),
            limitations=[*state.get("limitations", []), *draft.limitations],
            refusal=draft.refusal,
            trace_id=state["run_id"],
        )
        return {
            "support_assessment": support,
            "answer_claims": result.claims,
            "citations": result.citations,
            "final_response": response,
            "validation_error_codes": [],
            "trace_events": [
                *state.get("trace_events", []),
                *(
                    [
                        self._event(
                            "model_invocation",
                            "validate_citations",
                            operation="support_assessment",
                        )
                    ]
                    if draft.claims
                    else []
                ),
                self._event(
                    "citation_validation",
                    "validate_citations",
                    valid=True,
                    claim_count=len(result.claims),
                    citation_count=len(result.citations),
                    error_codes=[],
                    quote_diagnostics=[
                        item.model_dump(mode="json") for item in result.quote_diagnostics
                    ],
                ),
                self._event(
                    "validated_citations",
                    "validate_citations",
                    citations=[
                        {
                            "citation_id": item.citation_id,
                            "chunk_id": item.chunk_id,
                            "document_id": item.document_id,
                            "page_number": item.page_number,
                            "snippet": item.snippet,
                            "start_offset": item.evidence_span.start_offset,
                            "end_offset": item.evidence_span.end_offset,
                        }
                        for item in result.citations
                    ],
                ),
            ],
        }

    async def repair(self, state: AgentState) -> dict[str, object]:
        draft = state.get("draft_answer")
        if draft is None:
            return {"retry_count": 1}
        repaired = await self.model.repair(
            draft,
            state.get("selected_evidence", []),
            state.get("errors", []),
            task_type=state["task_type"],
            evidence_sufficient=state.get("evidence_sufficient", False),
            error_codes=state.get("validation_error_codes", []),
        )
        repaired = enrich_source_claims(
            repaired, state.get("selected_evidence", []), state["resolved_query"]
        )
        if state["task_type"] == "comparison":
            repaired = add_deterministic_derived_claims(repaired, state.get("calculations", []))
        return {
            "draft_answer": repaired,
            "retry_count": 1,
            "errors": [],
            "trace_events": [
                *state.get("trace_events", []),
                self._event("model_invocation", "repair", operation="answer_repair"),
                self._event("repair", "repair", attempt=1),
                self._event(
                    "structured_repair",
                    "repair",
                    claim_count=len(repaired.claims),
                    refusal=repaired.refusal,
                    claims=[
                        {
                            "claim_id": item.claim_id,
                            "text": item.text[:500],
                            "evidence_ids": item.evidence_ids,
                            "document_derived": item.document_derived,
                            "metric": item.metric,
                            "region": item.region,
                            "year": item.year,
                            "population_scope": item.population_scope,
                            "residence_scope": item.residence_scope,
                            "value": item.value,
                            "unit": item.unit,
                            "derivation": (
                                item.derivation.model_dump(mode="json")
                                if item.derivation is not None
                                else None
                            ),
                        }
                        for item in repaired.claims
                    ],
                    answer_preview=repaired.answer_markdown[:500],
                ),
            ],
        }

    async def graceful_response(self, state: AgentState) -> dict[str, object]:
        task = state.get("task_type")
        if task == "clarification":
            answer = (
                state["classification"].clarification_question
                or "Could you clarify which Census report or measure you mean?"
            )
            refusal = False
        elif task == "out_of_scope":
            answer = "I’m limited to answering questions grounded in the supplied Census documents."
            refusal = True
        elif task in {"artifact_chart", "artifact_table"}:
            if not state.get("evidence_sufficient"):
                answer = "I couldn’t find citation-safe evidence for the requested artifact."
            else:
                answer = "I couldn’t create a validated artifact from the available evidence."
            refusal = True
        else:
            if state.get("retry_count", 0) > 0 and state.get("validation_error_codes"):
                answer = (
                    "I found relevant evidence, but couldn’t produce a response that passed "
                    "citation validation."
                )
            else:
                answer = "I couldn’t find citation-safe evidence for that in the indexed documents."
            refusal = True
        limitations = state.get("limitations", [])
        if limitations:
            answer += f"\n\n{limitations[0]}"
        return {
            "final_response": AgentResponse(
                answer_markdown=answer,
                limitations=limitations,
                refusal=refusal,
                trace_id=state["run_id"],
            ),
            "trace_events": [
                *state.get("trace_events", []),
                self._event("refusal" if refusal else "clarification", "graceful_response"),
            ],
        }

    async def persist_result(self, state: AgentState) -> dict[str, object]:
        response = state["final_response"]
        if response is None:
            raise RuntimeError("Cannot persist a missing agent response")
        updates: dict[str, object] = {
            "messages": [AIMessage(content=response.answer_markdown)],
            "trace_events": [
                *state.get("trace_events", []),
                self._event(
                    "node_end",
                    "persist_result",
                    refusal=response.refusal,
                    claim_count=len(response.claims),
                    citation_count=len(response.citations),
                    validation_error_codes=state.get("validation_error_codes", []),
                ),
            ],
        }
        if (
            not response.refusal
            and response.claims
            and response.citations
            and not state.get("validation_error_codes")
            and state.get("task_type") != "source_support"
        ):
            turn = build_validated_claim_turn(
                state["run_id"],
                state["task_type"],
                response,
                state.get("selected_evidence", []),
            )
            history = [*state.get("validated_claim_history", []), turn]
            updates["validated_claim_history"] = history[-8:]
        return updates

    @staticmethod
    def _classification_route(state: AgentState) -> str:
        if state["task_type"] == "out_of_scope":
            return "stop"
        if state["task_type"] == "clarification" and len(state.get("messages", [])) <= 1:
            return "stop"
        return "continue"

    @staticmethod
    def _resolution_route(state: AgentState) -> str:
        return "stop" if state["task_type"] == "clarification" else "continue"

    @staticmethod
    def _plan_route(state: AgentState) -> str:
        return "stop" if state["task_type"] == "clarification" else "continue"

    @staticmethod
    def _evidence_route(state: AgentState) -> str:
        if not state.get("evidence_sufficient"):
            return "stop"
        return (
            "artifact"
            if state.get("task_type") in {"artifact_chart", "artifact_table"}
            else "continue"
        )

    @staticmethod
    def _artifact_prepare_route(state: AgentState) -> str:
        return "continue" if state.get("artifact_dataset") is not None else "stop"

    @staticmethod
    def _artifact_result_route(state: AgentState) -> str:
        if state.get("final_response") is not None:
            return "done"
        errors = set(state.get("artifact_errors", []))
        correctable = {"EXECUTION_FAILED", "INVALID_ARTIFACT", "ARTIFACT_NOT_CREATED"}
        if errors & correctable and state.get("artifact_attempt", 0) < 2:
            return "repair"
        return "stop"

    @staticmethod
    def _validation_route(state: AgentState) -> str:
        if state.get("final_response") is not None:
            return "done"
        return "repair" if state.get("retry_count", 0) == 0 else "stop"
