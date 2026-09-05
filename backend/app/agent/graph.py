import asyncio
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any, Literal, cast

from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage
from langgraph.graph import END, START, StateGraph

from backend.app.agent.calculations import (
    add_deterministic_derived_claims,
    enrich_source_claims,
    validated_calculation,
)
from backend.app.agent.checkpoint import checkpoint_node
from backend.app.agent.citations import render_citation, validate_and_materialize_citations
from backend.app.agent.errors import AgentOperationalError
from backend.app.agent.evidence import bound_assessment_evidence
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
    ArtifactResult,
    CalculationResult,
    DraftAnswer,
    EvidenceAssessment,
    SupportAssessment,
    TaskClassification,
    ToolCallRecord,
    TraceEvent,
)
from backend.app.agent.provider import AgentModel, ProviderCallTimeout
from backend.app.agent.tools import AgentTools, ArithmeticInput, SearchDocumentsInput
from backend.app.retrieval.models import DocumentSummary, RetrievedEvidence


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
    ) -> None:
        self.model = model
        self.tools = tools
        self.max_tool_calls = max_tool_calls
        self.memory_turn_threshold = memory_turn_threshold
        self.assessment_max_characters = assessment_max_characters
        self.assessment_max_chunks = assessment_max_chunks

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
            {"continue": "synthesize", "stop": "graceful_response"},
        )
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
        updates: dict[str, object] = {
            "resolved_query": resolved_query,
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
        artifact = task_type in {"artifact_chart", "artifact_table"}
        steps = ["retrieve citation-safe evidence", "validate evidence"]
        if artifact:
            steps.append("return capability_pending without executing an artifact")
        else:
            steps.extend(["synthesize cited claims", "validate claim citations"])
        return {
            "plan": AgentPlan(steps=steps, retrieval_top_k=10, capability_pending=artifact),
            "trace_events": [
                *state.get("trace_events", []),
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
        if state["task_type"] == "comparison" and classification.regions:
            targets.extend((None, [region]) for region in classification.regions)
        elif state["task_type"] == "comparison" and classification.document_ids:
            targets.extend(([document], None) for document in classification.document_ids)
        else:
            targets.append((classification.document_ids or None, classification.regions or None))
        if state["task_type"] == "summary":
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
                value = SearchDocumentsInput(
                    query=query,
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
                evidence.extend(cast(list[RetrievedEvidence], result or []))
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
        assessment_evidence = bound_assessment_evidence(
            evidence,
            max_characters=self.assessment_max_characters,
            max_chunks=self.assessment_max_chunks,
        )
        input_characters = sum(len(item.text) for item in assessment_evidence)
        counts = dict(Counter(item.document_id for item in assessment_evidence))
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
        }
        if not valid_selected_ids:
            valid_selected_ids = [item.chunk_id for item in evidence if item.chunk_id in direct]
            selected = [by_id[evidence_id] for evidence_id in valid_selected_ids]
        sufficient = bool(assessment.sufficient and direct)
        limitations = (
            state.get("available_limitations", [])
            if assessment.coverage_limitation_material
            else []
        )
        return {
            "evidence_assessment": assessment,
            "selected_evidence": selected,
            "evidence_sufficient": sufficient,
            "limitations": limitations,
            "trace_events": [
                *state.get("trace_events", []),
                self._event("model_invocation", "assess_evidence", operation="evidence_selection"),
                self._event(
                    "evidence_assessment",
                    "assess_evidence",
                    sufficient=sufficient,
                    candidate_count=len(evidence),
                    assessment_chunk_count=len(assessment_evidence),
                    assessment_characters=input_characters,
                    assessment_counts_by_document=counts,
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
        return "continue" if state.get("evidence_sufficient") else "stop"

    @staticmethod
    def _validation_route(state: AgentState) -> str:
        if state.get("final_response") is not None:
            return "done"
        return "repair" if state.get("retry_count", 0) == 0 else "stop"
