import asyncio
import warnings
from pathlib import Path
from typing import Any, cast

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

from backend.app.agent.checkpoint import CHECKPOINT_SCHEMA_VERSION
from backend.app.agent.graph import AgentGraph
from backend.app.agent.models import (
    CalculationRequest,
    CalculationResult,
    DraftAnswer,
    DraftClaim,
    EvidenceAssessment,
    EvidenceAssessmentItem,
    ResolvedQuery,
    SupportAssessment,
    TaskClassification,
    TraceEvent,
)
from backend.app.agent.persistence import safe_trace_details
from backend.app.agent.service import AgentChatError, AgentService
from backend.app.agent.skills import SkillRegistry
from backend.app.agent.tools import AgentTools, ArithmeticInput
from backend.app.config import Settings
from backend.app.retrieval.models import RetrievedEvidence


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def census_evidence(region: str = "Karnataka", page_number: int = 10) -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk_id=f"chunk-{region.casefold().replace(' ', '-')}-{page_number}",
        text=f"The literacy rate in {region} was 75.36 per cent in 2011.",
        document_title=f"Census 2011 {region}",
        document_id=f"doc-{region.casefold().replace(' ', '-')}",
        region=region,
        page_number=page_number,
        citation_snippet=f"The literacy rate in {region} was 75.36 per cent in 2011.",
        section_path=["Literacy"],
        extraction_method="provided_markdown",
        coverage_status="indexed_provided_markdown",
        source_checksum="a" * 64,
        retrieval_score=1.0,
    )


class FakeModel:
    def __init__(self) -> None:
        self.context_sizes: list[tuple[str, int]] = []
        self.repairs = 0

    async def classify(self, query: str, context: list[Any]) -> TaskClassification:
        self.context_sizes.append((query, len(context)))
        if "France" in query:
            return TaskClassification(task_type="out_of_scope", reason="Not supplied Census data")
        if "Summarize" in query:
            return TaskClassification(
                task_type="summary",
                document_ids=["doc-karnataka"],
                reason="Document summary",
            )
        if query == "How does that compare?" and not context:
            return TaskClassification(
                task_type="clarification",
                reason="Missing referent",
                clarification_question="What should I compare?",
            )
        if "compare" in query.casefold():
            return TaskClassification(
                task_type="comparison",
                regions=["Karnataka", "Odisha"],
                reason="Cross-region comparison",
            )
        return TaskClassification(task_type="lookup", regions=["Karnataka"], reason="Lookup")

    async def resolve(self, query: str, context: list[Any]) -> ResolvedQuery:
        if "compare" in query.casefold():
            return ResolvedQuery(
                query="Compare 2011 literacy rates for Karnataka and Odisha",
                task_type="comparison",
                regions=["Karnataka", "Odisha"],
            )
        return ResolvedQuery(query=query)

    async def assess_evidence(
        self, query: str, task_type: str, evidence: list[RetrievedEvidence]
    ) -> EvidenceAssessment:
        del task_type
        direct = "quantum" not in query.casefold()
        return EvidenceAssessment(
            items=[
                EvidenceAssessmentItem(
                    evidence_id=item.chunk_id,
                    relevance="direct_answer" if direct else "related_non_answering",
                    entity_match=True,
                    metric_match=True,
                    year_match=True,
                    has_explicit_value=True,
                    has_unit=True,
                    reason="Direct test evidence",
                )
                for item in evidence
            ],
            selected_evidence_ids=[item.chunk_id for item in evidence] if direct else [],
            sufficient=direct,
            explanation="Direct evidence exists" if direct else "No direct evidence",
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
        del query, task_type, skill, limitations
        claims = [
            DraftClaim(
                claim_id=f"claim-{index}",
                text=item.citation_snippet,
                evidence_ids=[item.chunk_id],
            )
            for index, item in enumerate(evidence)
        ]
        claims.extend(
            DraftClaim(
                claim_id=f"calculation-{index}",
                text=f"The calculated difference is {item.result} percentage points.",
                evidence_ids=item.evidence_ids,
                document_derived=True,
            )
            for index, item in enumerate(calculations)
        )
        return DraftAnswer(
            answer_markdown="Citation-safe result.",
            claims=claims,
        )

    async def assess_support(
        self, draft: DraftAnswer, evidence: list[RetrievedEvidence]
    ) -> SupportAssessment:
        del evidence
        return SupportAssessment(
            supported_claim_ids=[item.claim_id for item in draft.claims], explanation="Supported"
        )

    async def extract_calculations(
        self, query: str, evidence: list[RetrievedEvidence]
    ) -> list[CalculationRequest]:
        if "compare" in query.casefold() and len(evidence) >= 2:
            return [
                CalculationRequest(
                    description="Difference",
                    operation="difference",
                    values=[75.36, 75.36],
                    evidence_ids=[evidence[0].chunk_id, evidence[1].chunk_id],
                )
            ]
        return []

    async def repair(
        self,
        draft: DraftAnswer,
        evidence: list[RetrievedEvidence],
        errors: list[str],
        **context: object,
    ) -> DraftAnswer:
        del draft, errors, context
        self.repairs += 1
        return await self.synthesize("", "lookup", evidence, None, [], [])

    async def summarize_memory(self, context: list[Any]) -> str:
        del context
        return "Open user referent only; no source facts retained."


class RepairingModel(FakeModel):
    async def synthesize(
        self,
        query: str,
        task_type: str,
        evidence: list[RetrievedEvidence],
        skill: str | None,
        limitations: list[str],
        calculations: list[CalculationResult],
    ) -> DraftAnswer:
        del query, task_type, evidence, skill, limitations, calculations
        return DraftAnswer(
            answer_markdown="Unsafe first draft",
            claims=[
                DraftClaim(claim_id="bad-claim", text="Unsupported", evidence_ids=["invented-id"])
            ],
        )

    async def repair(
        self,
        draft: DraftAnswer,
        evidence: list[RetrievedEvidence],
        errors: list[str],
        **context: object,
    ) -> DraftAnswer:
        del context
        self.repairs += 1
        return await FakeModel.synthesize(self, "", "lookup", evidence, None, errors, [])


class EmptyRepairingModel(RepairingModel):
    async def repair(
        self,
        draft: DraftAnswer,
        evidence: list[RetrievedEvidence],
        errors: list[str],
        **context: object,
    ) -> DraftAnswer:
        del draft, evidence, errors, context
        self.repairs += 1
        return DraftAnswer(answer_markdown="Unable to answer", claims=[], refusal=False)


class TimeoutAfterSuccessModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.fail_assessment = False
        self.cancelled = False

    async def assess_evidence(
        self, query: str, task_type: str, evidence: list[RetrievedEvidence]
    ) -> EvidenceAssessment:
        if not self.fail_assessment:
            return await super().assess_evidence(query, task_type, evidence)

        async def slow() -> None:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                self.cancelled = True
                raise

        await asyncio.wait_for(slow(), timeout=0.01)
        raise AssertionError("unreachable")


class DroppedComparisonTargetModel(FakeModel):
    async def resolve(self, query: str, context: list[Any]) -> ResolvedQuery:
        del query, context
        return ResolvedQuery(
            query="What was the literacy rate in Odisha in 2011?",
            task_type="comparison",
            regions=["Karnataka", "Odisha"],
        )


class NoModelCalls:
    def __getattr__(self, name: str) -> Any:
        async def fail(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise AssertionError(f"Source-support route called model method {name}")

        return fail


class FakeTools:
    def __init__(self, skills: SkillRegistry) -> None:
        self.skills = skills
        self.search_calls = 0

    async def search_documents(self, value: Any) -> list[RetrievedEvidence]:
        self.search_calls += 1
        regions = value.regions or ["Karnataka"]
        return [census_evidence(region) for region in regions]

    async def get_evidence_by_ids(
        self, evidence_ids: list[str], expected: dict[str, Any]
    ) -> list[RetrievedEvidence]:
        return [
            census_evidence(
                expected[evidence_id].document_id.removeprefix("doc-").title(),
                expected[evidence_id].page_number,
            ).model_copy(update={"chunk_id": evidence_id})
            for evidence_id in evidence_ids
        ]

    async def list_documents(self) -> list[Any]:
        return []

    async def collect_summary_evidence(
        self, document_id: str, *, max_sections: int = 24
    ) -> list[RetrievedEvidence]:
        del document_id, max_sections
        return [census_evidence(page_number=10), census_evidence(page_number=52)]

    async def expand_candidate_pages(
        self, candidates: list[RetrievedEvidence], *, max_pages_per_document: int = 3
    ) -> list[RetrievedEvidence]:
        del max_pages_per_document
        return candidates

    async def get_document_coverage(self, document_id: str) -> tuple[None, list[Any]]:
        del document_id
        return None, []

    async def list_skills(self) -> list[Any]:
        return self.skills.list_skills()

    async def read_skill(self, skill_name: str) -> Any:
        return self.skills.read_skill(skill_name)

    @staticmethod
    def calculate(value: ArithmeticInput) -> float:
        return AgentTools.calculate(value)


def settings(tmp_path: Path) -> Settings:
    return Settings.model_validate(
        {
            "google_cloud_project": "test-project",
            "workspace_root": tmp_path / "workspace",
            "skills_dir": tmp_path / "skills",
            "data_root": tmp_path / "data",
        }
    )


def test_lookup_memory_survives_reconstruction_and_sessions_are_isolated(tmp_path: Path) -> None:
    model = FakeModel()
    tools = FakeTools(SkillRegistry(tmp_path / "skills"))
    first_service = AgentService(settings(tmp_path), model, tools)  # type: ignore[arg-type]
    first_session = run(first_service.create_session())
    other_session = run(first_service.create_session())
    first = run(first_service.chat(first_session.session_id, "What is Karnataka literacy?"))
    assert first.citations and all(claim.citation_ids for claim in first.claims)
    assert first_service.get_trace(first.trace_id) is not None

    reconstructed = AgentService(settings(tmp_path), model, tools)  # type: ignore[arg-type]
    second = run(reconstructed.chat(first_session.session_id, "How does that compare?"))
    assert tools.search_calls == 3
    isolated = run(reconstructed.chat(other_session.session_id, "What is Karnataka literacy?"))

    assert {citation.document_id for citation in second.citations} == {
        "doc-karnataka",
        "doc-odisha",
    }
    assert isolated.citations
    sizes = dict(model.context_sizes)
    assert sizes["How does that compare?"] > 0


def test_source_support_uses_validated_history_without_semantic_search(tmp_path: Path) -> None:
    model = FakeModel()
    tools = FakeTools(SkillRegistry(tmp_path / "skills"))
    first_service = AgentService(settings(tmp_path), model, tools)  # type: ignore[arg-type]
    session = run(first_service.create_session())
    run(first_service.chat(session.session_id, "What is Karnataka literacy?"))
    run(first_service.chat(session.session_id, "How does that compare with Odisha?"))
    search_calls = tools.search_calls

    reconstructed = AgentService(settings(tmp_path), NoModelCalls(), tools)  # type: ignore[arg-type]
    response = run(
        reconstructed.chat(session.session_id, "Which source pages support those values?")
    )

    assert not response.refusal
    assert tools.search_calls == search_calls
    assert {item.document_id for item in response.citations} == {
        "doc-karnataka",
        "doc-odisha",
    }
    assert "physical PDF page" in response.answer_markdown
    assert "|" not in response.answer_markdown
    status = run(reconstructed.get_context_status(session.session_id))
    assert status.has_validated_comparison


def test_refused_turn_does_not_erase_validated_claim_history(tmp_path: Path) -> None:
    model = FakeModel()
    tools = FakeTools(SkillRegistry(tmp_path / "skills"))
    service = AgentService(settings(tmp_path), model, tools)  # type: ignore[arg-type]
    session = run(service.create_session())
    run(service.chat(session.session_id, "What is Karnataka literacy?"))
    run(service.chat(session.session_id, "How does that compare with Odisha?"))
    refused = run(service.chat(session.session_id, "What is quantum flux?"))
    assert refused.refusal

    response = run(service.chat(session.session_id, "Which pages support those values?"))
    assert not response.refusal
    assert {item.document_id for item in response.citations} == {
        "doc-karnataka",
        "doc-odisha",
    }


def test_source_support_memory_is_session_isolated(tmp_path: Path) -> None:
    service = AgentService(
        settings(tmp_path),
        FakeModel(),
        FakeTools(SkillRegistry(tmp_path / "skills")),  # type: ignore[arg-type]
    )
    populated = run(service.create_session())
    empty = run(service.create_session())
    run(service.chat(populated.session_id, "What is Karnataka literacy?"))
    response = run(service.chat(empty.session_id, "Which pages support those values?"))
    assert not response.refusal
    assert not response.claims
    assert "Which previous answer" in response.answer_markdown


def test_strict_new_checkpoint_uses_json_safe_application_values_and_memory_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")
    model = FakeModel()
    tools = FakeTools(SkillRegistry(tmp_path / "skills"))
    service = AgentService(settings(tmp_path), model, tools)  # type: ignore[arg-type]
    session = run(service.create_session())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = run(service.chat(session.session_id, "What is Karnataka literacy?"))
    assert first.claims
    assert not any("unregistered" in str(item.message).casefold() for item in caught)

    async def latest_values() -> dict[str, Any]:
        async with AsyncSqliteSaver.from_conn_string(str(service.sessions.database)) as saver:
            async for item in saver.alist(
                {"configurable": {"thread_id": session.session_id}}, limit=1
            ):
                return item.checkpoint["channel_values"]
        raise AssertionError("checkpoint missing")

    values = run(latest_values())
    assert values["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION

    def assert_safe(value: Any) -> None:
        assert not isinstance(value, BaseModel)
        if isinstance(value, dict):
            for nested in value.values():
                assert_safe(nested)
        elif isinstance(value, list):
            for nested in value:
                assert_safe(nested)
        else:
            assert value is None or isinstance(value, str | int | float | bool)

    for key, value in values.items():
        if key != "messages":
            assert_safe(value)

    reconstructed = AgentService(settings(tmp_path), model, tools)  # type: ignore[arg-type]
    second = run(reconstructed.chat(session.session_id, "How does that compare?"))
    assert second.claims


def test_comparison_resolution_cannot_drop_a_classified_target(tmp_path: Path) -> None:
    graph = AgentGraph(
        DroppedComparisonTargetModel(),
        FakeTools(SkillRegistry(tmp_path / "skills")),
    )
    classification = TaskClassification(
        task_type="comparison",
        regions=["Karnataka", "Odisha"],
        reason="Cross-region comparison",
    )

    result = run(
        graph.resolve_query(
            {
                "user_query": "How does that compare with Odisha?",
                "task_type": "comparison",
                "classification": classification,
                "messages": [],
            }
        )
    )

    resolved = cast(str, result["resolved_query"])
    assert all(target in resolved.casefold() for target in ("literacy", "karnataka", "odisha"))
    event = cast(list[TraceEvent], result["trace_events"])[-1]
    assert event.details["target_guard_applied"] is True
    assert event.details["missing_targets"] == ["Karnataka"]


def test_ambiguous_followup_and_out_of_scope_are_graceful(tmp_path: Path) -> None:
    service = AgentService(
        settings(tmp_path),
        FakeModel(),
        FakeTools(SkillRegistry(tmp_path / "skills")),  # type: ignore[arg-type]
    )
    session = run(service.create_session())
    clarification = run(service.chat(session.session_id, "How does that compare?"))
    assert not clarification.refusal
    assert "compare" in clarification.answer_markdown

    out_of_scope = run(service.chat(session.session_id, "What was unemployment in France?"))
    assert out_of_scope.refusal
    assert "supplied Census documents" in out_of_scope.answer_markdown

    unsupported = run(service.chat(session.session_id, "What is quantum flux?"))
    assert unsupported.refusal
    assert "citation-safe evidence" in unsupported.answer_markdown


def test_summary_loads_relevant_skill_and_uses_multiple_sections(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "summary.md").write_text(
        "---\nname: summary\ndescription: Summary\ntask_types: [summary]\n---\nCite sections.\n"
    )
    service = AgentService(settings(tmp_path), FakeModel(), FakeTools(SkillRegistry(skills)))  # type: ignore[arg-type]
    session = run(service.create_session())
    response = run(service.chat(session.session_id, "Summarize the Karnataka report"))
    trace = service.get_trace(response.trace_id)

    assert {citation.page_number for citation in response.citations} == {10, 52}
    assert trace is not None
    assert {call.tool_name for call in trace.tool_calls} >= {
        "list_skills",
        "read_skill",
        "collect_summary_evidence",
    }


def test_repair_and_tool_limits_are_hard_bounded(tmp_path: Path) -> None:
    graph = AgentGraph(
        FakeModel(),
        FakeTools(SkillRegistry(tmp_path / "skills")),
        max_tool_calls=1,
    )
    assert graph._validation_route({"retry_count": 0}) == "repair"
    assert graph._validation_route({"retry_count": 1}) == "stop"

    _, record = run(
        graph._recorded_tool(
            {"tool_calls": [object()]},  # type: ignore[list-item]
            "search_documents",
            {},
            lambda: asyncio.sleep(0),
        )
    )
    assert record.status == "limit_exceeded"

    async def timeout() -> object:
        raise TimeoutError

    async def error() -> object:
        raise ValueError("typed failure")

    _, timed_out = run(graph._recorded_tool({}, "search_documents", {}, timeout))
    _, failed = run(graph._recorded_tool({}, "search_documents", {}, error))
    assert timed_out.status == "timeout"
    assert failed.status == "error"
    assert failed.error_type == "ValueError"


def test_internal_arithmetic_is_deterministic() -> None:
    assert AgentTools.calculate(ArithmeticInput(operation="difference", values=[100, 97])) == 3
    assert AgentTools.calculate(ArithmeticInput(operation="sum", values=[1, 2, 3])) == 6


def test_invalid_citation_gets_exactly_one_repair(tmp_path: Path) -> None:
    model = RepairingModel()
    service = AgentService(
        settings(tmp_path),
        model,
        FakeTools(SkillRegistry(tmp_path / "skills")),  # type: ignore[arg-type]
    )
    session = run(service.create_session())
    response = run(service.chat(session.session_id, "What is Karnataka literacy?"))

    assert model.repairs == 1
    assert response.citations


def test_failed_empty_repair_becomes_refusal(tmp_path: Path) -> None:
    model = EmptyRepairingModel()
    service = AgentService(
        settings(tmp_path),
        model,
        FakeTools(SkillRegistry(tmp_path / "skills")),  # type: ignore[arg-type]
    )
    session = run(service.create_session())
    response = run(service.chat(session.session_id, "What is Karnataka literacy?"))
    trace = service.get_trace(response.trace_id)

    assert model.repairs == 1
    assert response.refusal
    assert not response.claims
    assert "citation validation" in response.answer_markdown
    assert trace is not None
    assert trace.run_status == "completed"
    assert trace.answer_status == "refused"
    assert trace.refusal_reason == "citation_validation_failed"
    validations = [event for event in trace.events if event.event == "citation_validation"]
    codes = cast(list[str], validations[-1].details["error_codes"])
    assert "EMPTY_CLAIMS_FOR_ANSWERABLE_TASK" in codes
    assert all("prompt" not in event.details for event in trace.events)


def test_response_invariants_reject_vacuous_answer_and_missing_comparison_side() -> None:
    empty = DraftAnswer(answer_markdown="Factual prose", claims=[], refusal=False)
    base: dict[str, Any] = {
        "task_type": "lookup",
        "evidence_sufficient": True,
        "classification": TaskClassification(task_type="lookup", reason="test"),
    }
    codes = AgentGraph._response_invariant_errors(base, empty, 0)  # type: ignore[arg-type]
    assert "EMPTY_CLAIMS_FOR_ANSWERABLE_TASK" in codes
    assert "ANSWER_WITHOUT_STRUCTURED_CLAIM" in codes

    direct = census_evidence()
    comparison = {
        **base,
        "task_type": "comparison",
        "classification": TaskClassification(
            task_type="comparison", regions=["Karnataka", "Odisha"], reason="test"
        ),
        "selected_evidence": [direct],
        "calculations": [],
    }
    draft = DraftAnswer(
        answer_markdown="Karnataka finding",
        claims=[
            DraftClaim(claim_id="one", text="Karnataka finding", evidence_ids=[direct.chunk_id])
        ],
    )
    codes = AgentGraph._response_invariant_errors(comparison, draft, 0)  # type: ignore[arg-type]
    assert "MISSING_COMPARISON_TARGET" in codes
    assert "DERIVED_CLAIM_MISSING_INPUT_CITATIONS" in codes


def test_final_answer_is_rendered_from_validated_claims(tmp_path: Path) -> None:
    service = AgentService(
        settings(tmp_path),
        FakeModel(),
        FakeTools(SkillRegistry(tmp_path / "skills")),  # type: ignore[arg-type]
    )
    session = run(service.create_session())
    response = run(service.chat(session.session_id, "What is Karnataka literacy?"))
    assert "Citation-safe result" not in response.answer_markdown
    assert response.claims[0].text in response.answer_markdown
    serialized = response.model_dump(by_alias=True)
    serialized_citations = cast(list[dict[str, Any]], serialized["citations"])
    assert serialized_citations[0]["snippet"] == response.citations[0].snippet
    assert serialized_citations[0]["evidence_span"]["evidence_id"] == response.citations[0].chunk_id


def test_trace_detail_sanitizer_omits_secrets_and_hidden_reasoning() -> None:
    details = safe_trace_details(
        result_count=10,
        access_token="secret",
        hidden_prompt="private",
        embedding=[1.0, 2.0],
    )
    assert details == {"result_count": 10}


def test_assessment_timeout_persists_failed_trace_and_preserves_memory(tmp_path: Path) -> None:
    model = TimeoutAfterSuccessModel()
    tools = FakeTools(SkillRegistry(tmp_path / "skills"))
    service = AgentService(settings(tmp_path), model, tools)  # type: ignore[arg-type]
    session = run(service.create_session())
    first = run(service.chat(session.session_id, "What is Karnataka literacy?"))
    model.fail_assessment = True

    with pytest.raises(AgentChatError) as caught:
        run(service.chat(session.session_id, "How does that compare with Odisha?"))
    assert caught.value.error.code == "EVIDENCE_ASSESSMENT_TIMEOUT"
    assert model.cancelled
    failed = service.get_trace(caught.value.run_id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error_code == "EVIDENCE_ASSESSMENT_TIMEOUT"
    terminal = failed.events[-1]
    assert terminal.details["terminal_status"] == "failed"
    assert terminal.details["assessment_counts_by_document"] == {
        "doc-karnataka": 1,
        "doc-odisha": 1,
    }

    model.fail_assessment = False
    retried = run(service.chat(session.session_id, "How does that compare with Odisha?"))
    assert not retried.refusal
    assert model.context_sizes[-1][1] == 2
    assert service.get_trace(first.trace_id) is not None
