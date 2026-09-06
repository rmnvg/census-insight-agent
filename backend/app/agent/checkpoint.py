from collections.abc import Awaitable, Callable
from typing import Any, cast

from pydantic import BaseModel

from backend.app.agent.models import (
    AgentPlan,
    AgentResponse,
    AgentState,
    AnswerClaim,
    ArtifactDataRequirement,
    ArtifactResult,
    CalculationResult,
    Citation,
    DraftAnswer,
    EvidenceAssessment,
    SupportAssessment,
    TaskClassification,
    ToolCallRecord,
    TraceEvent,
    ValidatedClaimRecord,
    ValidatedClaimTurn,
)
from backend.app.execution.contracts import (
    ArtifactDataset,
    ArtifactDescriptor,
    ExecutionRequest,
    ExecutionResult,
    SourceRecord,
)
from backend.app.retrieval.models import RetrievedEvidence

CHECKPOINT_SCHEMA_VERSION = 4


def checkpoint_safe(value: Any) -> Any:
    """Recursively convert application-owned values to MessagePack-safe JSON primitives."""
    if isinstance(value, BaseModel):
        return checkpoint_safe(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): checkpoint_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [checkpoint_safe(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return value


def _model(value: Any, model: type[BaseModel]) -> Any:
    return value if isinstance(value, model) else model.model_validate(value)


def _models(value: Any, model: type[BaseModel]) -> list[Any]:
    return [_model(item, model) for item in value or []]


def hydrate_agent_state(raw: AgentState | dict[str, Any]) -> AgentState:
    """Reconstruct validated runtime models from primitive checkpoint values."""
    state = dict(raw)
    scalar_models: dict[str, type[BaseModel]] = {
        "classification": TaskClassification,
        "artifact_requirement": ArtifactDataRequirement,
        "plan": AgentPlan,
        "evidence_assessment": EvidenceAssessment,
        "support_assessment": SupportAssessment,
        "draft_answer": DraftAnswer,
        "final_response": AgentResponse,
        "artifact_dataset": ArtifactDataset,
        "ranking_winner": SourceRecord,
        "execution_request": ExecutionRequest,
        "execution_result": ExecutionResult,
    }
    list_models: dict[str, type[BaseModel]] = {
        "retrieved_evidence": RetrievedEvidence,
        "selected_evidence": RetrievedEvidence,
        "answer_claims": AnswerClaim,
        "citations": Citation,
        "tool_calls": ToolCallRecord,
        "trace_events": TraceEvent,
        "calculations": CalculationResult,
        "artifact_descriptors": ArtifactDescriptor,
        "validated_claim_history": ValidatedClaimTurn,
        "source_support_claims": ValidatedClaimRecord,
    }
    for key, model in scalar_models.items():
        value = state.get(key)
        if value is not None:
            state[key] = _model(value, model)
    for key, model in list_models.items():
        if key in state:
            state[key] = _models(state[key], model)
    if "artifacts" in state:
        state["artifacts"] = [
            _model(
                item,
                ArtifactDescriptor
                if isinstance(item, dict) and "artifact_id" in item
                else ArtifactResult,
            )
            for item in state["artifacts"]
        ]
    return cast(AgentState, state)


def checkpoint_node(
    node: Callable[[AgentState], Awaitable[dict[str, object]]],
) -> Any:
    """Hydrate on node entry and serialize application values on node exit."""

    async def wrapped(state: AgentState) -> dict[str, object]:
        updates = await node(hydrate_agent_state(state))
        updates.setdefault("checkpoint_schema_version", CHECKPOINT_SCHEMA_VERSION)
        return cast(dict[str, object], checkpoint_safe(updates))

    return wrapped
