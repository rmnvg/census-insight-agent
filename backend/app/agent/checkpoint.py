from collections.abc import Awaitable, Callable
from typing import Any, cast

from pydantic import BaseModel

from backend.app.agent.errors import AgentOperationalError
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
from backend.app.agent.provider import ProviderCallTimeout, ProviderOperationalError
from backend.app.execution.contracts import (
    ArtifactDataset,
    ArtifactDescriptor,
    ExecutionRequest,
    ExecutionResult,
    SourceRecord,
)
from backend.app.retrieval.models import RetrievedEvidence

_PROVIDER_ERROR_MESSAGES = {
    "MODEL_SCHEMA_REJECTED": "The model could not accept the response schema for this step.",
    "MODEL_RATE_LIMITED": "The model is temporarily unavailable because it is busy.",
    "MODEL_AUTHENTICATION_FAILED": "Could not authenticate with the configured model.",
    "MODEL_OUTPUT_INVALID": "The model returned an invalid structured response.",
    "MODEL_UNAVAILABLE": "The model is temporarily unavailable.",
}

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
        node_name = getattr(node, "__name__", "agent_node")
        try:
            updates = await node(hydrate_agent_state(state))
        except ProviderCallTimeout as error:
            # A safety net for nodes that call the model directly with no local error
            # handling (e.g. classify_task, resolve_query, synthesize, repair): without this,
            # the raw provider exception previously escaped as an unhandled HTTP 500 instead
            # of the sanitized typed-error contract every other failure uses. prepare_artifact
            # already converts its own provider errors before they would reach this wrapper,
            # so this is not reached for artifact preparation.
            raise AgentOperationalError(
                code="MODEL_TIMEOUT",
                node=node_name,
                message="The model call for this step timed out before completing.",
                retryable=True,
                elapsed_seconds=error.elapsed_seconds,
                configured_timeout_seconds=error.timeout_seconds,
                retry_count=error.retry_count,
                diagnostics={"failure_stage": "model_invocation"},
            ) from error
        except ProviderOperationalError as error:
            raise AgentOperationalError(
                code=error.code,
                node=node_name,
                message=_PROVIDER_ERROR_MESSAGES[error.code],
                retryable=error.retryable,
                elapsed_seconds=None,
                configured_timeout_seconds=None,
                retry_count=0,
                diagnostics={
                    "failure_stage": "model_invocation",
                    "provider_exception": error.provider_exception,
                    "provider_status_code": error.status_code,
                },
            ) from error
        updates.setdefault("checkpoint_schema_version", CHECKPOINT_SCHEMA_VERSION)
        return cast(dict[str, object], checkpoint_safe(updates))

    return wrapped
