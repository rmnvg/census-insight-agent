import asyncio
from typing import Any, cast

import pytest

from backend.app.agent.checkpoint import checkpoint_node
from backend.app.agent.errors import AgentOperationalError
from backend.app.agent.models import AgentState
from backend.app.agent.provider import ProviderCallTimeout, ProviderOperationalError


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


async def _raises_timeout(state: AgentState) -> dict[str, object]:
    del state
    raise ProviderCallTimeout(elapsed_seconds=61.0, timeout_seconds=60.0, retry_count=0)


async def _raises_operational(state: AgentState) -> dict[str, object]:
    del state
    raise ProviderOperationalError(
        "MODEL_UNAVAILABLE", retryable=True, provider_exception="ServiceUnavailable"
    )


def test_provider_timeout_from_any_node_becomes_typed_operational_error() -> None:
    # Discovered live: classify_task/resolve_query/synthesize/repair call the model with no
    # local error handling, unlike prepare_artifact. A raw provider exception from any of them
    # previously escaped as an unhandled HTTP 500 instead of the sanitized typed-error contract
    # every other failure in this system uses.
    node = checkpoint_node(_raises_timeout)
    with pytest.raises(AgentOperationalError) as excinfo:
        run(node(cast(AgentState, {})))
    assert excinfo.value.code == "MODEL_TIMEOUT"
    assert excinfo.value.node == "_raises_timeout"
    assert excinfo.value.retryable is True
    assert excinfo.value.retry_count == 0


def test_provider_operational_error_from_any_node_becomes_typed_operational_error() -> None:
    node = checkpoint_node(_raises_operational)
    with pytest.raises(AgentOperationalError) as excinfo:
        run(node(cast(AgentState, {})))
    assert excinfo.value.code == "MODEL_UNAVAILABLE"
    assert excinfo.value.node == "_raises_operational"
    assert excinfo.value.retryable is True
    assert excinfo.value.diagnostics["provider_exception"] == "ServiceUnavailable"
