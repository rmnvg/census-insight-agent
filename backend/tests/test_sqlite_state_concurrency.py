"""Concurrent first turns against a fresh SQLite state file, as research sections produce.

Regression: every turn opened its own checkpoint connection, and each new connection ran the
checkpointer's setup script (a WAL journal-mode switch plus table DDL). Sections starting
together collided and one failed with `sqlite3.OperationalError: database is locked`.
"""

import asyncio
from pathlib import Path
from typing import Any, cast

from backend.app.agent.service import AgentService
from backend.app.agent.skills import SkillRegistry
from backend.tests.test_agent_service import FakeModel, FakeTools, settings

CONCURRENT_TURNS = 8


def test_simultaneous_first_turns_on_a_fresh_database_all_succeed(tmp_path: Path) -> None:
    async def scenario() -> None:
        service = AgentService(
            settings(tmp_path),
            FakeModel(),
            cast(Any, FakeTools(SkillRegistry(tmp_path / "skills"))),
        )
        sessions = await asyncio.gather(
            *(service.create_session() for _ in range(CONCURRENT_TURNS))
        )
        responses = await asyncio.gather(
            *(
                service.chat(session.session_id, "What is Karnataka literacy?")
                for session in sessions
            ),
            return_exceptions=True,
        )
        failures = [item for item in responses if isinstance(item, BaseException)]
        assert not failures, failures
        statuses = await asyncio.gather(
            *(service.get_context_status(session.session_id) for session in sessions)
        )
        assert all(status.validated_turn_count == 1 for status in statuses)

    asyncio.run(scenario())
