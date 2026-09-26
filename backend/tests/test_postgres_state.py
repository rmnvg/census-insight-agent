"""Postgres state mode against a real server; skipped unless TEST_POSTGRES_URL is set.

Two `AgentService` instances over one database stand in for two API replicas: they share nothing
in process, exactly like separate containers.
"""

import asyncio
import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from backend.app.agent.postgres_state import (
    _SESSION_LOCK_NAMESPACE,
    PostgresStateBackend,
    _session_lock_key,
)
from backend.app.agent.service import AgentService
from backend.app.agent.skills import SkillRegistry
from backend.app.config import Settings
from backend.tests.test_agent_service import FakeModel, FakeTools, settings
from scripts.prune_checkpoints import prune

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES_URL"),
    reason="TEST_POSTGRES_URL is not set; Postgres state tests need a real server",
)


@pytest.fixture
def state_url() -> Iterator[str]:
    base = os.environ["TEST_POSTGRES_URL"]
    name = f"census_state_{uuid4().hex[:12]}"
    with psycopg.connect(base, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    yield urlsplit(base)._replace(path=f"/{name}").geturl()
    with psycopg.connect(base, autocommit=True) as admin:
        admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


def replica(tmp_path: Path, url: str, model: FakeModel | None = None) -> AgentService:
    # Validated construction: an unknown key must fail loudly, never fall back to SQLite.
    configured = Settings.model_validate(
        {**settings(tmp_path).model_dump(), "state_database_url": url}
    )
    tools = FakeTools(SkillRegistry(tmp_path / "skills"))
    service = AgentService(configured, model or FakeModel(), cast(Any, tools))
    assert isinstance(service.state, PostgresStateBackend)
    return service


def test_memory_written_by_one_replica_is_used_by_another(tmp_path: Path, state_url: str) -> None:
    async def scenario() -> None:
        model = FakeModel()
        first, second = (
            replica(tmp_path, state_url, model),
            replica(tmp_path, state_url, model),
        )
        try:
            session = await first.create_session()
            answer = await first.chat(session.session_id, "What is Karnataka literacy?")
            assert answer.claims

            follow_up = await second.chat(session.session_id, "How does that compare?")
            assert {citation.document_id for citation in follow_up.citations} == {
                "doc-karnataka",
                "doc-odisha",
            }
            assert dict(model.context_sizes)["How does that compare?"] > 0
            transcript = await first.get_transcript(session.session_id)
            assert [entry.role for entry in transcript.messages] == [
                "user",
                "assistant",
                "user",
                "assistant",
            ]
            listed = await second.list_sessions()
            assert [(item.session_id, item.message_count) for item in listed] == [
                (session.session_id, 4)
            ]
        finally:
            await first.close()
            await second.close()

    asyncio.run(scenario())


def test_overlapping_turns_on_two_replicas_are_serialized(tmp_path: Path, state_url: str) -> None:
    async def scenario() -> None:
        model = FakeModel()
        first, second = (
            replica(tmp_path, state_url, model),
            replica(tmp_path, state_url, model),
        )
        try:
            session = await first.create_session()
            # Either replica may win the lock. Without it, both turns would extend the same
            # stale checkpoint and only one validated turn would survive.
            one, two = await asyncio.gather(
                first.chat(session.session_id, "What is Karnataka literacy?"),
                second.chat(session.session_id, "What is Karnataka literacy?"),
            )
            assert one.claims and two.claims
            status = await second.get_context_status(session.session_id)
            assert status.validated_turn_count == 2
        finally:
            await first.close()
            await second.close()

    asyncio.run(scenario())


def test_session_lock_blocks_across_replicas_and_survives_a_crashed_holder(
    state_url: str,
) -> None:
    async def scenario() -> None:
        a = PostgresStateBackend(state_url)
        b = PostgresStateBackend(state_url)
        session_id = uuid4().hex
        try:
            await a.initialize()
            order: list[str] = []
            held = asyncio.Event()

            async def hold() -> None:
                async with a.session_lock(session_id):
                    order.append("a-start")
                    held.set()
                    await asyncio.sleep(0.5)
                    order.append("a-end")

            async def wait_then_enter() -> None:
                await held.wait()
                async with b.session_lock(session_id):
                    order.append("b")

            await asyncio.gather(hold(), wait_then_enter())
            assert order == ["a-start", "a-end", "b"]

            # A replica that dies mid-turn never unlocks; its connection closing must.
            crashed = await psycopg.AsyncConnection.connect(state_url, autocommit=True)
            await crashed.execute(
                "SELECT pg_advisory_lock(%s::int4, %s::int4)",
                (_SESSION_LOCK_NAMESPACE, _session_lock_key(session_id)),
            )
            waiter = asyncio.create_task(_enter(b, session_id))
            await asyncio.sleep(0.3)
            assert not waiter.done()
            await crashed.close()
            await asyncio.wait_for(waiter, timeout=5)
        finally:
            await a.close()
            await b.close()

    asyncio.run(scenario())


async def _enter(backend: PostgresStateBackend, session_id: str) -> None:
    async with backend.session_lock(session_id):
        return None


def test_replicas_starting_together_migrate_once(state_url: str) -> None:
    async def scenario() -> None:
        backends = [PostgresStateBackend(state_url) for _ in range(4)]
        try:
            await asyncio.gather(*(backend.initialize() for backend in backends))
            record = await backends[0].sessions.create()
            assert await backends[3].sessions.get(record.session_id) is not None
        finally:
            await asyncio.gather(*(backend.close() for backend in backends))

    asyncio.run(scenario())


def test_delete_removes_checkpoints_transcript_and_children(tmp_path: Path, state_url: str) -> None:
    async def scenario() -> None:
        service = replica(tmp_path, state_url)
        try:
            session = await service.create_session()
            child = await service.sessions.create(parent_session_id=session.session_id)
            await service.chat(session.session_id, "What is Karnataka literacy?")
            await service.chat(child.session_id, "What is Karnataka literacy?")
            await service.delete_session(session.session_id)
            assert await service.get_session(session.session_id) is None
            assert await service.get_session(child.session_id) is None
        finally:
            await service.close()
        with psycopg.connect(state_url) as connection:
            counts = connection.execute(
                "SELECT (SELECT COUNT(*) FROM checkpoints WHERE thread_id = ANY(%s)), "
                "(SELECT COUNT(*) FROM app_messages)",
                ([session.session_id, child.session_id],),
            ).fetchone()
        assert counts == (0, 0)

    asyncio.run(scenario())


def test_prune_removes_only_stale_sessions(tmp_path: Path, state_url: str) -> None:
    async def scenario() -> tuple[str, str]:
        service = replica(tmp_path, state_url)
        try:
            stale = await service.create_session()
            fresh = await service.create_session()
            await service.chat(stale.session_id, "What is Karnataka literacy?")
            await service.chat(fresh.session_id, "What is Karnataka literacy?")
        finally:
            await service.close()
        return stale.session_id, fresh.session_id

    stale_id, fresh_id = asyncio.run(scenario())
    with psycopg.connect(state_url, autocommit=True) as connection:
        connection.execute(
            "UPDATE app_sessions SET updated_at = %s WHERE session_id = %s",
            (datetime.now(UTC) - timedelta(days=45), stale_id),
        )
    workspace = tmp_path / "workspace"
    listed = asyncio.run(
        prune(workspace, older_than_days=30, allow_delete=False, state_url=state_url)
    )
    assert listed == [stale_id]
    started = time.monotonic()
    pruned = asyncio.run(
        prune(workspace, older_than_days=30, allow_delete=True, state_url=state_url)
    )
    assert pruned == [stale_id] and time.monotonic() - started < 30
    assert not (workspace / "sessions" / stale_id).exists()
    assert (workspace / "sessions" / fresh_id).exists()
    with psycopg.connect(state_url) as connection:
        remaining = connection.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
        sessions = connection.execute("SELECT session_id FROM app_sessions").fetchall()
    assert remaining == [(fresh_id,)]
    assert sessions == [(fresh_id,)]
