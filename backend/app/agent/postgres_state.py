"""Postgres-backed conversation state for horizontally scaled API replicas.

Checkpoints go through LangGraph's `AsyncPostgresSaver`; sessions and display transcripts use the
same schema as the SQLite store. Session exclusivity is a Postgres session-level advisory lock held
on a dedicated connection for the whole turn. If a replica dies mid-turn, Postgres ends its
connection and releases the lock, so no session can be wedged by a crashed process.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4
from weakref import WeakValueDictionary

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from backend.app.agent.models import (
    AgentErrorResponse,
    AgentResponse,
    ResearchReport,
    SessionRecord,
    SessionSummary,
    TranscriptEntry,
)
from backend.app.agent.persistence import _clean_title, _default_title, validate_session_id

# Two-key advisory locks keep this application's lock space apart from anything else sharing the
# database. A hash collision between two live sessions only serializes them; it never lets two
# turns run on one session.
_SESSION_LOCK_NAMESPACE = 0x43454E53  # "CENS"
_MIGRATION_LOCK_NAMESPACE = 0x43454E4D  # "CENM"

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS app_sessions ("
    "session_id TEXT PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL, "
    "updated_at TIMESTAMPTZ NOT NULL, checkpoint_id TEXT, title TEXT, parent_session_id TEXT)",
    "CREATE INDEX IF NOT EXISTS app_sessions_parent ON app_sessions (parent_session_id)",
    "CREATE INDEX IF NOT EXISTS app_sessions_updated ON app_sessions (updated_at DESC)",
    "CREATE TABLE IF NOT EXISTS app_messages ("
    "seq BIGINT GENERATED ALWAYS AS IDENTITY, message_id TEXT PRIMARY KEY, "
    "session_id TEXT NOT NULL REFERENCES app_sessions (session_id) ON DELETE CASCADE, "
    "role TEXT NOT NULL, payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL)",
    "CREATE INDEX IF NOT EXISTS app_messages_session ON app_messages (session_id, created_at, seq)",
)


def _session_lock_key(session_id: str) -> int:
    value = int(validate_session_id(session_id)[:8], 16)
    return value - (1 << 32) if value >= 1 << 31 else value


class PostgresSessionStore:
    def __init__(self, backend: "PostgresStateBackend") -> None:
        self._backend = backend

    async def initialize(self) -> None:
        await self._backend.initialize()

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[AsyncConnection[DictRow]]:
        pool = await self._backend.data_pool()
        async with pool.connection() as connection:
            yield connection

    async def create(self, parent_session_id: str | None = None) -> SessionRecord:
        if parent_session_id is not None:
            validate_session_id(parent_session_id)
        now = datetime.now(UTC)
        record = SessionRecord(session_id=uuid4().hex, created_at=now, updated_at=now)
        async with self._connection() as connection:
            await connection.execute(
                "INSERT INTO app_sessions "
                "(session_id, created_at, updated_at, checkpoint_id, parent_session_id) "
                "VALUES (%s, %s, %s, NULL, %s)",
                (record.session_id, now, now, parent_session_id),
            )
        return record

    async def children(self, session_id: str) -> list[str]:
        validate_session_id(session_id)
        async with self._connection() as connection:
            cursor = await connection.execute(
                "SELECT session_id FROM app_sessions WHERE parent_session_id = %s", (session_id,)
            )
            return [str(row["session_id"]) for row in await cursor.fetchall()]

    async def get(self, session_id: str) -> SessionRecord | None:
        validate_session_id(session_id)
        async with self._connection() as connection:
            cursor = await connection.execute(
                "SELECT session_id, created_at, updated_at, title FROM app_sessions "
                "WHERE session_id = %s",
                (session_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return SessionRecord(
            session_id=row["session_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            title=row["title"],
        )

    async def list_recent(self, limit: int = 100) -> list[SessionSummary]:
        async with self._connection() as connection:
            cursor = await connection.execute(
                "SELECT s.session_id, s.created_at, s.updated_at, s.title, "
                "COUNT(m.message_id) AS message_count "
                "FROM app_sessions s JOIN app_messages m ON m.session_id = s.session_id "
                "WHERE s.parent_session_id IS NULL "
                "GROUP BY s.session_id ORDER BY s.updated_at DESC LIMIT %s",
                (limit,),
            )
            rows = await cursor.fetchall()
        return [
            SessionSummary(
                session_id=row["session_id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                title=row["title"],
                message_count=int(row["message_count"]),
            )
            for row in rows
        ]

    async def set_title(self, session_id: str, title: str) -> None:
        validate_session_id(session_id)
        async with self._connection() as connection:
            await connection.execute(
                "UPDATE app_sessions SET title = %s WHERE session_id = %s",
                (_clean_title(title), session_id),
            )

    async def append_user_message(
        self, session_id: str, content: str, *, mode: Literal["chat", "research"] = "chat"
    ) -> TranscriptEntry:
        entry = TranscriptEntry(
            message_id=uuid4().hex,
            role="user",
            created_at=datetime.now(UTC),
            content=content,
            mode=mode,
        )
        await self._append(session_id, entry, _default_title(content))
        return entry

    async def append_assistant_message(
        self,
        session_id: str,
        *,
        response: AgentResponse | None = None,
        error: AgentErrorResponse | None = None,
        report: ResearchReport | None = None,
    ) -> TranscriptEntry:
        if sum(value is not None for value in (response, error, report)) != 1:
            raise ValueError("An assistant message needs exactly one of response, error, report")
        entry = TranscriptEntry(
            message_id=uuid4().hex,
            role="assistant",
            created_at=datetime.now(UTC),
            response=response,
            error=error,
            report=report,
        )
        await self._append(session_id, entry, None)
        return entry

    async def _append(
        self, session_id: str, entry: TranscriptEntry, default_title: str | None
    ) -> None:
        validate_session_id(session_id)
        # Stored by field name (not the `answer` wire alias) so it validates back unchanged.
        payload = entry.model_dump(mode="json", exclude={"message_id", "created_at"})
        async with self._connection() as connection, connection.transaction():
            await connection.execute(
                "INSERT INTO app_messages (message_id, session_id, role, payload, created_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (entry.message_id, session_id, entry.role, Jsonb(payload), entry.created_at),
            )
            await connection.execute(
                "UPDATE app_sessions SET updated_at = %s, title = COALESCE(title, %s) "
                "WHERE session_id = %s",
                (entry.created_at, default_title, session_id),
            )

    async def messages(self, session_id: str) -> list[TranscriptEntry]:
        validate_session_id(session_id)
        async with self._connection() as connection:
            cursor = await connection.execute(
                "SELECT message_id, payload, created_at FROM app_messages "
                "WHERE session_id = %s ORDER BY created_at, seq",
                (session_id,),
            )
            rows = await cursor.fetchall()
        entries: list[TranscriptEntry] = []
        for row in rows:
            raw = dict(row["payload"])
            raw.update(message_id=row["message_id"], created_at=row["created_at"])
            entries.append(TranscriptEntry.model_validate(raw))
        return entries

    async def delete(self, session_id: str) -> None:
        validate_session_id(session_id)
        async with self._connection() as connection:
            # Transcript rows go with the session through ON DELETE CASCADE.
            await connection.execute(
                "DELETE FROM app_sessions WHERE session_id = %s", (session_id,)
            )

    async def touch(self, session_id: str) -> None:
        async with self._connection() as connection:
            await connection.execute(
                "UPDATE app_sessions SET updated_at = %s WHERE session_id = %s",
                (datetime.now(UTC), session_id),
            )

    async def get_checkpoint_id(self, session_id: str) -> str | None:
        async with self._connection() as connection:
            cursor = await connection.execute(
                "SELECT checkpoint_id FROM app_sessions WHERE session_id = %s", (session_id,)
            )
            row = await cursor.fetchone()
        return None if row is None or row["checkpoint_id"] is None else str(row["checkpoint_id"])

    async def set_checkpoint_id(self, session_id: str, checkpoint_id: str) -> None:
        async with self._connection() as connection:
            await connection.execute(
                "UPDATE app_sessions SET checkpoint_id = %s WHERE session_id = %s",
                (checkpoint_id, session_id),
            )

    async def stale_session_ids(self, cutoff: datetime) -> list[str]:
        async with self._connection() as connection:
            cursor = await connection.execute(
                "SELECT session_id FROM app_sessions WHERE updated_at < %s ORDER BY updated_at",
                (cutoff,),
            )
            return [str(row["session_id"]) for row in await cursor.fetchall()]


class _Pools:
    """Connection pools bound to the event loop that opened them."""

    def __init__(self, url: str, *, lock_pool_size: int) -> None:
        self.loop = asyncio.get_running_loop()
        # AsyncPostgresSaver requires autocommit, dict rows, and no prepared statements.
        self.data: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
            url,
            connection_class=AsyncConnection[DictRow],
            min_size=1,
            max_size=10,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            open=False,
        )
        # Each in-flight turn holds one lock connection for its whole duration, so locks get
        # their own pool and can never starve the checkpoint and transcript queries.
        self.locks = AsyncConnectionPool(
            url, min_size=0, max_size=lock_pool_size, kwargs={"autocommit": True}, open=False
        )
        self.saver = AsyncPostgresSaver(conn=self.data)

    async def open(self) -> None:
        await self.data.open(wait=True, timeout=30)
        await self.locks.open()

    async def close(self) -> None:
        await self.locks.close()
        await self.data.close()


class PostgresStateBackend:
    def __init__(
        self, url: str, *, lock_wait_seconds: float = 240.0, lock_pool_size: int = 32
    ) -> None:
        self._url = url
        self._lock_wait_seconds = lock_wait_seconds
        self._lock_pool_size = lock_pool_size
        self._pools: _Pools | None = None
        self._opening = asyncio.Lock()
        self._local_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
        self.sessions = PostgresSessionStore(self)

    async def _ensure_pools(self) -> _Pools:
        pools = self._pools
        # Production serves from one event loop; a new loop (for example a test harness that
        # starts one per call) gets its own pools because psycopg connections are loop-bound.
        if pools is not None and pools.loop is asyncio.get_running_loop():
            return pools
        async with self._opening:
            if self._pools is None or self._pools.loop is not asyncio.get_running_loop():
                fresh = _Pools(self._url, lock_pool_size=self._lock_pool_size)
                await fresh.open()
                await self._migrate(fresh)
                self._pools = fresh
            return self._pools

    @staticmethod
    async def _migrate(pools: _Pools) -> None:
        # Replicas that start together must not run the same migrations concurrently. Waiters
        # poll instead of blocking in pg_advisory_lock: LangGraph's migrations include CREATE
        # INDEX CONCURRENTLY, which waits for every open transaction, and a blocked lock call is
        # one. That cycle runs through this process, so Postgres cannot detect the deadlock.
        async with pools.locks.connection() as connection:
            while True:
                cursor = await connection.execute(
                    "SELECT pg_try_advisory_lock(%s::int4, 0)", (_MIGRATION_LOCK_NAMESPACE,)
                )
                row = await cursor.fetchone()
                if row is not None and row[0]:
                    break
                await asyncio.sleep(0.2)
            try:
                await pools.saver.setup()
                async with pools.data.connection() as data:
                    for statement in _SCHEMA:
                        await data.execute(statement)
            finally:
                await connection.execute(
                    "SELECT pg_advisory_unlock(%s::int4, 0)", (_MIGRATION_LOCK_NAMESPACE,)
                )

    async def data_pool(self) -> AsyncConnectionPool[AsyncConnection[DictRow]]:
        return (await self._ensure_pools()).data

    async def initialize(self) -> None:
        await self._ensure_pools()

    @asynccontextmanager
    async def checkpointer(self) -> AsyncIterator[AsyncPostgresSaver]:
        yield (await self._ensure_pools()).saver

    @asynccontextmanager
    async def session_lock(self, session_id: str) -> AsyncIterator[None]:
        key = _session_lock_key(session_id)
        # Turns on one replica queue locally first, so a busy session holds one connection.
        local = self._local_locks.setdefault(session_id, asyncio.Lock())
        async with local:
            pools = await self._ensure_pools()
            async with pools.locks.connection(timeout=self._lock_wait_seconds) as connection:
                try:
                    await connection.execute(
                        "SELECT pg_advisory_lock(%s::int4, %s::int4)",
                        (_SESSION_LOCK_NAMESPACE, key),
                    )
                except BaseException:
                    # Cancelled or failed while waiting: the lock may have been granted anyway,
                    # and closing the connection is the only way to be sure it is released.
                    await connection.close()
                    raise
                try:
                    yield
                finally:
                    try:
                        await connection.execute(
                            "SELECT pg_advisory_unlock(%s::int4, %s::int4)",
                            (_SESSION_LOCK_NAMESPACE, key),
                        )
                    except BaseException:
                        await connection.close()
                        raise

    async def close(self) -> None:
        pools, self._pools = self._pools, None
        if pools is not None and pools.loop is asyncio.get_running_loop():
            await pools.close()
