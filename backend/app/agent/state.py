"""Where conversation state lives, and how a turn gets exclusive use of its session.

Two interchangeable backends hold LangGraph checkpoints, the session list, and display
transcripts:

- `SqliteStateBackend` (default): one SQLite file in the workspace plus in-process session
  locks. Correct for exactly one API process.
- `PostgresStateBackend` (`STATE_DATABASE_URL`): everything in Postgres, and the per-session lock
  is a Postgres advisory lock, so any number of API replicas can serve any session and a turn on
  one replica waits for an in-flight turn on another.
"""

import asyncio
import sqlite3
import threading
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from weakref import WeakValueDictionary

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.app.agent.models import (
    AgentErrorResponse,
    AgentResponse,
    ResearchReport,
    SessionRecord,
    SessionSummary,
    TranscriptEntry,
)
from backend.app.agent.persistence import SessionStore
from backend.app.config import Settings


class SessionRepository(Protocol):
    async def initialize(self) -> None: ...

    async def create(self, parent_session_id: str | None = None) -> SessionRecord: ...

    async def children(self, session_id: str) -> list[str]: ...

    async def get(self, session_id: str) -> SessionRecord | None: ...

    async def list_recent(self, limit: int = 100) -> list[SessionSummary]: ...

    async def set_title(self, session_id: str, title: str) -> None: ...

    async def append_user_message(
        self, session_id: str, content: str, *, mode: Literal["chat", "research"] = "chat"
    ) -> TranscriptEntry: ...

    async def append_assistant_message(
        self,
        session_id: str,
        *,
        response: AgentResponse | None = None,
        error: AgentErrorResponse | None = None,
        report: ResearchReport | None = None,
    ) -> TranscriptEntry: ...

    async def messages(self, session_id: str) -> list[TranscriptEntry]: ...

    async def delete(self, session_id: str) -> None: ...

    async def touch(self, session_id: str) -> None: ...

    async def get_checkpoint_id(self, session_id: str) -> str | None: ...

    async def set_checkpoint_id(self, session_id: str, checkpoint_id: str) -> None: ...

    async def stale_session_ids(self, cutoff: datetime) -> list[str]: ...


class StateBackend(Protocol):
    @property
    def sessions(self) -> SessionRepository: ...

    async def initialize(self) -> None: ...

    def checkpointer(self) -> AbstractAsyncContextManager[Any]: ...

    def session_lock(self, session_id: str) -> AbstractAsyncContextManager[None]: ...

    async def close(self) -> None: ...


# Serializes schema setup across every backend and thread in the process; setup runs once per
# backend, so this is never contended during normal turns.
_SQLITE_SETUP = threading.Lock()


class SqliteStateBackend:
    def __init__(self, workspace_root: Path) -> None:
        self.sessions = SessionStore(workspace_root)
        # Weak references release idle session locks without an ever-growing registry.
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
        self._ready = False

    async def initialize(self) -> None:
        if not self._ready:
            await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        with _SQLITE_SETUP:
            if self._ready:
                return
            self.sessions.root.mkdir(parents=True, exist_ok=True)
            self.sessions._initialize_sync()
            # The checkpointer's own setup (a WAL journal-mode switch plus table DDL) runs once
            # here instead of lazily on every per-turn connection, where concurrent first turns
            # (research sections) collided with `database is locked`.
            connection = sqlite3.connect(self.sessions.database)
            try:
                SqliteSaver(connection).setup()
            finally:
                connection.close()
            self._ready = True

    @asynccontextmanager
    async def checkpointer(self) -> AsyncIterator[AsyncSqliteSaver]:
        await self.initialize()
        async with AsyncSqliteSaver.from_conn_string(str(self.sessions.database)) as saver:
            saver.is_setup = True
            yield saver

    @asynccontextmanager
    async def session_lock(self, session_id: str) -> AsyncIterator[None]:
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            yield

    async def close(self) -> None:
        return None


def build_state_backend(settings: Settings) -> StateBackend:
    if settings.state_database_url:
        # Imported lazily so the default single-host mode never loads the Postgres driver.
        from backend.app.agent.postgres_state import PostgresStateBackend

        return PostgresStateBackend(
            settings.state_database_url,
            lock_wait_seconds=settings.agent_request_timeout_seconds,
        )
    return SqliteStateBackend(settings.workspace_root)
