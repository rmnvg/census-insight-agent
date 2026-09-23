import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.app.agent.service import AgentService
from backend.app.agent.skills import SkillRegistry
from backend.tests.test_agent_service import FakeModel, FakeTools, run, settings
from scripts.prune_checkpoints import prune


def _backdate(database: Path, session_id: str, when: datetime) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE app_sessions SET updated_at = ? WHERE session_id = ?",
            (when.isoformat(), session_id),
        )


def _thread_has_checkpoints(database: Path, session_id: str) -> bool:
    async def _check() -> bool:
        async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
            async for _ in saver.alist({"configurable": {"thread_id": session_id}}, limit=1):
                return True
        return False

    return asyncio.run(_check())


def test_dry_run_lists_stale_sessions_without_deleting_anything(tmp_path: Path) -> None:
    service = AgentService(
        settings(tmp_path),
        FakeModel(),
        FakeTools(SkillRegistry(tmp_path / "skills")),  # type: ignore[arg-type]
    )
    stale_session = run(service.create_session())
    fresh_session = run(service.create_session())
    run(service.chat(stale_session.session_id, "What was Karnataka's literacy rate in 2011?"))
    run(service.chat(fresh_session.session_id, "What was Karnataka's literacy rate in 2011?"))

    database = service.settings.workspace_root / "checkpoints.sqlite"
    _backdate(database, stale_session.session_id, datetime.now(UTC) - timedelta(days=45))

    listed = run(prune(service.settings.workspace_root, older_than_days=30, allow_delete=False))

    assert listed == [stale_session.session_id]
    assert _thread_has_checkpoints(database, stale_session.session_id)
    assert (service.settings.workspace_root / "sessions" / stale_session.session_id).exists()


def test_allow_delete_removes_checkpoints_session_row_and_directory(tmp_path: Path) -> None:
    service = AgentService(
        settings(tmp_path),
        FakeModel(),
        FakeTools(SkillRegistry(tmp_path / "skills")),  # type: ignore[arg-type]
    )
    stale_session = run(service.create_session())
    fresh_session = run(service.create_session())
    run(service.chat(stale_session.session_id, "What was Karnataka's literacy rate in 2011?"))
    run(service.chat(fresh_session.session_id, "What was Karnataka's literacy rate in 2011?"))

    database = service.settings.workspace_root / "checkpoints.sqlite"
    _backdate(database, stale_session.session_id, datetime.now(UTC) - timedelta(days=45))

    pruned = run(prune(service.settings.workspace_root, older_than_days=30, allow_delete=True))

    assert pruned == [stale_session.session_id]
    assert not _thread_has_checkpoints(database, stale_session.session_id)
    assert not (service.settings.workspace_root / "sessions" / stale_session.session_id).exists()
    with sqlite3.connect(database) as connection:
        remaining = {
            row[0] for row in connection.execute("SELECT session_id FROM app_sessions").fetchall()
        }
    assert remaining == {fresh_session.session_id}
    # The fresh session's own checkpoints and session directory must be untouched.
    assert _thread_has_checkpoints(database, fresh_session.session_id)
    assert (service.settings.workspace_root / "sessions" / fresh_session.session_id).exists()


def test_missing_checkpoint_database_returns_empty(tmp_path: Path) -> None:
    assert run(prune(tmp_path / "no-such-workspace", older_than_days=30, allow_delete=True)) == []
