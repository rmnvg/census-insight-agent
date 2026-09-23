#!/usr/bin/env python3
"""Prune stale session checkpoints, traces, and artifacts from `workspace/`.

Every graph superstep for every turn writes a full LangGraph state snapshot to
`workspace/checkpoints.sqlite`, and nothing else in this codebase ever deletes an old one —
`AsyncSqliteSaver` (and this application's own trace/artifact stores) are append-only by design.
Locally this file can grow to tens of megabytes across a handful of real sessions; nothing here is
a correctness bug, but it is unbounded local disk growth with no built-in cleanup.

This script finds sessions inactive longer than `--older-than-days` (using the
`app_sessions.updated_at` column `backend/app/agent/persistence.SessionStore` already maintains),
and removes:
  - their LangGraph checkpoints, via `AsyncSqliteSaver.adelete_thread`
  - their `app_sessions` row
  - their `workspace/sessions/<session_id>/` directory (traces and artifacts)

It defaults to a dry run that only lists what would be pruned; pass `--allow-delete` to actually
remove anything, mirroring the explicit-consent pattern this codebase already uses for other
irreversible or costly operations (e.g. `--allow-paid-calls` on `scripts/live_evaluation.py`).
"""

import argparse
import asyncio
import os
import shutil
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


def _stale_session_ids_sync(database: Path, cutoff: datetime) -> list[str]:
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT session_id, updated_at FROM app_sessions").fetchall()
    return [
        str(session_id)
        for session_id, updated_at in rows
        if datetime.fromisoformat(updated_at) < cutoff
    ]


def _delete_session_row_sync(database: Path, session_id: str) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM app_sessions WHERE session_id = ?", (session_id,))


async def prune(workspace_root: Path, *, older_than_days: int, allow_delete: bool) -> list[str]:
    database = workspace_root / "checkpoints.sqlite"
    if not database.is_file():
        return []
    cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
    stale = await asyncio.to_thread(_stale_session_ids_sync, database, cutoff)
    if not stale or not allow_delete:
        return stale
    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        for session_id in stale:
            await saver.adelete_thread(session_id)
    for session_id in stale:
        await asyncio.to_thread(_delete_session_row_sync, database, session_id)
        session_dir = workspace_root / "sessions" / session_id
        if session_dir.is_dir():
            await asyncio.to_thread(shutil.rmtree, session_dir)
    return stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(os.environ.get("WORKSPACE_ROOT", "workspace")),
        help="Defaults to $WORKSPACE_ROOT or ./workspace; deliberately independent of the full "
        "application Settings (no Vertex/GCP configuration required to prune local files).",
    )
    parser.add_argument("--older-than-days", type=int, default=30)
    parser.add_argument(
        "--allow-delete",
        action="store_true",
        help="Actually delete. Without this flag, only lists what would be pruned.",
    )
    args = parser.parse_args()
    stale = asyncio.run(
        prune(
            args.workspace_root,
            older_than_days=args.older_than_days,
            allow_delete=args.allow_delete,
        )
    )
    if not stale:
        print(f"No sessions inactive more than {args.older_than_days} day(s) were found.")
        return 0
    verb = "Pruned" if args.allow_delete else "Would prune (pass --allow-delete to actually remove)"
    print(f"{verb} {len(stale)} session(s):")
    for session_id in stale:
        print(f"  {session_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
