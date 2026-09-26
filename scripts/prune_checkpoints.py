#!/usr/bin/env python3
"""Prune stale session checkpoints, transcripts, traces, and artifacts.

Every graph superstep for every turn writes a full LangGraph state snapshot, and nothing else in
this codebase ever deletes an old one — the checkpointers (and this application's own
trace/artifact stores) are append-only by design. Locally `workspace/checkpoints.sqlite` can grow
to tens of megabytes across a handful of real sessions; nothing here is a correctness bug, but it
is unbounded growth with no built-in cleanup.

This script finds sessions inactive longer than `--older-than-days` (using the
`app_sessions.updated_at` column the session store already maintains), and removes:
  - their LangGraph checkpoints, via the checkpointer's `adelete_thread`
  - their `app_sessions` row and display transcript
  - their `workspace/sessions/<session_id>/` directory (traces and artifacts)

It works on the default SQLite state or, with `--state-url` / `$STATE_DATABASE_URL`, on Postgres
state. It defaults to a dry run that only lists what would be pruned; pass `--allow-delete` to
actually remove anything, mirroring the explicit-consent pattern this codebase already uses for
other irreversible or costly operations (e.g. `--allow-paid-calls` on
`scripts/live_evaluation.py`).
"""

import argparse
import asyncio
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from backend.app.agent.state import SqliteStateBackend, StateBackend


async def prune(
    workspace_root: Path,
    *,
    older_than_days: int,
    allow_delete: bool,
    state_url: str = "",
) -> list[str]:
    backend: StateBackend
    if state_url:
        from backend.app.agent.postgres_state import PostgresStateBackend

        backend = PostgresStateBackend(state_url)
    elif (workspace_root / "checkpoints.sqlite").is_file():
        backend = SqliteStateBackend(workspace_root)
    else:
        return []
    try:
        cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
        stale = await backend.sessions.stale_session_ids(cutoff)
        if not stale or not allow_delete:
            return stale
        async with backend.checkpointer() as saver:
            for session_id in stale:
                await saver.adelete_thread(session_id)
        for session_id in stale:
            await backend.sessions.delete(session_id)
            session_dir = workspace_root / "sessions" / session_id
            if session_dir.is_dir():
                await asyncio.to_thread(shutil.rmtree, session_dir)
        return stale
    finally:
        await backend.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(os.environ.get("WORKSPACE_ROOT", "workspace")),
        help="Defaults to $WORKSPACE_ROOT or ./workspace; deliberately independent of the full "
        "application Settings (no Vertex/GCP configuration required to prune local files).",
    )
    parser.add_argument(
        "--state-url",
        default=os.environ.get("STATE_DATABASE_URL", ""),
        help="Postgres state URL; defaults to $STATE_DATABASE_URL, else the SQLite workspace file.",
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
            state_url=args.state_url,
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
