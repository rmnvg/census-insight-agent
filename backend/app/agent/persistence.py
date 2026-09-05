import asyncio
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from backend.app.agent.models import (
    ClaimDerivation,
    RunTrace,
    SessionRecord,
    ValidatedClaimRecord,
    ValidatedClaimTurn,
)

SESSION_ID = re.compile(r"^[0-9a-f]{32}$")


class SessionStore:
    def __init__(self, workspace_root: Path) -> None:
        self.root = workspace_root
        self.database = self.root / "checkpoints.sqlite"
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS app_sessions "
                "(session_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
                "checkpoint_id TEXT)"
            )
            columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(app_sessions)")}
            if "checkpoint_id" not in columns:
                connection.execute("ALTER TABLE app_sessions ADD COLUMN checkpoint_id TEXT")

    async def create(self) -> SessionRecord:
        now = datetime.now(UTC)
        record = SessionRecord(session_id=uuid4().hex, created_at=now, updated_at=now)
        async with self._lock:
            await asyncio.to_thread(self._insert_sync, record)
        return record

    def _insert_sync(self, record: SessionRecord) -> None:
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "INSERT INTO app_sessions "
                "(session_id, created_at, updated_at, checkpoint_id) VALUES (?, ?, ?, NULL)",
                (record.session_id, record.created_at.isoformat(), record.updated_at.isoformat()),
            )

    async def get(self, session_id: str) -> SessionRecord | None:
        validate_session_id(session_id)
        row = await asyncio.to_thread(self._get_sync, session_id)
        if row is None:
            return None
        return SessionRecord(
            session_id=row[0],
            created_at=datetime.fromisoformat(row[1]),
            updated_at=datetime.fromisoformat(row[2]),
        )

    def _get_sync(self, session_id: str) -> tuple[str, str, str] | None:
        with sqlite3.connect(self.database) as connection:
            row = connection.execute(
                "SELECT session_id, created_at, updated_at FROM app_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row if row is None else (str(row[0]), str(row[1]), str(row[2]))

    async def touch(self, session_id: str) -> None:
        now = datetime.now(UTC).isoformat()
        await asyncio.to_thread(self._touch_sync, session_id, now)

    def _touch_sync(self, session_id: str, now: str) -> None:
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "UPDATE app_sessions SET updated_at = ? WHERE session_id = ?", (now, session_id)
            )

    async def get_checkpoint_id(self, session_id: str) -> str | None:
        return await asyncio.to_thread(self._get_checkpoint_id_sync, session_id)

    def _get_checkpoint_id_sync(self, session_id: str) -> str | None:
        with sqlite3.connect(self.database) as connection:
            row = connection.execute(
                "SELECT checkpoint_id FROM app_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return None if row is None or row[0] is None else str(row[0])

    async def set_checkpoint_id(self, session_id: str, checkpoint_id: str) -> None:
        await asyncio.to_thread(self._set_checkpoint_id_sync, session_id, checkpoint_id)

    def _set_checkpoint_id_sync(self, session_id: str, checkpoint_id: str) -> None:
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "UPDATE app_sessions SET checkpoint_id = ? WHERE session_id = ?",
                (checkpoint_id, session_id),
            )


class TraceStore:
    def __init__(self, workspace_root: Path) -> None:
        self.root = workspace_root

    def write(self, trace: RunTrace) -> Path:
        validate_session_id(trace.session_id)
        UUID(trace.run_id)
        path = self.root / "sessions" / trace.session_id / "traces" / f"{trace.run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(trace.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    def read(self, run_id: str) -> RunTrace | None:
        UUID(run_id)
        matches = list((self.root / "sessions").glob(f"*/traces/{run_id}.json"))
        if len(matches) != 1:
            return None
        return RunTrace.model_validate_json(matches[0].read_text(encoding="utf-8"))

    def recover_validated_claim_history(self, session_id: str) -> list[ValidatedClaimTurn]:
        """Migrate successful pre-v3 traces without treating assistant prose as evidence."""
        validate_session_id(session_id)
        recovered: list[tuple[str, ValidatedClaimTurn]] = []
        paths = (self.root / "sessions" / session_id / "traces").glob("*.json")
        for path in paths:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("answer_status") != "answered" or raw.get("run_status") != "completed":
                continue
            events = raw.get("events", [])
            classification = next(
                (item for item in events if item.get("event") == "task_classification"), None
            )
            draft_event = next(
                (item for item in reversed(events) if item.get("event") == "structured_draft"),
                None,
            )
            citations_event = next(
                (item for item in reversed(events) if item.get("event") == "validated_citations"),
                None,
            )
            if not classification or not draft_event or not citations_event:
                continue
            task_type = classification.get("details", {}).get("task_type")
            if task_type not in {"lookup", "summary", "comparison", "inconsistency_analysis"}:
                continue
            citations = citations_event.get("details", {}).get("citations", [])
            claims: list[ValidatedClaimRecord] = []
            for item in draft_event.get("details", {}).get("claims", []):
                evidence_ids = list(dict.fromkeys(item.get("evidence_ids", [])))
                matched = [value for value in citations if value.get("chunk_id") in evidence_ids]
                value = item.get("value")
                claims.append(
                    ValidatedClaimRecord(
                        claim_id=item["claim_id"],
                        text=item.get("text", ""),
                        metric=item.get("metric"),
                        region=item.get("region"),
                        year=item.get("year"),
                        population_scope=item.get("population_scope"),
                        residence_scope=item.get("residence_scope"),
                        value=round(float(value), 10) if value is not None else None,
                        displayed_value=(
                            f"{abs(float(value)):.10f}".rstrip("0").rstrip(".")
                            if value is not None
                            else None
                        ),
                        unit=item.get("unit"),
                        document_derived=bool(item.get("document_derived")),
                        derivation=(
                            ClaimDerivation.model_validate(item["derivation"])
                            if item.get("derivation")
                            else None
                        ),
                        evidence_ids=[value["chunk_id"] for value in matched],
                        citation_ids=[value["citation_id"] for value in matched],
                        document_ids=[value["document_id"] for value in matched],
                        page_numbers=[value["page_number"] for value in matched],
                        source_checksums=["" for _ in matched],
                    )
                )
            if claims:
                recovered.append(
                    (
                        events[0].get("timestamp", ""),
                        ValidatedClaimTurn(
                            run_id=raw["run_id"], task_type=task_type, claims=claims
                        ),
                    )
                )
        return [turn for _, turn in sorted(recovered, key=lambda item: item[0])][-8:]


def validate_session_id(value: str) -> str:
    if not SESSION_ID.fullmatch(value):
        raise ValueError("Invalid session ID")
    return value


def safe_trace_details(**values: object) -> dict[str, object]:
    """Keep trace summaries observable without prompts, secrets, tokens, or embeddings."""
    forbidden = ("credential", "token", "embedding", "prompt", "chain_of_thought", "api_key")
    return {
        key: value
        for key, value in values.items()
        if not any(part in key.casefold() for part in forbidden)
        and len(json.dumps(value, default=str)) <= 2000
    }
