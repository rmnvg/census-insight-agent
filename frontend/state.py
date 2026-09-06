import hashlib
from collections.abc import Callable, MutableMapping
from typing import Any, cast


def initialize_ui_state(state: MutableMapping[str, Any]) -> None:
    state.setdefault("session_id", None)
    state.setdefault("messages", [])
    state.setdefault("submission_in_progress", False)
    state.setdefault("pending_fingerprint", None)
    state.setdefault("turn_number", 0)
    state.setdefault("queued_prompt", None)
    state.setdefault("show_execution_details", True)
    state.setdefault("sidebar_snapshot", None)


def replace_conversation(state: MutableMapping[str, Any], session_id: str) -> None:
    state["session_id"] = session_id
    state["messages"] = []
    state["submission_in_progress"] = False
    state["pending_fingerprint"] = None
    state["turn_number"] = 0
    state["queued_prompt"] = None


def begin_submission(state: MutableMapping[str, Any], message: str) -> str | None:
    if state.get("submission_in_progress"):
        return None
    turn = int(state.get("turn_number", 0)) + 1
    fingerprint = hashlib.sha256(f"{turn}\0{message}".encode()).hexdigest()
    if fingerprint == state.get("pending_fingerprint"):
        return None
    state["submission_in_progress"] = True
    state["pending_fingerprint"] = fingerprint
    state["turn_number"] = turn
    return fingerprint


def finish_submission(state: MutableMapping[str, Any], fingerprint: str) -> None:
    if state.get("pending_fingerprint") == fingerprint:
        state["pending_fingerprint"] = None
    state["submission_in_progress"] = False
    state["queued_prompt"] = None


def cached_sidebar_snapshot[Snapshot](
    state: MutableMapping[str, Any], loader: Callable[[], Snapshot]
) -> Snapshot:
    cached = state.get("sidebar_snapshot")
    if cached is None:
        cached = loader()
        state["sidebar_snapshot"] = cached
    return cast(Snapshot, cached)


def clear_sidebar_snapshot(state: MutableMapping[str, Any]) -> None:
    state["sidebar_snapshot"] = None
