from typing import Any

from frontend.state import (
    begin_submission,
    cached_sidebar_snapshot,
    clear_sidebar_snapshot,
    finish_submission,
    initialize_ui_state,
    replace_conversation,
)


def test_initial_session_and_reuse_across_turns() -> None:
    state: dict[str, Any] = {}
    initialize_ui_state(state)
    replace_conversation(state, "session-one")
    first = begin_submission(state, "First")
    assert first is not None
    finish_submission(state, first)
    second = begin_submission(state, "Follow-up")
    assert second is not None
    assert state["session_id"] == "session-one"
    assert state["messages"] == []


def test_new_conversation_clears_only_ui_messages() -> None:
    state: dict[str, Any] = {"messages": [{"role": "user"}], "session_id": "old"}
    initialize_ui_state(state)
    replace_conversation(state, "new")
    assert state["session_id"] == "new"
    assert state["messages"] == []


def test_duplicate_submission_guard() -> None:
    state: dict[str, Any] = {}
    initialize_ui_state(state)
    fingerprint = begin_submission(state, "Question")
    assert fingerprint is not None
    assert begin_submission(state, "Question") is None
    finish_submission(state, fingerprint)
    assert not state["submission_in_progress"]


def test_sidebar_snapshot_is_loaded_once_until_explicit_refresh() -> None:
    state: dict[str, Any] = {}
    initialize_ui_state(state)
    calls = 0

    def load() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"status": "ok"}

    assert cached_sidebar_snapshot(state, load) == {"status": "ok"}
    assert cached_sidebar_snapshot(state, load) == {"status": "ok"}
    assert calls == 1
    clear_sidebar_snapshot(state)
    assert cached_sidebar_snapshot(state, load) == {"status": "ok"}
    assert calls == 2
