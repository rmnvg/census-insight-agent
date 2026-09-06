from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_completed_chat_renders_once_without_forced_rerun() -> None:
    fixture = Path(__file__).parent / "fixtures" / "app_flow.py"
    app = AppTest.from_file(fixture).run(timeout=10)
    app.chat_input[0].set_value("Test question")
    app = app.run(timeout=10)
    assert not app.exception
    assert app.session_state["fixture_chat_calls"] == 1
    assert any(element.value == "Rendered answer" for element in app.markdown)
    assert app.session_state["submission_in_progress"] is False
