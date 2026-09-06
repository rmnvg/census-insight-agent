from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_full_response_with_claims_citations_and_artifact_renders_without_exception() -> None:
    fixture = Path(__file__).parent / "fixtures" / "app_flow_rich.py"
    app = AppTest.from_file(fixture).run(timeout=10)
    assert not app.exception
    app.chat_input[0].set_value("Test question")
    app = app.run(timeout=10)
    assert not app.exception
    assert any("Balaghat" in element.value for element in app.markdown)
    assert any("Citation-safe" in element.value for element in app.markdown)
    assert any(button.label == "Download table CSV" for button in app.download_button)
