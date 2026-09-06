from pathlib import Path

from scripts.smoke_ui_offline import run_smoke


def test_all_ui_fixtures_build_safe_view_models_without_requests() -> None:
    fixture = Path(__file__).parent / "fixtures" / "ui-smoke.json"
    result = run_smoke(fixture)
    assert result["passed"] is True
    assert result["external_requests"] == 0
