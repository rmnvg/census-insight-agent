from pathlib import Path

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).parents[2]


def test_frontend_does_not_import_backend_qdrant_or_langgraph() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "frontend").rglob("*.py")
        if "tests" not in path.parts
    )
    for forbidden in ("backend.app", "qdrant", "langgraph", "executor."):
        assert f"import {forbidden}" not in source
        assert f"from {forbidden}" not in source


def test_frontend_compose_security_boundary() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    frontend = compose["services"]["frontend"]
    serialized = str(frontend)
    assert frontend["read_only"] is True
    assert frontend["user"] == "10002:10002"
    assert frontend["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in frontend["security_opt"]
    assert "healthcheck" in frontend
    assert "volumes" not in frontend
    assert frontend["networks"] == ["frontend_api"]
    assert compose["services"]["backend"]["networks"] == ["frontend_api", "backend_data"]
    assert compose["services"]["qdrant"]["networks"] == ["backend_data"]
    assert "internal" not in compose["networks"]["frontend_api"]
    for forbidden in (
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "QDRANT_URL",
        "execution-queue",
        "workspace",
        "docker.sock",
    ):
        assert forbidden not in serialized


def test_source_rendering_never_enables_unsafe_html() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "frontend").rglob("*.py")
        if "tests" not in path.parts
    )
    assert "unsafe_allow_html=True" not in source


def test_citation_renderer_does_not_nest_streamlit_expanders() -> None:
    source = (ROOT / "frontend" / "renderers" / "citations.py").read_text(encoding="utf-8")
    assert source.count("with st.expander(") == 1
