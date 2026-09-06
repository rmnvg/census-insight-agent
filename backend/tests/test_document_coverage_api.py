import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend.app import api
from backend.app.ingestion.coverage import build_coverage_report
from backend.app.ingestion.models import PageCoverageEntry


def public_documents() -> list[api.DocumentPublicSummary]:
    return [
        api.DocumentPublicSummary(
            document_id="doc-karnataka",
            title="Karnataka report",
            region="Karnataka",
            source_checksum="a" * 64,
        )
    ]


def write_report(root: Path) -> None:
    processed = root / "processed"
    processed.mkdir()
    coverage = build_coverage_report(
        "doc-karnataka",
        2,
        [
            PageCoverageEntry(page_number=1, status="indexed_provided_markdown"),
            PageCoverageEntry(
                page_number=2,
                status="excluded_unverified_visual",
                reason="Reviewed visual page",
            ),
        ],
    )
    (processed / "dry-run-report.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_id": "doc-karnataka",
                        "coverage": coverage.model_dump(mode="json"),
                        "limitations": [
                            {
                                "document_id": "doc-karnataka",
                                "excluded_pages": [2],
                                "statuses": ["excluded_unverified_visual"],
                                "message": (
                                    "Some visual pages were excluded from automated answering."
                                ),
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_document_coverage_endpoint_uses_trusted_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_report(tmp_path)
    monkeypatch.setattr(api, "_public_documents", public_documents)
    monkeypatch.setattr(api, "get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    result = api.document_coverage("doc-karnataka")
    assert result.document.title == "Karnataka report"
    assert result.coverage.indexed_pages == 1
    assert result.coverage.excluded_visual_pages == 1
    assert result.limitations[0].excluded_pages == [2]


def test_document_coverage_endpoint_hides_unknown_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_report(tmp_path)
    monkeypatch.setattr(api, "_public_documents", public_documents)
    monkeypatch.setattr(api, "get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    with pytest.raises(HTTPException) as caught:
        api.document_coverage("unknown")
    assert caught.value.status_code == 404


def test_document_listing_does_not_scroll_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "_public_documents", public_documents)
    monkeypatch.setattr(
        api,
        "_store",
        lambda: (_ for _ in ()).throw(AssertionError("Qdrant must not be scanned")),
    )
    result = api.documents()
    assert [item.document_id for item in result] == ["doc-karnataka"]
