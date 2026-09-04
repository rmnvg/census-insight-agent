from pathlib import Path

import pytest

from backend.app.ingestion.errors import PageMappingError
from backend.app.ingestion.models import PageCoverageDecision
from backend.app.ingestion.pages import (
    map_document_pages,
    parse_converter_page_sequence,
    parse_page_markers,
)


def test_page_marker_parsing_detects_duplicates() -> None:
    mapped, duplicates, saw_markers = parse_page_markers(
        "<!-- page: 1 -->\nFirst\n<!-- PAGE 1 -->\nRepeated"
    )

    assert saw_markers is True
    assert duplicates == [1]
    assert "First" in mapped[1]


def test_missing_markdown_page_uses_fallback_and_one_based_numbering(tmp_path: Path) -> None:
    markdown = tmp_path / "sample.md"
    markdown.write_text("<!-- page: 1 -->\nProvided first page", encoding="utf-8")

    result = map_document_pages(
        document_id="doc",
        pdf_path=tmp_path / "sample.pdf",
        markdown_path=markdown,
        pdf_text_reader=lambda _: ["PDF first", "PDF second"],
        fallback_extractor=lambda _, page: f"Fallback page {page}",
    )

    assert [page.page_number for page in result.pages] == [1, 2]
    assert result.pages[0].extraction_method == "provided_markdown"
    assert result.pages[0].coverage_status == "indexed_provided_markdown"
    assert result.pages[1].text == "Fallback page 2"
    assert result.pages[1].coverage_status == "indexed_pymupdf4llm_fallback"
    assert result.fallback_pages == [2]


def test_unresolved_unmarked_markdown_fails_without_override(tmp_path: Path) -> None:
    markdown = tmp_path / "sample.md"
    markdown.write_text("Completely unrelated supplied prose", encoding="utf-8")

    with pytest.raises(PageMappingError, match="unresolved Markdown page alignment"):
        map_document_pages(
            document_id="doc",
            pdf_path=tmp_path / "sample.pdf",
            markdown_path=markdown,
            pdf_text_reader=lambda _: ["alpha census page", "beta census page"],
            fallback_extractor=lambda _, page: f"Fallback {page}",
        )


def test_empty_pdf_page_is_reported(tmp_path: Path) -> None:
    result = map_document_pages(
        document_id="doc",
        pdf_path=tmp_path / "sample.pdf",
        markdown_path=None,
        pdf_text_reader=lambda _: [""],
        fallback_extractor=lambda _path, _page: "",
    )

    assert result.empty_pages == [1]
    assert result.pages[0].page_number == 1
    assert result.pages[0].coverage_status == "failed_page_mapping"


def test_reviewed_exclusion_skips_fallback_and_is_not_a_failure(tmp_path: Path) -> None:
    decision = PageCoverageDecision(
        document_id="doc",
        page_number=1,
        coverage_status="excluded_unverified_visual",
        reason="Unverified chart OCR",
        ocr_confidence=90.0,
        review_status="unverified",
        source_checksum="a" * 64,
    )
    fallback_calls: list[int] = []

    def unsafe_fallback(_path: Path, page: int) -> str:
        fallback_calls.append(page)
        return "unsafe OCR"

    result = map_document_pages(
        document_id="doc",
        pdf_path=tmp_path / "sample.pdf",
        markdown_path=None,
        coverage_decisions={1: decision},
        pdf_text_reader=lambda _: [""],
        fallback_extractor=unsafe_fallback,
    )

    assert fallback_calls == []
    assert result.pages[0].text == ""
    assert result.pages[0].coverage_status == "excluded_unverified_visual"
    assert result.coverage.excluded_visual_pages == 1
    assert result.coverage.failed_mappings == 0


def test_converter_sequence_uses_counts_skips_and_order() -> None:
    markdown = """<!-- Source: report.pdf
Converted via Datalab Marker API
Pages converted: 2/3
Skipped: [2] -->

<!-- page 0 -->
First authoritative page

{99}------------------------------------------------
Third authoritative page
"""

    mapped = parse_converter_page_sequence(markdown, 3)

    assert mapped == {1: "First authoritative page", 3: "Third authoritative page"}


def test_converter_sequence_rejects_count_mismatch() -> None:
    markdown = """<!-- Source: report.pdf
Pages converted: 2/3
Skipped: [] -->
<!-- page 0 -->
Only block
"""

    with pytest.raises(PageMappingError, match="counts are inconsistent"):
        parse_converter_page_sequence(markdown, 3)
