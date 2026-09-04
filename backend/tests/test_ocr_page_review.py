from __future__ import annotations

import json
import socket
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from scripts.ocr_page_review import PDF_FILENAME, REQUESTED_PAGES, run_review, sha256_file


class FakeTesseract:
    def __init__(self, *, empty: bool = False) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.empty = empty

    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        captured = tuple(command)
        self.commands.append(captured)
        output_base = Path(captured[2])
        page_label = Path(captured[1]).stem
        text = "" if self.empty else f"token-{page_label}\n"
        output_base.with_suffix(".txt").write_text(text, encoding="utf-8")
        tsv = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
            "left\ttop\twidth\theight\tconf\ttext\n"
        )
        if not self.empty:
            tsv += f"5\t1\t1\t1\t1\t1\t1\t2\t10\t8\t91.5\ttoken-{page_label}\n"
        output_base.with_suffix(".tsv").write_text(tsv, encoding="utf-8")
        return subprocess.CompletedProcess(captured, 0, "", "")


def make_review_root(tmp_path: Path, pages: Sequence[int]) -> Path:
    pdf_path = tmp_path / "data" / "source" / "pdf" / PDF_FILENAME
    pdf_path.parent.mkdir(parents=True)
    document = pymupdf.open()
    for _ in range(max(pages)):
        document.new_page()
    document.save(pdf_path)
    document.close()

    preview_dir = tmp_path / "data" / "processed" / "page-review" / "karnataka"
    preview_dir.mkdir(parents=True)
    for page_number in pages:
        Image.new("RGB", (40, 40), "white").save(preview_dir / f"page-{page_number}.png")
    return pdf_path


def test_requested_pages_are_exactly_the_approved_one_based_pages() -> None:
    assert REQUESTED_PAGES == (13, 19, 21, 24, 28, 31, 33, 37, 43, 48, 51, 56)
    assert all(page_number >= 1 for page_number in REQUESTED_PAGES)


def test_processes_only_requested_pages_and_keeps_outputs_page_bounded(tmp_path: Path) -> None:
    pages = (13, 19)
    make_review_root(tmp_path, pages)
    fake = FakeTesseract()

    report = run_review(root=tmp_path, page_numbers=pages, runner=fake)

    assert report["requested_pages"] == [13, 19]
    assert [page["page_number"] for page in report["pages"]] == [13, 19]
    assert len(fake.commands) == 2
    assert (tmp_path / "data/processed/page-review/karnataka/ocr/page-13.txt").read_text() == (
        "token-page-13\n"
    )
    assert (tmp_path / "data/processed/page-review/karnataka/ocr/page-19.txt").read_text() == (
        "token-page-19\n"
    )
    assert not (tmp_path / "data/processed/page-review/karnataka/ocr/page-21.txt").exists()


def test_empty_ocr_is_preserved_and_reported(tmp_path: Path) -> None:
    make_review_root(tmp_path, (13,))

    report = run_review(root=tmp_path, page_numbers=(13,), runner=FakeTesseract(empty=True))
    page = report["pages"][0]

    assert page["non_empty_text"] is False
    assert page["ocr_confidence"] is None
    assert page["detected_word_count"] == 0
    assert (tmp_path / "data/processed/page-review/karnataka/ocr/page-13.txt").read_bytes() == b""


def test_source_checksum_is_preserved_per_page(tmp_path: Path) -> None:
    pdf_path = make_review_root(tmp_path, (13, 19))

    report = run_review(root=tmp_path, page_numbers=(13, 19), runner=FakeTesseract())
    expected = sha256_file(pdf_path)

    assert report["source_checksum"] == expected
    assert all(page["source_checksum"] == expected for page in report["pages"])


def test_rejects_unapproved_pages_before_running_ocr(tmp_path: Path) -> None:
    fake = FakeTesseract()

    with pytest.raises(ValueError, match="Unapproved OCR pages"):
        run_review(root=tmp_path, page_numbers=(12,), runner=fake)

    assert fake.commands == []


def test_workflow_makes_no_external_api_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_review_root(tmp_path, (13,))

    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"Unexpected network call: {args!r} {kwargs!r}")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    fake = FakeTesseract()
    run_review(root=tmp_path, page_numbers=(13,), runner=fake)

    assert fake.commands[0][0] == "tesseract"
    report_path = tmp_path / "data/processed/page-review/ocr-review-report.json"
    assert json.loads(report_path.read_text())["processed_page_count"] == 1
