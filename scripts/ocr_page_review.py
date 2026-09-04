"""Run offline Tesseract OCR for the approved Karnataka review pages only.

This utility is intentionally isolated from the production ingestion workflow.
It reads existing page-review PNGs, invokes only the local Tesseract executable,
and writes page-bounded raw text, annotations, and review metadata.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from PIL import Image, ImageDraw

DOCUMENT_ID = "census-2011-karnataka-pca-highlights"
PDF_FILENAME = "PC11_PCA_Data_Highlights_Karnataka.pdf"
REQUESTED_PAGES = (13, 19, 21, 24, 28, 31, 33, 37, 43, 48, 51, 56)
MAP_PAGES = frozenset({13, 31, 37, 51})
EXTRACTION_METHOD = "Tesseract 5 local CLI (eng, --psm 11)"


@dataclass(frozen=True)
class OcrWord:
    """A word and its Tesseract-provided location and confidence."""

    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class OcrPageReport(TypedDict):
    """Serializable review metadata for one authoritative PDF page."""

    document_id: str
    page_number: int
    source_pdf: str
    source_checksum: str
    extraction_method: str
    raw_ocr_text_path: str
    non_empty_text: bool
    ocr_confidence: float | None
    detected_word_count: int
    requires_manual_review: bool
    warnings: list[str]
    short_extraction_summary: str


class OcrReviewReport(TypedDict):
    """Serializable aggregate OCR review report."""

    document_id: str
    source_pdf: str
    source_checksum: str
    extraction_method: str
    requested_pages: list[int]
    processed_page_count: int
    all_pages_require_manual_review: bool
    pages: list[OcrPageReport]


def run_local_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run a local executable without a shell."""
    return subprocess.run(command, check=True, capture_output=True, text=True)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a source file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_tsv(raw_tsv: str) -> list[OcrWord]:
    """Parse word-level rows from unmodified Tesseract TSV output."""
    words: list[OcrWord] = []
    for row in csv.DictReader(io.StringIO(raw_tsv), delimiter="\t"):
        text = row.get("text", "")
        confidence_text = row.get("conf", "-1")
        if not text or not confidence_text:
            continue
        confidence = float(confidence_text)
        if confidence < 0:
            continue
        words.append(
            OcrWord(
                text=text,
                confidence=confidence,
                left=int(row["left"]),
                top=int(row["top"]),
                width=int(row["width"]),
                height=int(row["height"]),
            )
        )
    return words


def annotate_words(source_image: Path, destination: Path, words: Sequence[OcrWord]) -> None:
    """Draw Tesseract word boxes without altering the source preview."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_image) as image:
        annotated = image.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        for word in words:
            draw.rectangle(
                (
                    word.left,
                    word.top,
                    word.left + word.width,
                    word.top + word.height,
                ),
                outline="red",
                width=2,
            )
        annotated.save(destination, format="PNG")


def page_warnings(page_number: int, non_empty_text: bool) -> list[str]:
    """Return review warnings without interpreting or correcting OCR content."""
    warnings = [
        "Raw OCR is uncorrected and must be checked against the authoritative PDF page.",
        "Plain OCR text does not preserve two-dimensional spatial relationships.",
    ]
    if page_number in MAP_PAGES:
        warnings.append("District labels may become detached from map regions and legend ranges.")
    else:
        warnings.append(
            "District or chart-category labels may become detached from values and percentages."
        )
    if not non_empty_text:
        warnings.append("Tesseract returned no non-whitespace text for this page.")
    return warnings


def extract_page(
    *,
    root: Path,
    page_number: int,
    source_checksum: str,
    runner: CommandRunner = run_local_command,
) -> OcrPageReport:
    """Extract exactly one one-based page preview into page-specific outputs."""
    if page_number not in REQUESTED_PAGES:
        raise ValueError(f"Page {page_number} is not approved for Karnataka OCR review")

    review_root = root / "data" / "processed" / "page-review"
    source_pdf = root / "data" / "source" / "pdf" / PDF_FILENAME
    source_image = review_root / "karnataka" / f"page-{page_number}.png"
    raw_text_path = review_root / "karnataka" / "ocr" / f"page-{page_number}.txt"
    annotation_path = review_root / "karnataka" / "ocr-annotated" / f"page-{page_number}.png"
    if not source_image.is_file():
        raise FileNotFoundError(f"Missing page preview: {source_image}")

    raw_text_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"ocr-page-{page_number}-") as temporary:
        output_base = Path(temporary) / f"page-{page_number}"
        runner(
            (
                "tesseract",
                str(source_image),
                str(output_base),
                "-l",
                "eng",
                "--psm",
                "11",
                "txt",
                "tsv",
            )
        )
        generated_text = output_base.with_suffix(".txt")
        generated_tsv = output_base.with_suffix(".tsv")
        raw_text_path.write_bytes(generated_text.read_bytes())
        raw_text = generated_text.read_text(encoding="utf-8")
        words = parse_tsv(generated_tsv.read_text(encoding="utf-8"))

    annotate_words(source_image, annotation_path, words)
    non_empty_text = bool(raw_text.strip())
    confidence = round(sum(word.confidence for word in words) / len(words), 2) if words else None
    relative_pdf = source_pdf.relative_to(root).as_posix()
    relative_text = raw_text_path.relative_to(root).as_posix()
    return {
        "document_id": DOCUMENT_ID,
        "page_number": page_number,
        "source_pdf": relative_pdf,
        "source_checksum": source_checksum,
        "extraction_method": EXTRACTION_METHOD,
        "raw_ocr_text_path": relative_text,
        "non_empty_text": non_empty_text,
        "ocr_confidence": confidence,
        "detected_word_count": len(words),
        "requires_manual_review": True,
        "warnings": page_warnings(page_number, non_empty_text),
        "short_extraction_summary": (
            f"Tesseract detected {len(words)} word tokens; spatial chart or map "
            "relationships require manual verification."
        ),
    }


def run_review(
    *,
    root: Path,
    page_numbers: Sequence[int] = REQUESTED_PAGES,
    runner: CommandRunner = run_local_command,
) -> OcrReviewReport:
    """Run OCR for an approved, unique subset and write the review report."""
    requested = tuple(page_numbers)
    if len(requested) != len(set(requested)):
        raise ValueError("OCR page list contains duplicates")
    unapproved = sorted(set(requested) - set(REQUESTED_PAGES))
    if unapproved:
        raise ValueError(f"Unapproved OCR pages requested: {unapproved}")

    source_pdf = root / "data" / "source" / "pdf" / PDF_FILENAME
    source_checksum = sha256_file(source_pdf)
    results = [
        extract_page(
            root=root,
            page_number=page_number,
            source_checksum=source_checksum,
            runner=runner,
        )
        for page_number in requested
    ]
    report: OcrReviewReport = {
        "document_id": DOCUMENT_ID,
        "source_pdf": source_pdf.relative_to(root).as_posix(),
        "source_checksum": source_checksum,
        "extraction_method": EXTRACTION_METHOD,
        "requested_pages": list(requested),
        "processed_page_count": len(results),
        "all_pages_require_manual_review": True,
        "pages": results,
    }
    report_path = root / "data" / "processed" / "page-review" / "ocr-review-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    """Run the fixed offline OCR review from the repository root."""
    root = Path(__file__).resolve().parents[1]
    run_review(root=root)


if __name__ == "__main__":
    main()
