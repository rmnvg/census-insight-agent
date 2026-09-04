"""Render the fixed set of pages awaiting manual ingestion review.

This utility is deliberately separate from the production ingestion workflow. It
only reads the authoritative PDFs and writes PNG previews for human inspection.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF_ROOT = ROOT / "data" / "source" / "pdf"
OUTPUT_ROOT = ROOT / "data" / "processed" / "page-review"

REVIEW_PAGES = {
    "karnataka": (
        "PC11_PCA_Data_Highlights_Karnataka.pdf",
        [1, 13, 19, 21, 24, 28, 31, 33, 37, 43, 48, 51, 56],
    ),
    "odisha": ("PC11_PCA_Data_Highlights_Odisha.pdf", [134]),
    "madhya-pradesh": ("PCA Data Highlights MP.pdf", [7, 19, 47, 63]),
}


def main() -> None:
    """Render configured one-based PDF pages at 150 DPI."""
    for document_name, (pdf_filename, page_numbers) in REVIEW_PAGES.items():
        output_dir = OUTPUT_ROOT / document_name
        output_dir.mkdir(parents=True, exist_ok=True)

        with pymupdf.open(PDF_ROOT / pdf_filename) as document:
            for page_number in page_numbers:
                if not 1 <= page_number <= document.page_count:
                    raise ValueError(
                        f"Page {page_number} is outside {pdf_filename} (1-{document.page_count})"
                    )
                page = document.load_page(page_number - 1)
                pixmap = page.get_pixmap(dpi=150, alpha=False)
                pixmap.save(output_dir / f"page-{page_number}.png")


if __name__ == "__main__":
    main()
