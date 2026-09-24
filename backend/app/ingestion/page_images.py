"""Render one authoritative PDF page, optionally highlighting a citation's quoted text.

This lets a reader check a citation against the original page image rather than against
extracted text. Highlighting is best effort: extracted Markdown (tables in particular) does not
always match the PDF text layer token-for-token, so a fragment is only highlighted when it
locates unambiguously on the page.
"""

import re
from pathlib import Path
from typing import Literal

import pymupdf

_MAX_FRAGMENTS = 40
_MAX_MATCHES_PER_FRAGMENT = 3
_ZOOM = 2.0


def highlight_fragments(snippet: str) -> list[str]:
    fragments: list[str] = []
    for part in re.split(r"[\n|]+", snippet):
        # Extracted Markdown carries inline HTML (`<b>KARNATAKA</b>`, `<br>`) as well as
        # Markdown emphasis; neither exists in the PDF text layer.
        text = re.sub(r"</?[A-Za-z][^>]*>", " ", part)
        text = re.sub(r"[*#_`>]+", " ", text)
        text = " ".join(text.split()).strip(" -:")
        if len(text) >= 4 and re.search(r"[A-Za-z0-9]", text) and set(text) != {"-"}:
            fragments.append(text)
    return list(dict.fromkeys(fragments))[:_MAX_FRAGMENTS]


HighlightStatus = Literal["none", "matched", "unmatched", "no_text_layer"]


def render_page_png(
    pdf_path: Path, page_number: int, highlight: str | None = None
) -> tuple[bytes, HighlightStatus]:
    """Return the page as PNG plus whether the quote could be highlighted.

    Some source PDFs (including the bundled Census reports) draw every glyph as vector paths and
    have no text layer, so there is nothing to search; the caller reports that honestly instead
    of implying the quote was located.
    """
    with pymupdf.open(pdf_path) as pdf:
        if not 1 <= page_number <= pdf.page_count:
            raise IndexError("Page out of range")
        page = pdf[page_number - 1]
        status: HighlightStatus = "none"
        if highlight:
            if not page.get_text().strip():
                status = "no_text_layer"
            else:
                status = "unmatched"
                for fragment in highlight_fragments(highlight):
                    rects = page.search_for(fragment)
                    if 0 < len(rects) <= _MAX_MATCHES_PER_FRAGMENT:
                        annotation = page.add_highlight_annot(rects)
                        annotation.set_colors(stroke=(1.0, 0.82, 0.2))
                        annotation.update()
                        status = "matched"
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(_ZOOM, _ZOOM), annots=True)
        return pixmap.tobytes("png"), status


def page_count(pdf_path: Path) -> int:
    with pymupdf.open(pdf_path) as pdf:
        return int(pdf.page_count)
