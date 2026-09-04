import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pymupdf
import pymupdf4llm  # type: ignore[import-untyped]

from backend.app.ingestion.coverage import build_coverage_report
from backend.app.ingestion.errors import PageMappingError
from backend.app.ingestion.models import (
    DocumentPage,
    ManualTranscriptionRecord,
    PageCoverageDecision,
    PageCoverageEntry,
    PageCoverageStatus,
    PageExtractionMethod,
    PageMappingResult,
)

PAGE_MARKER = re.compile(
    r"^\s*(?:<!--\s*page(?:_number)?\s*[: ]\s*(\d+)\s*-->|\[\s*page\s+(\d+)\s*\])\s*$",
    re.IGNORECASE,
)
CONVERTER_PAGE_MARKER = re.compile(
    r"^\s*(?:<!--\s*page(?:_number)?\s*[: ]\s*\d+\s*-->|\{\d+\}-{3,})\s*$",
    re.IGNORECASE | re.MULTILINE,
)
CONVERTER_METADATA = re.compile(
    r"<!--\s*Source:.*?Pages\s+converted:\s*(\d+)\s*/\s*(\d+)"
    r"(?:\s*Skipped:\s*\[([^\]]*)\])?.*?-->",
    re.IGNORECASE | re.DOTALL,
)
WORD = re.compile(r"[a-z0-9]+")


def read_pdf_text(pdf_path: Path) -> list[str]:
    """Read authoritative one-entry-per-page PDF text."""
    with pymupdf.open(pdf_path) as document:
        return [page.get_text("text") for page in document]


def extract_pdf_page_markdown(pdf_path: Path, page_number: int) -> str:
    """Extract exactly one one-based PDF page with PyMuPDF4LLM."""
    with pymupdf.open(pdf_path) as document:
        extracted = pymupdf4llm.to_markdown(document, pages=[page_number - 1])
    return str(extracted).strip()


def parse_page_markers(markdown: str) -> tuple[dict[int, str], list[int], bool]:
    """Parse explicit reliable page markers without inferring page numbers."""
    pages: dict[int, list[str]] = {}
    duplicates: list[int] = []
    current_page: int | None = None
    saw_marker = False
    preamble: list[str] = []

    for line in markdown.splitlines():
        match = PAGE_MARKER.match(line)
        if match:
            saw_marker = True
            current_page = int(match.group(1) or match.group(2))
            if current_page in pages:
                duplicates.append(current_page)
            pages.setdefault(current_page, [])
        elif current_page is None:
            preamble.append(line)
        else:
            pages[current_page].append(line)

    if saw_marker and any(line.strip() for line in preamble):
        pages.setdefault(0, []).extend(preamble)
    mapped = {number: "\n".join(lines).strip() for number, lines in pages.items()}
    return mapped, duplicates, saw_marker


def parse_converter_page_sequence(markdown: str, pdf_page_count: int) -> dict[int, str] | None:
    """Map validated converter blocks by order when marker labels are inconsistent."""
    metadata = CONVERTER_METADATA.search(markdown)
    if metadata is None:
        return None
    converted_count = int(metadata.group(1))
    declared_page_count = int(metadata.group(2))
    skipped_text = metadata.group(3) or ""
    skipped_pages = {int(value) for value in re.findall(r"\d+", skipped_text)}
    if declared_page_count != pdf_page_count:
        raise PageMappingError(
            "Converter metadata page count does not match the authoritative PDF",
            page_count_mismatch=True,
        )
    if any(page < 1 or page > pdf_page_count for page in skipped_pages):
        raise PageMappingError(
            "Converter metadata contains an invalid skipped page",
            unresolved_pages=sorted(skipped_pages),
            page_count_mismatch=True,
        )
    if converted_count != pdf_page_count - len(skipped_pages):
        raise PageMappingError(
            "Converter counts are inconsistent with its skipped-page list",
            page_count_mismatch=True,
        )

    content = f"{markdown[: metadata.start()]}{markdown[metadata.end() :]}"
    markers = list(CONVERTER_PAGE_MARKER.finditer(content))
    blocks: list[str] = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(content)
        blocks.append(content[marker.end() : end].strip())
    if len(blocks) != converted_count:
        raise PageMappingError(
            f"Converter declared {converted_count} pages but supplied {len(blocks)} page blocks",
            page_count_mismatch=True,
        )

    authoritative_pages = [
        page for page in range(1, pdf_page_count + 1) if page not in skipped_pages
    ]
    return dict(zip(authoritative_pages, blocks, strict=True))


def _tokens(text: str) -> list[str]:
    return WORD.findall(text.casefold())


def _blocks(markdown: str) -> list[str]:
    raw_blocks = [block.strip() for block in re.split(r"\n\s*\n", markdown) if block.strip()]
    blocks: list[str] = []
    pending_heading: str | None = None
    for block in raw_blocks:
        if all(line.lstrip().startswith("#") for line in block.splitlines()):
            pending_heading = f"{pending_heading}\n{block}" if pending_heading else block
            continue
        if pending_heading:
            block = f"{pending_heading}\n\n{block}"
            pending_heading = None
        blocks.append(block)
    if pending_heading:
        blocks.append(pending_heading)
    return blocks


def _alignment_score(block: str, page_text: str) -> float:
    block_tokens = _tokens(block)
    page_tokens = _tokens(page_text)
    if not block_tokens or not page_tokens:
        return 0.0
    normalized_block = " ".join(block_tokens)
    normalized_page = " ".join(page_tokens)
    if len(block_tokens) >= 4 and normalized_block in normalized_page:
        return 1.0
    block_counts = Counter(block_tokens)
    page_counts = Counter(page_tokens)
    overlap = sum(min(count, page_counts[token]) for token, count in block_counts.items())
    return overlap / len(block_tokens)


def align_unmarked_markdown(markdown: str, pdf_pages: list[str]) -> tuple[dict[int, str], bool]:
    """Align Markdown blocks only when one PDF page is a confident unique match."""
    if len(pdf_pages) == 1 and markdown.strip():
        return {1: markdown.strip()}, False

    assigned: dict[int, list[str]] = {}
    unresolved = False
    for block in _blocks(markdown):
        scores = [_alignment_score(block, page) for page in pdf_pages]
        ranked = sorted(enumerate(scores, start=1), key=lambda item: item[1], reverse=True)
        best_page, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score < 0.75 or best_score - second_score < 0.15:
            unresolved = True
            continue
        assigned.setdefault(best_page, []).append(block)
    return ({page: "\n\n".join(blocks) for page, blocks in assigned.items()}, unresolved)


def map_document_pages(
    *,
    document_id: str,
    pdf_path: Path,
    markdown_path: Path | None,
    review_override: bool = False,
    coverage_decisions: dict[int, PageCoverageDecision] | None = None,
    manual_transcriptions: dict[int, ManualTranscriptionRecord] | None = None,
    pdf_text_reader: Callable[[Path], list[str]] = read_pdf_text,
    fallback_extractor: Callable[[Path, int], str] = extract_pdf_page_markdown,
) -> PageMappingResult:
    """Map preferred Markdown to authoritative PDF pages, falling back page-by-page."""
    pdf_pages = pdf_text_reader(pdf_path)
    if not pdf_pages:
        raise PageMappingError(f"PDF has no pages: {pdf_path.name}")
    coverage_decisions = coverage_decisions or {}
    manual_transcriptions = manual_transcriptions or {}
    invalid_review_pages = sorted(
        (set(coverage_decisions) | set(manual_transcriptions)) - set(range(1, len(pdf_pages) + 1))
    )
    if invalid_review_pages:
        raise PageMappingError(
            f"Coverage records contain invalid PDF pages {invalid_review_pages}",
            unresolved_pages=invalid_review_pages,
        )

    mapped: dict[int, str] = {}
    duplicates: list[int] = []
    unresolved: list[int] = []
    saw_markers = False
    trusted_converter_sequence = False
    if markdown_path:
        markdown = markdown_path.read_text(encoding="utf-8")
        converter_mapping = parse_converter_page_sequence(markdown, len(pdf_pages))
        if converter_mapping is not None:
            mapped = converter_mapping
            trusted_converter_sequence = True
        else:
            mapped, duplicates, saw_markers = parse_page_markers(markdown)
            if saw_markers:
                invalid = sorted(
                    number for number in mapped if number < 1 or number > len(pdf_pages)
                )
                unresolved.extend(invalid)
                mapped = {
                    number: text for number, text in mapped.items() if 1 <= number <= len(pdf_pages)
                }
            else:
                mapped, alignment_unresolved = align_unmarked_markdown(markdown, pdf_pages)
                if alignment_unresolved:
                    unresolved.append(0)

    if (duplicates or unresolved) and not review_override:
        problems = []
        if duplicates:
            problems.append(f"duplicate page markers {sorted(set(duplicates))}")
        if unresolved:
            problems.append("unresolved Markdown page alignment")
        raise PageMappingError(
            f"{pdf_path.name}: {'; '.join(problems)}",
            unresolved_pages=unresolved,
            duplicate_pages=duplicates,
            page_count_mismatch=bool(unresolved or duplicates),
        )
    if (unresolved or duplicates) and review_override:
        mapped = {}

    pages: list[DocumentPage] = []
    markdown_pages: list[int] = []
    fallback_pages: list[int] = []
    empty_pages: list[int] = []
    coverage_entries: list[PageCoverageEntry] = []
    for page_number, _pdf_text in enumerate(pdf_pages, start=1):
        transcription = manual_transcriptions.get(page_number)
        decision = coverage_decisions.get(page_number)
        reason: str | None = None
        if transcription:
            text = transcription.transcription
            method: PageExtractionMethod | None = "approved_manual_transcription"
            status: PageCoverageStatus = "approved_manual_transcription"
        elif decision:
            text = ""
            method = None
            status = decision.coverage_status
            reason = decision.reason
            if status in {"excluded_blank", "excluded_decorative"}:
                empty_pages.append(page_number)
        else:
            supplied = mapped.get(page_number, "").strip()
            if supplied:
                text = supplied
                method = "provided_markdown"
                status = "indexed_provided_markdown"
                markdown_pages.append(page_number)
            else:
                text = fallback_extractor(pdf_path, page_number).strip()
                method = "pymupdf4llm_fallback"
                fallback_pages.append(page_number)
                if text:
                    status = "indexed_pymupdf4llm_fallback"
                else:
                    method = None
                    status = "failed_page_mapping"
                    reason = "No citation-safe text was available for the PDF page"
                    empty_pages.append(page_number)
                    unresolved.append(page_number)
        pages.append(
            DocumentPage(
                document_id=document_id,
                page_number=page_number,
                text=text,
                extraction_method=method,
                coverage_status=status,
            )
        )
        coverage_entries.append(
            PageCoverageEntry(page_number=page_number, status=status, reason=reason)
        )

    expected_pages = set(range(1, len(pdf_pages) + 1))
    provided_pages = set(mapped)
    return PageMappingResult(
        pages=pages,
        markdown_pages=markdown_pages,
        fallback_pages=fallback_pages,
        unresolved_pages=unresolved,
        duplicate_pages=duplicates,
        empty_pages=empty_pages,
        coverage=build_coverage_report(document_id, len(pdf_pages), coverage_entries),
        page_count_mismatch=bool(
            markdown_path
            and saw_markers
            and not trusted_converter_sequence
            and provided_pages != expected_pages
        ),
    )
