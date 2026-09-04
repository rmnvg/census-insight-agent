import re
import uuid
from dataclasses import dataclass

from backend.app.ingestion.models import ChunkMetadata, DocumentChunk, DocumentPage
from backend.app.ingestion.safety import UnsafeExtractionError, validate_page_for_chunking

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}")
CHUNK_NAMESPACE = uuid.UUID("8f125211-267f-47f0-a9dd-751177dcc485")


@dataclass(frozen=True)
class ChunkingConfig:
    max_characters: int = 1600
    overlap_characters: int = 200
    citation_characters: int = 240

    def __post_init__(self) -> None:
        if self.max_characters <= 0:
            raise ValueError("max_characters must be positive")
        if self.overlap_characters < 0 or self.overlap_characters >= self.max_characters:
            raise ValueError("overlap_characters must be between zero and max_characters")


DEFAULT_CHUNKING_CONFIG = ChunkingConfig()


def _is_table(block: str) -> bool:
    lines = [line for line in block.splitlines() if line.strip()]
    return len(lines) >= 2 and "|" in lines[0] and bool(TABLE_SEPARATOR.match(lines[1]))


def _split_blocks(markdown: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", markdown) if block.strip()]


def _table_parts(table: str, max_characters: int) -> list[str]:
    if len(table) <= max_characters:
        return [table]
    lines = table.splitlines()
    header = lines[:2]
    rows = lines[2:]
    parts: list[str] = []
    current = header.copy()
    for row in rows:
        candidate = "\n".join([*current, row])
        if len(candidate) > max_characters and len(current) > len(header):
            parts.append("\n".join(current))
            current = [*header, row]
        else:
            current.append(row)
    if current:
        parts.append("\n".join(current))
    return parts


def _prose_parts(text: str, config: ChunkingConfig) -> list[str]:
    if len(text) <= config.max_characters:
        return [text]
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    parts: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > config.max_characters:
            sentences: list[str] = []
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
                if len(sentence) <= config.max_characters:
                    sentences.append(sentence)
                    continue
                remaining = sentence
                while len(remaining) > config.max_characters:
                    split_at = remaining.rfind(" ", 0, config.max_characters)
                    split_at = split_at if split_at > 0 else config.max_characters
                    sentences.append(remaining[:split_at].rstrip())
                    overlap_start = max(0, split_at - config.overlap_characters)
                    if overlap_start == 0:
                        overlap_start = split_at
                    remaining = remaining[overlap_start:].lstrip()
                if remaining:
                    sentences.append(remaining)
        else:
            sentences = [paragraph]
        for sentence in sentences:
            candidate = f"{current}\n\n{sentence}".strip()
            if current and len(candidate) > config.max_characters:
                parts.append(current)
                overlap = current[-config.overlap_characters :].lstrip()
                current = f"{overlap}\n\n{sentence}".strip()
            else:
                current = candidate
    if current:
        parts.append(current)
    return parts


def _citation_snippet(final_text: str, body: str, limit: int) -> tuple[str, str | None]:
    """Select an exact contiguous fallback citation from the final stored chunk."""
    stripped_body = body.strip()
    start = final_text.find(stripped_body) if stripped_body else -1
    limitation = None
    if start < 0:
        start = 0
        limitation = "No distinct non-heading evidence region was available"
    available = final_text[start:]
    if len(available) <= limit:
        return available, limitation

    end = start + limit
    if _is_table(stripped_body):
        line_end = final_text.rfind("\n", start, end + 1)
        if line_end > start:
            end = line_end
    else:
        boundary = max(
            final_text.rfind("\n", start, end + 1), final_text.rfind(" ", start, end + 1)
        )
        if boundary > start:
            end = boundary
    return final_text[start:end], limitation


def chunk_pages(
    pages: list[DocumentPage],
    *,
    document_title: str,
    region: str,
    source_checksum: str,
    config: ChunkingConfig = DEFAULT_CHUNKING_CONFIG,
) -> list[DocumentChunk]:
    """Create deterministic page-bounded chunks with heading context."""
    chunks: list[DocumentChunk] = []
    chunk_index = 0
    section_path: list[str] = []
    for page in pages:
        validate_page_for_chunking(page)
        if not page.text.strip():
            continue
        prose_buffer: list[str] = []

        for block in _split_blocks(page.text):
            heading_match = HEADING.match(block)
            if heading_match:
                chunk_index = _emit_prose(
                    chunks,
                    prose_buffer,
                    page=page,
                    section_path=section_path,
                    chunk_index=chunk_index,
                    document_title=document_title,
                    region=region,
                    source_checksum=source_checksum,
                    config=config,
                )
                level = len(heading_match.group(1))
                heading = heading_match.group(2).strip()
                section_path[level - 1 :] = [heading]
                continue
            if _is_table(block):
                chunk_index = _emit_prose(
                    chunks,
                    prose_buffer,
                    page=page,
                    section_path=section_path,
                    chunk_index=chunk_index,
                    document_title=document_title,
                    region=region,
                    source_checksum=source_checksum,
                    config=config,
                )
                available = max(200, config.max_characters - len(" > ".join(section_path)) - 2)
                for table_part in _table_parts(block, available):
                    chunks.append(
                        _make_chunk(
                            page=page,
                            body=table_part,
                            section_path=section_path,
                            chunk_index=chunk_index,
                            document_title=document_title,
                            region=region,
                            source_checksum=source_checksum,
                            citation_limit=config.citation_characters,
                        )
                    )
                    chunk_index += 1
            else:
                prose_buffer.append(block)
        chunk_index = _emit_prose(
            chunks,
            prose_buffer,
            page=page,
            section_path=section_path,
            chunk_index=chunk_index,
            document_title=document_title,
            region=region,
            source_checksum=source_checksum,
            config=config,
        )
    return chunks


def _emit_prose(
    chunks: list[DocumentChunk],
    prose_buffer: list[str],
    *,
    page: DocumentPage,
    section_path: list[str],
    chunk_index: int,
    document_title: str,
    region: str,
    source_checksum: str,
    config: ChunkingConfig,
) -> int:
    if not prose_buffer:
        return chunk_index
    prose = "\n\n".join(prose_buffer)
    for part in _prose_parts(prose, config):
        chunks.append(
            _make_chunk(
                page=page,
                body=part,
                section_path=section_path,
                chunk_index=chunk_index,
                document_title=document_title,
                region=region,
                source_checksum=source_checksum,
                citation_limit=config.citation_characters,
            )
        )
        chunk_index += 1
    prose_buffer.clear()
    return chunk_index


def _make_chunk(
    *,
    page: DocumentPage,
    body: str,
    section_path: list[str],
    chunk_index: int,
    document_title: str,
    region: str,
    source_checksum: str,
    citation_limit: int,
) -> DocumentChunk:
    if page.extraction_method not in {
        "provided_markdown",
        "pymupdf4llm_fallback",
        "approved_manual_transcription",
    }:
        raise UnsafeExtractionError(
            f"Page {page.page_number} does not have an approved extraction method"
        )
    heading_context = " > ".join(section_path)
    text = f"{heading_context}\n\n{body}".strip() if heading_context else body.strip()
    stable_key = f"{page.document_id}:{page.page_number}:{chunk_index}:{text}"
    chunk_id = str(uuid.uuid5(CHUNK_NAMESPACE, stable_key))
    citation_snippet, citation_limitation = _citation_snippet(text, body, citation_limit)
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            document_id=page.document_id,
            document_title=document_title,
            region=region,
            page_number=page.page_number,
            section_path=list(section_path),
            chunk_index=chunk_index,
            citation_snippet=citation_snippet,
            citation_limitation=citation_limitation,
            extraction_method=page.extraction_method,
            coverage_status=page.coverage_status,
            source_checksum=source_checksum,
        ),
    )
