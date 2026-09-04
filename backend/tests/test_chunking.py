from backend.app.ingestion.chunking import ChunkingConfig, chunk_pages
from backend.app.ingestion.models import DocumentPage


def sample_page(page_number: int, text: str) -> DocumentPage:
    return DocumentPage(
        document_id="doc",
        page_number=page_number,
        text=text,
        extraction_method="provided_markdown",
        coverage_status="indexed_provided_markdown",
    )


def test_chunks_never_span_pages_and_ids_are_deterministic() -> None:
    pages = [sample_page(1, "First page evidence."), sample_page(2, "Second page evidence.")]

    first = chunk_pages(pages, document_title="Doc", region="Region", source_checksum="sum")
    second = chunk_pages(pages, document_title="Doc", region="Region", source_checksum="sum")

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert {chunk.metadata.page_number for chunk in first} == {1, 2}
    assert all("First" not in chunk.text for chunk in first if chunk.metadata.page_number == 2)


def test_markdown_table_is_preserved_when_it_fits() -> None:
    table = "| District | Population |\n|---|---:|\n| Mysuru | 100 |\n| Kodagu | 50 |"
    chunks = chunk_pages(
        [sample_page(1, f"# Population\n\n{table}")],
        document_title="Doc",
        region="Karnataka",
        source_checksum="sum",
    )

    assert len(chunks) == 1
    assert table in chunks[0].text


def test_table_splits_only_between_rows() -> None:
    table = "| District | Value |\n|---|---|\n" + "\n".join(
        f"| District {index} | {index} |" for index in range(20)
    )
    chunks = chunk_pages(
        [sample_page(1, table)],
        document_title="Doc",
        region="Region",
        source_checksum="sum",
        config=ChunkingConfig(max_characters=180, overlap_characters=20),
    )

    assert len(chunks) > 1
    assert all(chunk.text.splitlines()[1] == "|---|---|" for chunk in chunks)
    assert all(line.count("|") >= 3 for chunk in chunks for line in chunk.text.splitlines())


def test_citation_snippet_is_verbatim_non_heading_evidence() -> None:
    chunks = chunk_pages(
        [sample_page(1, "# Heading\n\nThe district population is exactly 150 residents.")],
        document_title="Doc",
        region="Region",
        source_checksum="sum",
    )

    snippet = chunks[0].metadata.citation_snippet
    assert snippet in chunks[0].text
    assert "exactly 150 residents" in snippet


def test_citation_preserves_repeated_whitespace_newlines_and_unicode() -> None:
    evidence = "District  A  has  value  10.\n\n‘Unicode’ — evidence remains exact."
    chunk = chunk_pages(
        [sample_page(1, evidence)],
        document_title="Doc",
        region="Region",
        source_checksum="sum",
    )[0]

    assert chunk.metadata.citation_snippet == evidence
    assert chunk.metadata.citation_snippet in chunk.text


def test_markdown_table_citation_preserves_pipes_and_complete_rows() -> None:
    table = "| District | Value |\n|---|---:|\n| Mysuru | 100 |\n| Kodagu | 50 |"
    chunk = chunk_pages(
        [sample_page(1, table)],
        document_title="Doc",
        region="Region",
        source_checksum="sum",
        config=ChunkingConfig(max_characters=1600, citation_characters=55),
    )[0]

    snippet = chunk.metadata.citation_snippet
    assert snippet in chunk.text
    assert snippet == "| District | Value |\n|---|---:|\n| Mysuru | 100 |"
    assert snippet.count("|") == 9


def test_long_citation_is_a_direct_slice_without_generated_ellipsis() -> None:
    evidence = "Evidence " + "word  " * 100
    chunk = chunk_pages(
        [sample_page(1, evidence)],
        document_title="Doc",
        region="Region",
        source_checksum="sum",
    )[0]

    snippet = chunk.metadata.citation_snippet
    assert len(snippet) <= 240
    assert snippet in chunk.text
    assert "…" not in snippet
    assert not snippet.endswith("...")


def test_heading_is_skipped_when_followed_by_evidence() -> None:
    chunk = chunk_pages(
        [sample_page(1, "# Population\n\nExact district evidence")],
        document_title="Doc",
        region="Region",
        source_checksum="sum",
    )[0]

    assert chunk.text.startswith("Population\n\n")
    assert chunk.metadata.citation_snippet == "Exact district evidence"
