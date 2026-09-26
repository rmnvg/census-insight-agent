"""Parse Census statement tables from indexed pages and bind each row to its chunk.

Only tables with a printed column-number row ("1 | 2 | 3 ...") are extracted: every statement
table in these reports has one, and graph data transcribed from images does not. The Markdown
table is complete where the chunker splits it, so headers are resolved here once, from the whole
table, instead of from whichever fragment a retrieval happened to return.

A blank header cell continues the cell to its left, but only under the same parent header, so
"Literacy Rate | | 2001 | | | 2011" resolves every column to its own metric, year, and
residence. A row is usable only when its exact line occurs in exactly one indexed chunk on its
page; that chunk is what an answer cites.
"""

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from backend.app.ingestion.models import DocumentPage
from backend.app.tables.models import (
    DocumentTables,
    EntityKind,
    PopulationGroup,
    Residence,
    StatementTable,
    TableColumn,
    TableRow,
)

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}")
_STATEMENT = re.compile(r"^statement\s*[-–—]?\s*(\d+)\b\s*[:.–—-]?\s*(.*)$", re.IGNORECASE)
# Graphs, maps, and annexures follow a statement's table; their content is not the statement's.
_RESET_PARAGRAPH = re.compile(r"^(?:graph|map|figure|chart|annexure)\b", re.IGNORECASE)
_FOOTNOTE = re.compile(r"^(?:notes?\s*:|n\.\s?b\.)", re.IGNORECASE)
_GROUP_WORD = re.compile(r"\b(persons|males|females)\b", re.IGNORECASE)
_HTML = re.compile(r"<[^>]+>")
_MARKDOWN = re.compile(r"[*_`]+")
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
PERIOD = re.compile(r"\b((?:19|20)\d{2})\s*[-–—]\s*((?:19|20)?\d{2})\b")
_TITLE_GROUP = re.compile(r"\(\s*(persons?|males?|females?)\s*\)\s*$", re.IGNORECASE)
UNIT_PARENTHETICAL = re.compile(r"\(([^()]*\bper\b[^()]*)\)", re.IGNORECASE)
QUALIFIER_PATTERNS = (
    ("scheduled castes", re.compile(r"\bscheduled\s+castes?\b", re.IGNORECASE)),
    ("scheduled tribes", re.compile(r"\bscheduled\s+tribes?\b", re.IGNORECASE)),
    ("children 0-6", re.compile(r"\bchild(?:ren)?\b|\(?\b0\s*-\s*6\s*years?\b\)?", re.IGNORECASE)),
)
RESIDENCES: dict[str, Residence] = {"total": "total", "rural": "rural", "urban": "urban"}
POPULATION_GROUPS: dict[str, PopulationGroup] = {
    "persons": "persons",
    "person": "persons",
    "males": "males",
    "male": "males",
    "females": "females",
    "female": "females",
}
_MAX_CONTEXT_CHARACTERS = 300


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    page_number: int
    text: str


def plain(value: str) -> str:
    return " ".join(_MARKDOWN.sub("", _HTML.sub(" ", value)).split())


def qualifiers(text: str) -> list[str]:
    """Population subgroups a text names; a table for one is not the whole-population table."""
    return [name for name, pattern in QUALIFIER_PATTERNS if pattern.search(text)]


def _italic(block: str) -> bool:
    stripped = block.strip()
    return stripped.startswith("*") and not stripped.startswith("**") and stripped.endswith("*")


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _blocks(text: str) -> list[str]:
    # Identical to the chunker's split, so each row line is byte-for-byte the chunk's line.
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]


def _is_table(block: str) -> bool:
    lines = [line for line in block.splitlines() if line.strip()]
    return len(lines) >= 2 and "|" in lines[0] and bool(_SEPARATOR.match(lines[1]))


def _is_separator(cells: list[str]) -> bool:
    return all(not cell or set(cell) <= {"-", ":"} for cell in cells)


def _is_column_numbers(cells: list[str]) -> bool:
    return len(cells) >= 2 and [plain(cell) for cell in cells] == [
        str(number) for number in range(1, len(cells) + 1)
    ]


def _header_paths(header_rows: list[list[str]], width: int) -> list[list[str]]:
    filled: list[list[str]] = []
    for row in header_rows:
        values = [plain(row[index]) if index < len(row) else "" for index in range(width)]
        resolved: list[str] = []
        for index, value in enumerate(values):
            if not value and index > 0:
                parent = [level[index] for level in filled]
                left_parent = [level[index - 1] for level in filled]
                if parent == left_parent:
                    value = resolved[index - 1]
            resolved.append(value)
        filled.append(resolved)
    return [[level[index] for level in filled if level[index]] for index in range(width)]


def _period(match: re.Match[str]) -> str:
    start, end = match.group(1), match.group(2)
    return f"{start}-{end if len(end) == 4 else start[:2] + end}"


def _single_year(text: str) -> int | None:
    if PERIOD.search(text):
        return None
    years = set(YEAR.findall(text))
    return int(years.pop()) if len(years) == 1 else None


def _unit(metric: str, title: str | None) -> str:
    for text in (metric, title or ""):
        if match := UNIT_PARENTHETICAL.search(text):
            unit = re.sub(r"^\s*number\s+of\s+", "", match.group(1), flags=re.IGNORECASE)
            return " ".join(unit.split())
    folded = metric.casefold()
    if "sex ratio" in folded:
        return "girls per 1000 boys" if qualifiers(metric) else "females per 1000 males"
    if re.search(r"\b(?:rate|percentage|proportion|per\s*cent|decadal)\b", folded):
        return "percent"
    return "persons"


def _column(index: int, number: str, path: list[str], title: str | None) -> TableColumn:
    residence: Residence | None = None
    group: PopulationGroup | None = None
    years: list[int] = []
    periods: list[str] = []
    parts: list[str] = []
    for part in path:
        folded = part.casefold()
        if folded in RESIDENCES:
            residence = RESIDENCES[folded]
            continue
        if folded in POPULATION_GROUPS:
            group = POPULATION_GROUPS[folded]
            continue
        # "Scheduled Tribe population<br>Males" puts the group inside the metric's own cell.
        named = set(_GROUP_WORD.findall(UNIT_PARENTHETICAL.sub(" ", folded)))
        if len(named) == 1:
            group = POPULATION_GROUPS[named.pop()]
            part = _GROUP_WORD.sub(" ", part)
        periods.extend(_period(match) for match in PERIOD.finditer(part))
        text = PERIOD.sub(" ", part)
        years.extend(int(year) for year in YEAR.findall(text))
        text = " ".join(YEAR.sub(" ", text).split()).strip(" -–—:/,")
        if text:
            parts.append(text)
    raw_metric = " ".join(parts)
    metric = " ".join(UNIT_PARENTHETICAL.sub(" ", raw_metric).split()) or None
    year: int | None = None
    period: str | None = None
    if len(set(years)) == 1 and not periods:
        year = years[0]
    elif len(set(periods)) == 1 and not years:
        period = periods[0]
    elif not years and not periods and title:
        # Columns whose header names no year take the title's single year ("... : 2011").
        year = _single_year(title)
    return TableColumn(
        index=index,
        number=number,
        header_path=path,
        metric=metric,
        year=year,
        period=period,
        residence=residence,
        population_group=group,
        unit=_unit(raw_metric, title) if metric else None,
    )


def _entity_kind(header: str) -> EntityKind | None:
    if "district" in header:
        return "district"
    if re.search(r"\b(?:state|india|union territory)\b", header):
        return "state"
    return None


def _parse_table(
    block: str,
    *,
    table_id: str,
    page_number: int,
    statement: str | None,
    title: str | None,
    region: str,
    chunks_by_line: dict[tuple[int, str], set[str]],
) -> StatementTable | None:
    lines = [line for line in block.splitlines() if line.strip()]
    rows = [_cells(line) for line in lines]
    number_row = next((index for index, row in enumerate(rows) if _is_column_numbers(row)), None)
    if number_row is None:
        return None
    width = len(rows[number_row])
    header_rows = [row for row in rows[:number_row] if not _is_separator(row)]
    paths = _header_paths(header_rows, width)
    headers = [" ".join(path).casefold() for path in paths]
    code_column = next(
        (index for index, header in enumerate(headers) if re.search(r"\bcode\b", header)), None
    )
    label_column = next(
        (
            index
            for index, header in enumerate(headers)
            if index != code_column and re.search(r"district|state|india|union territory", header)
        ),
        None,
    )
    if label_column is None:
        return None
    columns = [
        _column(index, str(index + 1), path, title)
        if index not in {label_column, code_column}
        else TableColumn(index=index, number=str(index + 1), header_path=path)
        for index, path in enumerate(paths)
    ]
    parsed: list[TableRow] = []
    malformed = 0
    region_folded = region.casefold()
    for line, cells in zip(lines[number_row + 1 :], rows[number_row + 1 :], strict=True):
        if len(cells) != width:
            malformed += 1
            continue
        label = plain(cells[label_column])
        if not label:
            malformed += 1
            continue
        code = plain(cells[code_column]) if code_column is not None else ""
        chunk_ids = chunks_by_line.get((page_number, line), set())
        parsed.append(
            TableRow(
                label=label,
                code=code or None,
                aggregate=label.casefold() in {region_folded, "india"},
                page_number=page_number,
                chunk_id=next(iter(chunk_ids)) if len(chunk_ids) == 1 else None,
                line=line,
                cells=[plain(cell) for cell in cells],
            )
        )
    title_group = _TITLE_GROUP.search(title or "")
    return StatementTable(
        table_id=table_id,
        statement=statement,
        title=title,
        population_group=POPULATION_GROUPS[title_group.group(1).casefold()]
        if title_group
        else None,
        qualifiers=qualifiers(title or ""),
        entity_kind=_entity_kind(headers[label_column]),
        label_column=label_column,
        code_column=code_column,
        columns=columns,
        rows=parsed,
        page_numbers=[page_number],
        malformed_rows=malformed,
    )


def extract_document_tables(
    *,
    document_id: str,
    region: str,
    source_checksum: str,
    pages: list[DocumentPage],
    chunks: Iterable[IndexedChunk],
) -> DocumentTables:
    """Every column-numbered table on the indexed pages, with rows bound to `chunks`."""
    chunks_by_line: dict[tuple[int, str], set[str]] = defaultdict(set)
    for chunk in chunks:
        for line in chunk.text.splitlines():
            if line.count("|") >= 2:
                chunks_by_line[(chunk.page_number, line)].add(chunk.chunk_id)
    tables: list[StatementTable] = []
    statement: str | None = None
    title: str | None = None
    awaiting_title = False
    heading_title: str | None = None
    current: StatementTable | None = None
    for page in sorted(pages, key=lambda item: item.page_number):
        for block in _blocks(page.text):
            if _is_table(block):
                table = _parse_table(
                    block,
                    table_id=f"t{len(tables) + 1:03d}-p{page.page_number}",
                    page_number=page.page_number,
                    statement=statement,
                    title=title if statement else heading_title,
                    region=region,
                    chunks_by_line=chunks_by_line,
                )
                awaiting_title = False
                if table is None:
                    continue
                continues = (
                    current is not None
                    and statement is not None
                    and current.statement == statement
                    and [column.header_path for column in current.columns]
                    == [column.header_path for column in table.columns]
                )
                if continues and current is not None:
                    # The same statement's table carried onto another page.
                    current.rows.extend(table.rows)
                    current.malformed_rows += table.malformed_rows
                    if page.page_number not in current.page_numbers:
                        current.page_numbers.append(page.page_number)
                else:
                    tables.append(table)
                    current = table
                continue
            heading = _HEADING.match(block)
            text = plain(heading.group(2) if heading else block)
            if not text:
                continue
            short = len(text) <= _MAX_CONTEXT_CHARACTERS
            if short and (match := _STATEMENT.match(text)):
                statement = match.group(1)
                title = match.group(2).strip(" :.–—-") or None
                awaiting_title = title is None
                current = None
            elif awaiting_title and short:
                title = text
                awaiting_title = False
            elif heading or _RESET_PARAGRAPH.match(text):
                statement = title = None
                awaiting_title = False
                current = None
                heading_title = text if heading else None
            elif current is not None and short and (_FOOTNOTE.match(text) or _italic(block)):
                current.footnotes.append(text)
    return DocumentTables(
        document_id=document_id,
        region=region,
        source_checksum=source_checksum,
        tables=tables,
    )
