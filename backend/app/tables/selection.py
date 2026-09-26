"""Choose the one table column a request means, deterministically, or refuse.

A column matches when its metric names exactly the requested metric (ignoring stop words, years,
and units), its table covers exactly the requested population subgroups (none, Scheduled Castes,
Scheduled Tribes, children), and its year, residence, and population group agree with the
request. Several matching columns are accepted only when they hold identical values (the Odisha
report prints its first two chapters twice); anything else is ambiguous and refused.
"""

import re

from pydantic import BaseModel

from backend.app.tables.extraction import (
    PERIOD,
    POPULATION_GROUPS,
    QUALIFIER_PATTERNS,
    RESIDENCES,
    UNIT_PARENTHETICAL,
    YEAR,
    qualifiers,
)
from backend.app.tables.models import (
    DocumentTables,
    EntityKind,
    PopulationGroup,
    Residence,
    StatementTable,
    TableColumn,
    TableRecord,
    parse_number,
)

# Every bundled report is a Census 2011 report; hydration applies the same default.
DEFAULT_YEAR = 2011
_STOP_WORDS = {
    "a",
    "among",
    "an",
    "and",
    "as",
    "by",
    "effective",
    "for",
    "in",
    "number",
    "of",
    "on",
    "the",
    "to",
    "with",
}
_DIMENSION_WORDS = set(RESIDENCES) | set(POPULATION_GROUPS)


class TableSelectionError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ColumnSelection(BaseModel):
    document_id: str
    region: str
    table_id: str
    statement: str | None
    title: str | None
    column: TableColumn
    population_group: PopulationGroup | None
    qualifiers: list[str]
    records: list[TableRecord]
    aggregate: TableRecord | None = None


def _singular(word: str) -> str:
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def metric_words(text: str) -> frozenset[str]:
    for _, pattern in QUALIFIER_PATTERNS:
        text = pattern.sub(" ", text)
    text = YEAR.sub(" ", PERIOD.sub(" ", UNIT_PARENTHETICAL.sub(" ", text)))
    words = {_singular(word) for word in re.findall(r"[a-z0-9]+", text.casefold())}
    return frozenset(words - _STOP_WORDS - {_singular(word) for word in _DIMENSION_WORDS})


def _requested_residence(residence: str | None, metric: str) -> tuple[Residence, bool]:
    """The requested residence, and whether it narrows the default (total)."""
    words = re.findall(r"[a-z]+", (residence or metric).casefold())
    named = {RESIDENCES[word] for word in words if word in RESIDENCES}
    if len(named) == 1:
        value = named.pop()
        return value, value != "total"
    return "total", False


def _requested_group(population: str | None, metric: str) -> tuple[PopulationGroup, bool]:
    text = f"{population or ''} {metric}".casefold()
    # A sex-ratio unit ("females per 1000 males") sometimes arrives as the population scope.
    text = re.sub(r"\b\w+\s+per\s+1,?000\s+\w+", " ", text)
    words = set(re.findall(r"[a-z]+", text))
    if words & {"female", "females", "women", "girls"}:
        return "females", True
    if words & {"male", "males", "men", "boys"}:
        return "males", True
    return "persons", False


def _column_selection(
    document: DocumentTables, table: StatementTable, column: TableColumn
) -> ColumnSelection:
    if table.malformed_rows:
        raise TableSelectionError("TABLE_ROWS_MALFORMED")
    population = column.population_group or table.population_group
    records: list[TableRecord] = []
    aggregate: TableRecord | None = None
    for row in table.rows:
        raw = row.cells[column.index]
        value = parse_number(raw)
        if value is None or row.chunk_id is None:
            if row.aggregate:
                continue
            # Ranking over a column with a blank or uncitable district row would be incomplete.
            raise TableSelectionError("TABLE_COLUMN_INCOMPLETE")
        record = TableRecord(
            document_id=document.document_id,
            region=document.region,
            table_id=table.table_id,
            statement=table.statement,
            title=table.title,
            entity=row.label,
            entity_code=row.code,
            aggregate=row.aggregate,
            metric=column.metric or "",
            year=column.year,
            period=column.period,
            residence=column.residence,
            population_group=population,
            qualifiers=table.qualifiers,
            unit=column.unit,
            raw_value=raw,
            value=value,
            page_number=row.page_number,
            chunk_id=row.chunk_id,
            line=row.line,
        )
        if row.aggregate:
            aggregate = record
        else:
            records.append(record)
    if len(records) < 2:
        raise TableSelectionError("TABLE_TOO_FEW_ENTITIES")
    return ColumnSelection(
        document_id=document.document_id,
        region=document.region,
        table_id=table.table_id,
        statement=table.statement,
        title=table.title,
        column=column,
        population_group=population,
        qualifiers=table.qualifiers,
        records=records,
        aggregate=aggregate,
    )


def select_column(
    document: DocumentTables,
    *,
    metric: str,
    year: int | None = None,
    residence: str | None = None,
    population: str | None = None,
    entity_kind: EntityKind | None = "district",
) -> ColumnSelection:
    wanted = metric_words(metric)
    if not wanted:
        raise TableSelectionError("TABLE_METRIC_UNRECOGNIZED")
    wanted_qualifiers = set(qualifiers(f"{metric} {population or ''}"))
    wanted_year = year or DEFAULT_YEAR
    wanted_residence, residence_explicit = _requested_residence(residence, metric)
    wanted_group, group_explicit = _requested_group(population, metric)
    exact: list[tuple[StatementTable, TableColumn]] = []
    broader: list[tuple[StatementTable, TableColumn]] = []
    for table in document.tables:
        if entity_kind is not None and table.entity_kind != entity_kind:
            continue
        for column in table.value_columns():
            if column.metric is None or column.year != wanted_year:
                continue
            if set(table.qualifiers) | set(qualifiers(column.metric)) != wanted_qualifiers:
                continue
            words = metric_words(column.metric)
            if not wanted <= words:
                continue
            if column.residence is None:
                if residence_explicit:
                    continue
            elif column.residence != wanted_residence:
                continue
            group = column.population_group or table.population_group
            if group is None:
                # Sex ratio has no persons/males/females split: the metric relates the two.
                if group_explicit:
                    continue
            elif group != wanted_group:
                continue
            (exact if words == wanted else broader).append((table, column))
    # "literacy" may name "Literacy Rate", but only when no column is named exactly and just one
    # distinct metric is a superset of the request.
    candidates = exact or broader
    if not candidates:
        raise TableSelectionError("TABLE_NO_MATCHING_COLUMN")
    if len({metric_words(column.metric or "") for _, column in candidates}) != 1:
        raise TableSelectionError("TABLE_AMBIGUOUS_COLUMN")
    selections = [_column_selection(document, table, column) for table, column in candidates]
    signatures = {
        tuple((record.entity.casefold(), record.raw_value) for record in selection.records)
        for selection in selections
    }
    if len(signatures) != 1:
        raise TableSelectionError("TABLE_AMBIGUOUS_COLUMN")
    return selections[0]
