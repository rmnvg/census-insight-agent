from collections.abc import Iterator
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, Field

TABLE_STORE_VERSION = 1

Residence = Literal["total", "rural", "urban"]
PopulationGroup = Literal["persons", "males", "females"]
EntityKind = Literal["district", "state"]


class TableColumn(BaseModel):
    """One printed column, with the meaning its full header path gives it."""

    index: int = Field(ge=0)
    number: str
    header_path: list[str]
    metric: str | None = None
    year: int | None = None
    period: str | None = None
    residence: Residence | None = None
    population_group: PopulationGroup | None = None
    unit: str | None = None


class TableRow(BaseModel):
    label: str
    code: str | None = None
    aggregate: bool = False
    page_number: int = Field(gt=0)
    # None when the row's line is not in exactly one indexed chunk on its page; such a row can
    # never be cited, so no selection may use it.
    chunk_id: str | None = None
    line: str
    cells: list[str]


class StatementTable(BaseModel):
    table_id: str
    statement: str | None = None
    title: str | None = None
    population_group: PopulationGroup | None = None
    qualifiers: list[str] = Field(default_factory=list)
    entity_kind: EntityKind | None = None
    label_column: int = Field(ge=0)
    code_column: int | None = None
    columns: list[TableColumn]
    rows: list[TableRow]
    footnotes: list[str] = Field(default_factory=list)
    page_numbers: list[int]
    malformed_rows: int = 0

    def value_columns(self) -> list[TableColumn]:
        return [
            column
            for column in self.columns
            if column.index not in {self.label_column, self.code_column}
        ]


class DocumentTables(BaseModel):
    version: int = TABLE_STORE_VERSION
    document_id: str
    region: str
    source_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    tables: list[StatementTable]


class TableRecord(BaseModel):
    """One cell as a self-describing fact: entity, metric, dimensions, value, and source span."""

    document_id: str
    region: str
    table_id: str
    statement: str | None
    title: str | None
    entity: str
    entity_code: str | None
    aggregate: bool
    metric: str
    year: int | None
    period: str | None
    residence: Residence | None
    population_group: PopulationGroup | None
    qualifiers: list[str]
    unit: str | None
    raw_value: str
    value: Decimal
    page_number: int
    chunk_id: str | None
    line: str


def parse_number(raw: str) -> Decimal | None:
    """A cell's value when the whole cell is a number ("4,06,47,322", "86.24", "-3.5")."""
    text = raw.replace(",", "").strip()
    if not text or not any(character.isdigit() for character in text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def table_records(document: DocumentTables) -> Iterator[TableRecord]:
    for table in document.tables:
        population = table.population_group
        for column in table.value_columns():
            if column.metric is None:
                continue
            for row in table.rows:
                raw = row.cells[column.index]
                value = parse_number(raw)
                if value is None:
                    continue
                yield TableRecord(
                    document_id=document.document_id,
                    region=document.region,
                    table_id=table.table_id,
                    statement=table.statement,
                    title=table.title,
                    entity=row.label,
                    entity_code=row.code,
                    aggregate=row.aggregate,
                    metric=column.metric,
                    year=column.year,
                    period=column.period,
                    residence=column.residence,
                    population_group=column.population_group or population,
                    qualifiers=table.qualifiers,
                    unit=column.unit,
                    raw_value=raw,
                    value=value,
                    page_number=row.page_number,
                    chunk_id=row.chunk_id,
                    line=row.line,
                )
