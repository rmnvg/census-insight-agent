"""Census-year scope: which years a question names, and which year a cited cell belongs to.

The bundled reports are Census 2011 publications whose tables print 2001 columns beside the 2011
ones. Scope follows the years the table store actually contains; a claim's year is bound to its
cell's column, so a 2001 value can never pass as a 2011 one (or the reverse).
"""

import re
from typing import Literal

from backend.app.agent.models import ClaimUnit, DraftAnswer, DraftClaim
from backend.app.agent.scopes import population_subgroups
from backend.app.tables.models import TableRecord
from backend.app.tables.selection import metric_words

# A value stated with no year is read as the report's own year; see quotes.quote_rejection_reason.
CORPUS_DEFAULT_YEAR = 2011
_YEAR = re.compile(r"(?<![\w,.])((?:18|19|20|21)\d{2})(?!\w|[,.]\d)")
_CHANGE = re.compile(
    r"\b(?:chang\w*|increas\w*|decreas\w*|grow\w*|grew|rise|rose|risen|fall|fell|fallen|"
    r"declin\w*|improv\w*|drop\w*|between|since|from)\b",
    re.IGNORECASE,
)

YearVerdict = Literal["confirmed", "contradicted"]


def named_years(text: str) -> list[int]:
    """Distinct four-digit years a text names, in order of appearance."""
    return list(dict.fromkeys(int(value) for value in _YEAR.findall(text)))


def is_year_change(query: str) -> bool:
    """A request for how one value changed between two named years."""
    return len(named_years(query)) >= 2 and bool(_CHANGE.search(query))


def years_beyond(query: str, corpus_years: set[int]) -> list[int]:
    """Named years later than every Census year the tables contain (e.g. 2021).

    Earlier years are left to evidence checks: decadal series and narrative text carry years
    (1961, 1991) that no structured table column does.
    """
    if not corpus_years:
        return []
    latest = max(corpus_years)
    return [year for year in named_years(query) if year > latest]


def table_year_verdict(claim: DraftClaim, records: list[TableRecord]) -> YearVerdict | None:
    """Whether the claim's year is the column year of the cited cell holding its value.

    Only the claim's own entity row counts. None means the store cannot tell (no such cell, or a
    matching column has no single year), and the textual checks decide alone.
    """
    if claim.value is None or not claim.region:
        return None
    region = claim.region.casefold()
    values = [
        record
        for record in records
        if record.entity.casefold() == region and abs(float(record.value) - claim.value) < 1e-9
    ]
    if not values or not claim.metric:
        return None
    metric = metric_words(claim.metric.replace("_", " "))
    residence = (claim.residence_scope or "total").casefold()
    population = (claim.population_scope or "persons").casefold()
    population = {"male": "males", "female": "females", "total": "persons"}.get(
        population, population
    )
    subgroups = population_subgroups(f"{claim.metric} {claim.population_scope or ''}")
    matches = [
        record
        for record in values
        if metric == metric_words(record.metric)
        and (record.residence or "total") == residence
        and (record.population_group or "persons") == population
        and population_subgroups(f"{record.metric} {record.title or ''}") == subgroups
    ]
    if not matches:
        return "contradicted"
    if any(record.year is None for record in matches):
        return None
    text_years = named_years(claim.text)
    if claim.year is not None and text_years and set(text_years) != {claim.year}:
        return "contradicted"
    wanted = claim.year or (text_years[0] if len(text_years) == 1 else CORPUS_DEFAULT_YEAR)
    return "confirmed" if any(record.year == wanted for record in matches) else "contradicted"


def year_cell_draft(records: list[TableRecord]) -> DraftAnswer:
    """Present selected cells; the existing comparison path adds derived arithmetic."""
    claims: list[DraftClaim] = []
    for index, cell in enumerate(records):
        unit: ClaimUnit = "percent" if cell.unit in {"percent", "%"} else "count"
        metric = cell.metric.lower()
        residence = cell.residence or "total"
        suffix = " percent" if unit == "percent" else ""
        population = (
            f" ({cell.population_group})"
            if cell.population_group and cell.population_group != "persons"
            else ""
        )
        text = (
            f"{cell.region}'s {residence} {metric}{population} in {cell.year} "
            f"was {cell.raw_value}{suffix}."
        )
        claims.append(
            DraftClaim(
                claim_id=f"year_cell_{index}",
                text=text,
                evidence_ids=[cell.chunk_id or ""],
                region=cell.region,
                metric=cell.metric,
                year=cell.year,
                value=float(cell.value),
                unit=unit,
                population_scope=cell.population_group,
                residence_scope=residence,
            )
        )
    return DraftAnswer(answer_markdown="\n\n".join(claim.text for claim in claims), claims=claims)
