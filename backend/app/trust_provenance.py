"""Independent provenance check for the trust benchmark.

Given a claim's value, region, year, and scope, find where that value sits in the claim's own
cited quotes: which table row, and which column according to the quote's header rows. A number
merely appearing somewhere in a table quote proves nothing, because a Census table row holds a
dozen numbers for several years, residences, and groups. This check shares no code with the
agent's validators; it parses the quoted Markdown on its own.

A value is supported when some cited quote holds it on a row labelled with the claim's region
(or on a prose line naming the region), and the column headers above it, where they state a year
or residence, agree with the claim.
"""

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

_TAG = re.compile(r"<[^>]+>")
_SEPARATOR = re.compile(r"^:?-{3,}:?$")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w])")
_RESIDENCES = ("total", "rural", "urban")
_SEX_RATIO_DEFINITION = re.compile(r"females?\s+per\s+[\d,]+\s+males?")
# Population groups a column or statement can be limited to. A claim about everyone must not
# come from one of these.
_GROUPS = {
    "females": re.compile(r"\bfemales?\b"),
    "males": re.compile(r"\bmales?\b"),
    "scheduled castes": re.compile(r"\bscheduled castes?\b"),
    "scheduled tribes": re.compile(r"\bscheduled tribes?\b"),
    "children": re.compile(r"\bchild\b|\b0\s*-\s*6\b"),
}


@dataclass(frozen=True)
class Location:
    kind: str  # "table" or "prose"
    column: tuple[str, ...]  # header labels above the value, outermost first
    context: str  # the quote's section path


def normalize_label(text: str) -> str:
    return re.sub(r"[^a-z]", "", _TAG.sub(" ", text).casefold())


def _clean(cell: str) -> str:
    return " ".join(_TAG.sub(" ", cell).split())


def _cells(line: str) -> list[str]:
    return [_clean(cell) for cell in line.strip().strip("|").split("|")]


def _number(text: str) -> Decimal | None:
    compact = text.replace(",", "").replace(" ", "")
    try:
        return Decimal(compact) if re.fullmatch(r"-?\d+(\.\d+)?", compact) else None
    except InvalidOperation:
        return None


def _is_index_row(cells: list[str]) -> bool:
    return len(cells) > 2 and cells == [str(number) for number in range(1, len(cells) + 1)]


def _column_paths(header_rows: list[list[str]], width: int) -> list[tuple[str, ...]]:
    filled: list[list[str]] = []
    for row in header_rows:
        current = ""
        values = []
        for index in range(width):
            cell = row[index] if index < len(row) else ""
            current = cell or current
            values.append(current)
        filled.append(values)
    paths = []
    for index in range(width):
        labels: list[str] = []
        for row in filled:
            if row[index] and row[index] not in labels:
                labels.append(row[index])
        paths.append(tuple(labels))
    return paths


def _row_matches(cells: list[str], region: str) -> bool:
    return any(normalize_label(cell) == region for cell in cells if re.search(r"[A-Za-z]", cell))


def locate(
    value: float, region: str | None, snippet: str, context: str, tolerance: Decimal
) -> list[Location]:
    """Every place in one quote where `value` sits on a row or line naming `region`."""
    target = Decimal(str(value))
    wanted = normalize_label(region or "")
    lines = snippet.splitlines()
    table = [line for line in lines if line.strip().startswith("|")]
    found: list[Location] = []
    if table:
        rows = [_cells(line) for line in table]
        width = max(len(row) for row in rows)
        separator = next(
            (
                index
                for index, row in enumerate(rows)
                if all(_SEPARATOR.match(c) or not c for c in row)
            ),
            None,
        )
        header_rows = rows[:separator] if separator is not None else []
        body = rows[separator + 1 :] if separator is not None else rows
        data_rows: list[list[str]] = []
        for row in body:
            if _is_index_row(row):
                continue
            # Header continuations leave the code and name columns empty.
            if (
                len(row) > 2
                and not row[0]
                and not row[1]
                and separator is not None
                and not data_rows
            ):
                header_rows.append(row)
                continue
            data_rows.append(row)
        paths = _column_paths(header_rows, width)
        for row in data_rows:
            if wanted and not _row_matches(row, wanted):
                continue
            for index, cell in enumerate(row):
                number = _number(cell)
                if number is not None and abs(number - target) <= tolerance:
                    found.append(
                        Location("table", paths[index] if index < len(paths) else (), context)
                    )
    for line in lines:
        if line.strip().startswith("|"):
            continue
        if wanted and wanted not in normalize_label(line):
            continue
        for token in _NUMBER.findall(line):
            number = _number(token)
            if number is not None and abs(number - target) <= tolerance:
                found.append(Location("prose", (), context))
                break
    return found


def conflicts(
    location: Location, *, year: int | None, residence: str, group: str | None
) -> list[str]:
    """Where the quote's own headers contradict the claim's year, residence, or group."""
    problems: list[str] = []
    header = " ".join(location.column).casefold()
    years = {int(match) for match in _YEAR.findall(header)}
    if year is not None and len(years) == 1 and year not in years:
        problems.append(f"the value is in the {years.pop()} column, not {year}")
    stated = [name for name in _RESIDENCES if re.search(rf"\b{name}\b", header)]
    if len(stated) == 1 and stated[0] != residence:
        problems.append(f"the value is in the {stated[0].title()} column, not {residence}")
    scope = f"{header} {location.context.casefold()}"
    # A sex ratio is defined as "females per 1000 males"; that is not a group restriction.
    scope = _SEX_RATIO_DEFINITION.sub(" ", scope)
    groups = [name for name, pattern in _GROUPS.items() if pattern.search(scope)]
    # "females" also matches the males pattern's plural; keep the more specific label.
    if "females" in groups and "males" in groups:
        groups.remove("males")
    if group is None and groups:
        problems.append(f"the value is for {groups[0]}, not all persons")
    elif group is not None and groups and group not in groups:
        problems.append(f"the value is for {groups[0]}, not {group}")
    return problems


def support(
    claim: dict[str, Any],
    citations: list[dict[str, Any]],
    *,
    year: int | None,
    residence: str,
    group: str | None,
    tolerance: Decimal,
) -> str | None:
    """None when a cited quote independently supports the claim, else the reason it does not."""
    value = claim.get("value")
    if value is None:
        return None
    region = claim.get("region")
    locations = [
        location
        for citation in citations
        for location in locate(
            float(value),
            region,
            str(citation.get("snippet", "")),
            " > ".join(str(part) for part in citation.get("section_path", [])),
            tolerance,
        )
    ]
    if not locations:
        where = f"a row or sentence about {region}" if region else "its cited quotes"
        return f"{value:g} is not on {where}"
    reasons = [
        conflicts(location, year=year, residence=residence, group=group) for location in locations
    ]
    if any(not reason for reason in reasons):
        return None
    return reasons[0][0]
