import re
from decimal import Decimal
from typing import cast

from backend.app.retrieval.models import RetrievedEvidence

_HTML_TAG = re.compile(r"<[^>]+>")
_MARKDOWN_DECORATION = re.compile(r"[*_`]+")


def compatible_rounding(left: str, right: str) -> bool:
    """Return true when the less precise decimal is the rounded form of the other."""
    values = (Decimal(left), Decimal(right))
    places = tuple(max(0, -cast(int, value.as_tuple().exponent)) for value in values)
    if places[0] == places[1]:
        return values[0] == values[1]
    precise_index = 0 if places[0] > places[1] else 1
    rounded_index = 1 - precise_index
    quantum = Decimal(1).scaleb(-places[rounded_index])
    return values[precise_index].quantize(quantum) == values[rounded_index]


def contains_document_level_table_row(item: RetrievedEvidence) -> bool:
    """Identify a table row whose entity cell is the document's own region."""
    expected = item.region.strip().casefold()
    if not expected:
        return False
    for line in item.text.splitlines():
        if line.count("|") < 2:
            continue
        cells = line.strip().strip("|").split("|")
        for cell in cells:
            plain = _HTML_TAG.sub("", cell)
            plain = _MARKDOWN_DECORATION.sub("", plain).strip().casefold()
            if plain == expected:
                return True
    return False


def bound_assessment_evidence(
    evidence: list[RetrievedEvidence], *, max_characters: int, max_chunks: int
) -> list[RetrievedEvidence]:
    """Fairly select complete current-run chunks under a deterministic context budget.

    Document-level table rows are considered before ranked previews. This prevents a
    state total found by page expansion from being crowded out by several district
    rows or chart descriptions from the same document.
    """
    deduplicated = list({item.chunk_id: item for item in evidence}.values())
    groups: dict[str, list[RetrievedEvidence]] = {}
    for item in deduplicated:
        groups.setdefault(item.document_id, []).append(item)
    for items in groups.values():
        items.sort(
            key=lambda item: (
                not contains_document_level_table_row(item),
                -item.retrieval_score,
            )
        )
    selected: list[RetrievedEvidence] = []
    used = 0
    positions = {document_id: 0 for document_id in groups}
    while len(selected) < max_chunks:
        progressed = False
        for document_id, items in groups.items():
            position = positions[document_id]
            if position >= len(items):
                continue
            item = items[position]
            positions[document_id] += 1
            size = len(item.text)
            if used + size > max_characters:
                continue
            selected.append(item)
            used += size
            progressed = True
            if len(selected) >= max_chunks:
                break
        if not progressed:
            break
    return selected
