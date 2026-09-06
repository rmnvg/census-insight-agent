import re
from dataclasses import dataclass
from decimal import Decimal
from typing import cast

from backend.app.retrieval.models import RetrievedEvidence

_HTML_TAG = re.compile(r"<[^>]+>")
_MARKDOWN_DECORATION = re.compile(r"[*_`]+")
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)*(?!\w)")
_LOW_VALUE_VISUAL = re.compile(
    r"\b(?:map|graph|figure|contents|table of contents)\b", re.IGNORECASE
)


class EvidenceBudgetInsufficient(ValueError):
    """Required-target evidence cannot fit within the configured assessment budget."""


@dataclass(frozen=True)
class EvidencePackingResult:
    included: list[RetrievedEvidence]
    excluded_reasons: dict[str, str]
    reserved_by_target: dict[str, str]


def _target_for(item: RetrievedEvidence, targets: list[str]) -> str | None:
    for target in targets:
        folded = target.casefold()
        if folded in {item.region.casefold(), item.document_id.casefold()}:
            return target
    return None


def _candidate_priority(item: RetrievedEvidence) -> tuple[bool, bool, float]:
    text = _HTML_TAG.sub("", item.text)
    return (
        not contains_document_level_table_row(item),
        not (bool(_NUMBER.search(text)) and not bool(_LOW_VALUE_VISUAL.search(text))),
        -item.retrieval_score,
    )


def pack_targeted_assessment_evidence(
    evidence: list[RetrievedEvidence],
    *,
    required_targets: list[str],
    max_characters: int,
    max_chunks: int,
) -> EvidencePackingResult:
    """Reserve one complete, high-value candidate per target, then fill fairly.

    A target is never silently omitted. The selected objects are complete trusted chunks;
    text is not shortened to force it into the budget.
    """
    targets = list(dict.fromkeys(value.strip() for value in required_targets if value.strip()))
    deduplicated = list({item.chunk_id: item for item in evidence}.values())
    if not targets:
        bounded = bound_assessment_evidence(
            deduplicated, max_characters=max_characters, max_chunks=max_chunks
        )
        selected_ids = {item.chunk_id for item in bounded}
        return EvidencePackingResult(
            included=bounded,
            excluded_reasons={
                item.chunk_id: "budget_or_chunk_limit"
                for item in deduplicated
                if item.chunk_id not in selected_ids
            },
            reserved_by_target={},
        )
    if max_chunks < len(targets):
        raise EvidenceBudgetInsufficient("Chunk limit cannot reserve evidence for every target")

    groups: dict[str, list[RetrievedEvidence]] = {target: [] for target in targets}
    for item in deduplicated:
        target = _target_for(item, targets)
        if target is not None:
            groups[target].append(item)
    for items in groups.values():
        items.sort(key=_candidate_priority)

    included: list[RetrievedEvidence] = []
    included_ids: set[str] = set()
    reserved: dict[str, str] = {}
    used = 0
    for target in targets:
        if not groups[target]:
            continue
        candidate = groups[target][0]
        if used + len(candidate.text) > max_characters:
            raise EvidenceBudgetInsufficient(
                "Required-target evidence exceeds the assessment character budget"
            )
        included.append(candidate)
        included_ids.add(candidate.chunk_id)
        reserved[target] = candidate.chunk_id
        used += len(candidate.text)

    positions = {target: 1 for target in targets}
    while len(included) < max_chunks:
        progressed = False
        for target in targets:
            items = groups[target]
            position = positions[target]
            while position < len(items) and items[position].chunk_id in included_ids:
                position += 1
            positions[target] = position + 1
            if position >= len(items):
                continue
            candidate = items[position]
            if used + len(candidate.text) > max_characters:
                continue
            included.append(candidate)
            included_ids.add(candidate.chunk_id)
            used += len(candidate.text)
            progressed = True
            if len(included) >= max_chunks:
                break
        if not progressed:
            break

    excluded = {
        item.chunk_id: (
            "unrequested_target" if _target_for(item, targets) is None else "budget_or_chunk_limit"
        )
        for item in deduplicated
        if item.chunk_id not in included_ids
    }
    return EvidencePackingResult(included, excluded, reserved)


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
        items.sort(key=_candidate_priority)
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
