import json
from typing import Any

from pydantic import BaseModel

PROPOSAL_SCHEMA_MAX_CHARACTERS = 2_500
PROPOSAL_SCHEMA_MAX_DEFINITIONS = 2
PROPOSAL_SCHEMA_MAX_DEPTH = 10


def _depth(value: object) -> int:
    if isinstance(value, dict):
        return 1 + max((_depth(item) for item in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((_depth(item) for item in value), default=0)
    return 1


def _locations(value: object, keys: set[str], path: str = "$") -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            if key in keys:
                found.append({"path": child, "value": item})
            found.extend(_locations(item, keys, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_locations(item, keys, f"{path}[{index}]"))
    return found


def schema_complexity(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    serialized = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    descriptions = _locations(schema, {"description"})
    return {
        "model": model.__name__,
        "serialized_characters": len(serialized),
        "definition_count": len(schema.get("$defs", {})),
        "definitions": sorted(schema.get("$defs", {})),
        "maximum_nesting_depth": _depth(schema),
        "formats": _locations(schema, {"format"}),
        "patterns": _locations(schema, {"pattern"}),
        "numeric_constraints": _locations(
            schema, {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}
        ),
        "array_constraints": _locations(schema, {"minItems", "maxItems"}),
        "enums": _locations(schema, {"enum"}),
        "description_count": len(descriptions),
        "longest_description_characters": max(
            (len(str(item["value"])) for item in descriptions), default=0
        ),
    }


def assert_proposal_schema_safe(model: type[BaseModel]) -> dict[str, Any]:
    report = schema_complexity(model)
    serialized = json.dumps(model.model_json_schema(), sort_keys=True).casefold()
    forbidden = (
        "checksum",
        "sourcemanifest",
        "executionrequest",
        "executionresult",
        'artifactdataset"',
        "artifact_type",
        "date-time",
        "uuid",
        "filesystem",
        "filename",
    )
    violations = [value for value in forbidden if value in serialized]
    if violations:
        raise ValueError(f"Proposal schema contains forbidden internal concepts: {violations}")
    if report["formats"] or report["patterns"]:
        raise ValueError("Proposal schema contains format or pattern constraints")
    if report["serialized_characters"] >= PROPOSAL_SCHEMA_MAX_CHARACTERS:
        raise ValueError("Proposal schema exceeds the conservative character limit")
    if report["definition_count"] > PROPOSAL_SCHEMA_MAX_DEFINITIONS:
        raise ValueError("Proposal schema exceeds the conservative definition limit")
    if report["maximum_nesting_depth"] > PROPOSAL_SCHEMA_MAX_DEPTH:
        raise ValueError("Proposal schema exceeds the conservative nesting limit")
    return report
