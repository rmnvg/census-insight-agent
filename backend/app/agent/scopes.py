import re

from backend.app.agent.models import ArtifactDataRequirement

_RESIDENCE = ("rural", "urban", "total")
_POPULATION = ("female", "male", "persons")
# "Total" is a residence column; as a population scope it only means "everyone", which is the
# corpus default. Passing it through made hydration look for a nonexistent "Total" population
# category (live: a district sex-ratio ranking that had succeeded began refusing).
_UNSPECIFIED_POPULATION = re.compile(
    r"^\s*(?:total|all|all persons|overall|both sexes|everyone|general)\s*$", re.IGNORECASE
)
_METRIC_ALIASES = {
    "literacy rate": "literacy rate",
    "literacy rates": "literacy rate",
    "effective literacy rate": "literacy rate",
    "effective literacy rates": "literacy rate",
    "gender ratio": "sex ratio",
    "gender ratios": "sex ratio",
}


def _explicit(value: str, choices: tuple[str, ...]) -> str | None:
    matches = [choice for choice in choices if re.search(rf"\b{choice}\b", value)]
    return matches[0] if len(matches) == 1 else None


def canonical_scopes(
    query: str,
    population_scope: str | None,
    residence_scope: str | None,
) -> tuple[str | None, str | None]:
    """Resolve independent Census population/sex and residence dimensions."""
    folded_query = query.casefold()
    folded_population = (population_scope or "").casefold()
    folded_residence = (residence_scope or "").casefold()

    explicit_population = _explicit(folded_query, _POPULATION)
    provided_population = _explicit(folded_population, _POPULATION)
    population_category = explicit_population or provided_population
    population: str | None
    if population_category == "female":
        population = "female"
    elif population_category == "male":
        population = "male"
    elif population_category == "persons":
        population = "persons"
    elif population_scope is not None and _UNSPECIFIED_POPULATION.match(population_scope):
        population = None
    else:
        population = population_scope

    explicit_residence = _explicit(folded_query, _RESIDENCE)
    provided_residence = _explicit(folded_residence, _RESIDENCE)
    residence = explicit_residence or provided_residence
    if residence is None and (
        re.search(r"\btotal\s+persons\b", folded_query) or "total persons" in folded_population
    ):
        residence = "total"
    return population, residence


def canonicalize_artifact_requirement(
    requirement: ArtifactDataRequirement, query: str
) -> ArtifactDataRequirement:
    population, residence = canonical_scopes(
        query, requirement.population_scope, requirement.residence_scope
    )
    normalized_metric = " ".join(requirement.metric.split()).casefold()
    metric = _METRIC_ALIASES.get(normalized_metric, requirement.metric.strip())
    residences = [
        scope
        for scope in ("total", "rural", "urban")
        if re.search(rf"\b{scope}\b", query.casefold())
    ]
    return requirement.model_copy(
        update={
            "metric": metric,
            "population_scope": population,
            "residence_scope": None if len(residences) > 1 else residence,
            "residence_scopes": residences if len(residences) > 1 else [],
        }
    )


# Census statements that report a population subgroup. Their titles live in a separate chunk
# (see AgentTools table titles), and a subgroup cell is never an answer for the whole
# population: Karnataka's Scheduled Tribes sex ratio (990) was once reported as the state's
# sex ratio (973) because the "Statement 17" breadcrumb dropped its title.
_SUBGROUP_PATTERNS = {
    # Full names match in any case; the SC/ST abbreviations only in capitals.
    "scheduled castes": re.compile(r"\b(?i:scheduled\s+castes?)\b|\bSCs?\b"),
    "scheduled tribes": re.compile(r"\b(?i:scheduled\s+tribes?)\b|\bSTs?\b"),
    "children": re.compile(r"\b(?i:child(?:ren)?)\b|\b0\s*-\s*6\b"),
}


def population_subgroups(text: str) -> set[str]:
    return {name for name, pattern in _SUBGROUP_PATTERNS.items() if pattern.search(text)}


def unrequested_subgroups(table_context: str, request: str) -> set[str]:
    """Subgroups a table reports that the request never asked for."""
    return population_subgroups(table_context) - population_subgroups(request)
