import re

from backend.app.agent.models import ArtifactDataRequirement

_RESIDENCE = ("rural", "urban", "total")
_POPULATION = ("female", "male", "persons")
_METRIC_ALIASES = {
    "literacy rate": "literacy rate",
    "literacy rates": "literacy rate",
    "effective literacy rate": "literacy rate",
    "effective literacy rates": "literacy rate",
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
    return requirement.model_copy(
        update={
            "metric": metric,
            "population_scope": population,
            "residence_scope": residence,
        }
    )
