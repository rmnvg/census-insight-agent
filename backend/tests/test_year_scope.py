"""Year-aware scope: 2001 lookups, a 2001→2011 change, and refusal of years the tables lack.

Built on the real Karnataka pages and live chunks in `fixtures/tables_real_pages.json`: Statement 6
(sex ratio, 2001 and 2011 columns side by side) and Statement 19 (literacy, whose years sit on the
second header row, which later fragments of the table do not repeat). Values checked against the
source table: sex ratio 2001 Total 965, 2011 Total 973; literacy rate 2001 Total 66.64, 2011 Total
75.36.
"""

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.messages import BaseMessage, HumanMessage

from backend.app.agent.calculations import add_deterministic_derived_claims, validated_calculation
from backend.app.agent.citations import validate_and_materialize_citations
from backend.app.agent.graph import AgentGraph
from backend.app.agent.models import (
    AgentPlan,
    AgentResponse,
    CalculationRequest,
    ClaimUnit,
    DraftAnswer,
    DraftClaim,
    ResolvedQuery,
    TaskClassification,
    YearScope,
)
from backend.app.agent.provider import GeminiAgentModel
from backend.app.agent.years import is_year_change, named_years, table_year_verdict, years_beyond
from backend.app.retrieval.models import DocumentSummary, RetrievedEvidence
from backend.app.tables.models import TableRecord, census_years
from backend.tests.test_tables import FIXTURE, KARNATAKA, build, fixture_tools

SEX_RATIO = "966a994e-0557-58c5-b4cd-0924ea0260b7"  # Statement 6, first fragment (page 30)
LITERACY = "c1663038-700a-5623-94a1-1ca1d3a425c0"  # Statement 19, first fragment (page 50)


def payload(chunk_id: str) -> dict[str, Any]:
    return next(item for item in FIXTURE[KARNATAKA]["chunks"] if item["chunk_id"] == chunk_id)


def evidence(chunk_id: str) -> RetrievedEvidence:
    return RetrievedEvidence.model_validate({**payload(chunk_id), "retrieval_score": 1.0})


def cells(tmp_path: Path, items: list[RetrievedEvidence]) -> dict[str, list[TableRecord]]:
    return asyncio.run(fixture_tools(tmp_path, KARNATAKA).table_cells(items))


def claim(
    claim_id: str,
    text: str,
    chunk_id: str,
    value: float,
    *,
    year: int | None,
    metric: str,
    unit: ClaimUnit,
) -> DraftClaim:
    return DraftClaim(
        claim_id=claim_id,
        text=text,
        evidence_ids=[chunk_id],
        metric=metric,
        region="Karnataka",
        year=year,
        value=value,
        unit=unit,
    )


def sex_ratio(value: float, year: int | None) -> DraftClaim:
    when = f" in {year}" if year else ""
    return claim(
        "c1",
        f"The sex ratio of Karnataka{when} was {value:g}.",
        SEX_RATIO,
        value,
        year=year,
        metric="sex ratio",
        unit="count",
    )


def literacy(claim_id: str, value: float, year: int) -> DraftClaim:
    return claim(
        claim_id,
        f"Karnataka's literacy rate in {year} was {value} percent.",
        LITERACY,
        value,
        year=year,
        metric="literacy rate",
        unit="percent",
    )


def test_census_years_come_from_the_table_columns(tmp_path: Path) -> None:
    assert census_years(build(KARNATAKA)) == [2001, 2011]
    tools = fixture_tools(tmp_path, KARNATAKA)
    assert asyncio.run(tools.census_years()) == {KARNATAKA: [2001, 2011]}


def test_named_years_and_change_detection() -> None:
    assert named_years("Literates 4,06,47,322 in 2011, 66.64 in 2001") == [2011, 2001]
    assert is_year_change("How did Karnataka's literacy rate change from 2001 to 2011?")
    assert not is_year_change("What was the sex ratio of Karnataka in 2001?")
    assert years_beyond("literacy rate of Karnataka in the 2021 census", {2001, 2011}) == [2021]
    # Earlier years belong to decadal series and narrative text; evidence checks decide those.
    assert years_beyond("Odisha's population in 1961", {2001, 2011}) == []
    assert years_beyond("anything in 2021", set()) == []


def test_2001_lookup_binds_the_2001_column(tmp_path: Path) -> None:
    item = evidence(SEX_RATIO)
    table_cells = cells(tmp_path, [item])
    assert {record.year for record in table_cells[SEX_RATIO]} == {2001, 2011}

    draft = DraftAnswer(answer_markdown="x", claims=[sex_ratio(965, 2001)])
    result = validate_and_materialize_citations(draft, [item], [], table_cells)
    assert result.valid, result.errors
    assert result.citations[0].snippet.splitlines()[-1].startswith("| -")
    assert "<b>965</b>" in result.citations[0].snippet


def test_wrong_year_value_is_rejected(tmp_path: Path) -> None:
    item = evidence(SEX_RATIO)
    table_cells = cells(tmp_path, [item])
    # 973 is the 2011 Total, beside the 2001 one.
    mislabelled = DraftAnswer(answer_markdown="x", claims=[sex_ratio(973, 2001)])
    result = validate_and_materialize_citations(mislabelled, [item], [], table_cells)
    assert not result.valid
    assert "TABLE_YEAR_COLUMN_MISMATCH" in result.error_codes
    # The textual header check still rejects it without the store.
    assert not validate_and_materialize_citations(mislabelled, [item], []).valid

    # A 2001 value stated with no year reads as the report's own year (2011). Only the store
    # catches this one: the quote names no year for the claim to contradict.
    unlabelled = DraftAnswer(answer_markdown="x", claims=[sex_ratio(965, None)])
    assert validate_and_materialize_citations(unlabelled, [item], []).valid
    result = validate_and_materialize_citations(unlabelled, [item], [], table_cells)
    assert not result.valid
    assert "TABLE_YEAR_COLUMN_MISMATCH" in result.error_codes


def test_wrong_year_cell_is_not_pruned_when_another_citation_passes(tmp_path: Path) -> None:
    table = evidence(SEX_RATIO)
    # The same row in a chunk the store does not cover passes the textual checks on its own.
    uncovered = table.model_copy(update={"chunk_id": "uncovered-copy"})
    wrong = sex_ratio(965, None).model_copy(update={"evidence_ids": [SEX_RATIO, "uncovered-copy"]})
    draft = DraftAnswer(answer_markdown="x", claims=[wrong])
    assert validate_and_materialize_citations(draft, [uncovered], []).citations
    result = validate_and_materialize_citations(
        draft, [table, uncovered], [], cells(tmp_path, [table])
    )
    assert not result.valid
    assert "TABLE_YEAR_COLUMN_MISMATCH" in result.error_codes


def test_store_binds_years_in_a_fragment_whose_header_lost_them(tmp_path: Path) -> None:
    fragment = next(
        item
        for item in FIXTURE[KARNATAKA]["chunks"]
        if item["page_number"] == 50 and "| Udupi" in item["text"]
    )
    # Later fragments repeat only the first header row, which names no literacy-rate year.
    assert "2001" not in fragment["text"]
    item = RetrievedEvidence.model_validate({**fragment, "retrieval_score": 1.0})
    udupi = [
        record for record in cells(tmp_path, [item])[item.chunk_id] if record.entity == "Udupi"
    ]

    def verdict(value: float, year: int | None) -> str | None:
        udupi_claim = claim(
            "c", "x", item.chunk_id, value, year=year, metric="literacy rate", unit="percent"
        )
        return table_year_verdict(udupi_claim.model_copy(update={"region": "Udupi"}), udupi)

    assert verdict(81.25, 2001) == "confirmed"
    assert verdict(86.24, 2011) == "confirmed"
    assert verdict(81.25, 2011) == "contradicted"
    assert verdict(81.25, None) == "contradicted"
    assert verdict(99.99, 2001) is None


def test_store_cells_require_the_row_verbatim_in_the_chunk(tmp_path: Path) -> None:
    item = evidence(SEX_RATIO)
    edited = item.model_copy(update={"text": item.text.replace("<b>965</b>", "<b>966</b>")})
    rows = {record.entity for record in cells(tmp_path, [edited]).get(SEX_RATIO, [])}
    assert "KARNATAKA" not in rows and "Belgaum" in rows
    stale = item.model_copy(update={"source_checksum": "b" * 64})
    assert cells(tmp_path, [stale]) == {}


def test_2001_to_2011_change_reuses_the_comparison_difference(tmp_path: Path) -> None:
    item = evidence(LITERACY)
    query = "How did Karnataka's literacy rate change from 2001 to 2011?"
    calculation = validated_calculation(
        query,
        CalculationRequest(
            description="Change in Karnataka's literacy rate from 2001 to 2011",
            operation="difference",
            values=[75.36, 66.64],
            evidence_ids=[LITERACY],
        ),
        [item],
    )
    assert calculation is not None
    assert (calculation.operation, calculation.unit, calculation.result) == (
        "difference",
        "percentage_points",
        8.72,
    )
    draft = add_deterministic_derived_claims(
        DraftAnswer(
            answer_markdown="x",
            claims=[literacy("c2011", 75.36, 2011), literacy("c2001", 66.64, 2001)],
        ),
        [calculation],
    )
    derived = draft.claims[-1]
    assert derived.text == (
        "Karnataka's literacy rate was higher in 2011 than in 2001 by 8.72 percentage points."
    )
    assert derived.year is None and derived.derivation is not None
    assert derived.derivation.input_claim_ids == ["c2011", "c2001"]

    table_cells = cells(tmp_path, [item])
    result = validate_and_materialize_citations(draft, [item], [calculation], table_cells)
    assert result.valid, result.errors
    by_id = {answer.claim_id: answer for answer in result.claims}
    # Both input cells are cited, and the change inherits exactly those citations.
    assert by_id["c2011"].citation_ids and by_id["c2001"].citation_ids
    assert set(by_id[derived.claim_id].citation_ids) == {
        *by_id["c2011"].citation_ids,
        *by_id["c2001"].citation_ids,
    }


def test_change_with_swapped_years_is_rejected(tmp_path: Path) -> None:
    item = evidence(LITERACY)
    swapped = DraftAnswer(
        answer_markdown="x",
        claims=[literacy("c2011", 66.64, 2011), literacy("c2001", 75.36, 2001)],
    )
    result = validate_and_materialize_citations(swapped, [item], [], cells(tmp_path, [item]))
    assert not result.valid
    assert "TABLE_YEAR_COLUMN_MISMATCH" in result.error_codes


def test_resolver_cannot_silently_replace_the_requested_year() -> None:
    state = {
        "task_type": "lookup",
        "evidence_sufficient": True,
        "user_query": "What was Karnataka's sex ratio in 2001?",
        "resolved_query": "What was Karnataka's sex ratio in 2011?",
    }
    draft = DraftAnswer(answer_markdown="x", claims=[sex_ratio(973, 2011)])
    assert "MISSING_REQUESTED_YEAR" in AgentGraph._response_invariant_errors(
        cast(Any, state),
        draft,
        0,
    )


def test_unknown_year_cannot_pass_even_if_its_value_matches_a_known_cell(tmp_path: Path) -> None:
    item = evidence(SEX_RATIO)
    draft = DraftAnswer(answer_markdown="x", claims=[sex_ratio(973, 2021)])
    result = validate_and_materialize_citations(draft, [item], [], cells(tmp_path, [item]))
    assert not result.valid and "TABLE_YEAR_COLUMN_MISMATCH" in result.error_codes


class ScopeModel:
    def __init__(self, resolved: str = "") -> None:
        self.resolved = resolved

    async def classify(self, query: str, context: list[BaseMessage]) -> TaskClassification:
        return TaskClassification(task_type="lookup", regions=["Karnataka"], reason="lookup")

    async def resolve(self, query: str, context: list[BaseMessage]) -> ResolvedQuery:
        return ResolvedQuery(query=self.resolved, task_type="lookup", regions=["Karnataka"])


def classify(tmp_path: Path, query: str) -> tuple[AgentGraph, dict[str, Any], dict[str, Any]]:
    graph = AgentGraph(cast(Any, ScopeModel()), fixture_tools(tmp_path, KARNATAKA))
    state: dict[str, Any] = {
        "user_query": query,
        "messages": [HumanMessage(content=query)],
        "trace_events": [],
        "run_id": "33333333-3333-4333-8333-333333333333",
    }
    return graph, state, asyncio.run(graph.classify_task(cast(Any, state)))


def test_a_census_year_the_tables_lack_is_refused(tmp_path: Path) -> None:
    graph, state, result = classify(
        tmp_path, "What was the literacy rate of Karnataka in the 2021 census?"
    )
    assert result["task_type"] == "out_of_scope"
    assert result["scope_refusal"] == (
        "The supplied Census reports contain figures for 2001 and 2011; they do not cover 2021, "
        "so I can’t answer that from them."
    )
    state.update(result)
    response = cast(
        AgentResponse, asyncio.run(graph.graceful_response(cast(Any, state)))["final_response"]
    )
    assert response.refusal and not response.claims
    assert response.answer_markdown == result["scope_refusal"]


def test_2001_questions_stay_in_scope(tmp_path: Path) -> None:
    for query in (
        "What was the sex ratio of Karnataka in 2001?",
        "How did Karnataka's literacy rate change from 2001 to 2011?",
    ):
        _, _, result = classify(tmp_path, query)
        assert (result["task_type"], result["scope_refusal"]) == ("lookup", None)


@pytest.mark.parametrize(
    "rewrite",
    [
        "How did Karnataka's literacy rate change from 2001 to 2011?",
        "What was Karnataka's literacy rate in 2001 and 2011?",
    ],
)
def test_year_change_lookup_takes_the_comparison_path(tmp_path: Path, rewrite: str) -> None:
    query = "How did Karnataka's literacy rate change from 2001 to 2011?"
    graph = AgentGraph(cast(Any, ScopeModel(resolved=rewrite)), fixture_tools(tmp_path, KARNATAKA))
    state = {
        "user_query": query,
        "messages": [HumanMessage(content=query)],
        "task_type": "lookup",
        "classification": TaskClassification(
            task_type="lookup", regions=["Karnataka"], reason="lookup"
        ),
        "trace_events": [],
    }
    result: dict[str, Any] = asyncio.run(graph.resolve_query(cast(Any, state)))
    assert result["task_type"] == "comparison"
    assert result["classification"].task_type == "comparison"
    assert result["resolved_query"].endswith(
        "Compute the change for Karnataka in the same metric, population category, and units "
        "between 2001 and 2011, taking each year's value from that year's own column."
    )


def test_catalog_lists_each_documents_census_years() -> None:
    async def documents() -> list[DocumentSummary]:
        return [
            DocumentSummary(
                document_id=KARNATAKA, title="Karnataka", region="Karnataka", source_checksum="a"
            )
        ]

    async def years() -> dict[str, list[int]]:
        return {KARNATAKA: [2001, 2011]}

    model = GeminiAgentModel(cast(Any, None), catalog=documents, census_years=years)
    note = asyncio.run(model._catalog_note())
    assert (
        f"- {KARNATAKA}: Karnataka (region: Karnataka; Census years in its tables: 2001, 2011)"
        in note
    )


def year_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edit: Any = None) -> Any:
    tools = fixture_tools(tmp_path, KARNATAKA, edit)

    async def documents() -> list[DocumentSummary]:
        return [
            DocumentSummary(
                document_id=KARNATAKA,
                title="Karnataka",
                region="Karnataka",
                source_checksum=FIXTURE[KARNATAKA]["source_checksum"],
            )
        ]

    monkeypatch.setattr(tools, "list_documents", documents)
    return tools


@pytest.mark.parametrize(
    ("metric", "years", "expected"),
    [
        ("sex ratio", [2001], [(2001, 965.0)]),
        ("literacy rate", [2001, 2011], [(2011, 75.36), (2001, 66.64)]),
    ],
)
def test_selects_state_cells_from_the_full_column_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric: str,
    years: list[int],
    expected: list[tuple[int, float]],
) -> None:
    tools = year_tools(tmp_path, monkeypatch)
    lookup = asyncio.run(
        tools.select_year_cells(
            YearScope(
                metric=metric,
                region="Karnataka",
                years=years,
            )
        )
    )
    assert lookup.reason is None
    assert [(cell.year, float(cell.value)) for cell in lookup.records] == expected
    for cell in lookup.records:
        chunk = next(item for item in lookup.evidence if item.chunk_id == cell.chunk_id)
        assert cell.line in chunk.text.splitlines()
        assert cell.page_number == chunk.page_number


@pytest.mark.parametrize(
    "edit",
    [
        lambda payload: payload.update(text=payload["text"].replace("<b>965</b>", "<b>966</b>")),
        lambda payload: payload.update(source_checksum="b" * 64),
        lambda payload: payload.update(page_number=payload["page_number"] + 1),
    ],
)
def test_year_selection_refuses_stale_source_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    edit: Any,
) -> None:
    tools = year_tools(tmp_path, monkeypatch, edit)
    lookup = asyncio.run(
        tools.select_year_cells(
            YearScope(
                metric="sex ratio",
                region="Karnataka",
                years=[2001],
            )
        )
    )
    assert not lookup.records and lookup.reason == "TABLE_STORE_STALE"


def test_year_selection_does_not_substitute_2011_for_2021(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = year_tools(tmp_path, monkeypatch)
    lookup = asyncio.run(
        tools.select_year_cells(
            YearScope(
                metric="literacy rate",
                region="Karnataka",
                years=[2021],
            )
        )
    )
    assert not lookup.records and lookup.reason == "TABLE_NO_MATCHING_COLUMN"


def test_same_number_in_a_different_residence_column_cannot_confirm_a_year(tmp_path: Path) -> None:
    item = evidence(SEX_RATIO)
    records = cells(tmp_path, [item])[item.chunk_id]
    # 979 is Karnataka's 2011 Rural sex ratio, not its 2011 Total or 2001 Total.
    for year in (2001, 2011):
        assert table_year_verdict(sex_ratio(979, year), records) == "contradicted"
    rural = sex_ratio(979, 2011).model_copy(update={"residence_scope": "rural"})
    assert table_year_verdict(rural, records) == "confirmed"


def test_graph_reuses_arithmetic_on_selected_year_cells_without_model_value_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.agent.checkpoint import checkpoint_safe, hydrate_agent_state

    tools = year_tools(tmp_path, monkeypatch)
    graph = AgentGraph(cast(Any, ScopeModel()), tools)
    scope = YearScope(metric="literacy rate", region="Karnataka", years=[2001, 2011])
    state: dict[str, Any] = {
        "task_type": "comparison",
        "resolved_query": "How did Karnataka's literacy rate change from 2001 to 2011?",
        "classification": TaskClassification(
            task_type="comparison", regions=["Karnataka"], reason="year change", year_scope=scope
        ),
        "plan": AgentPlan(steps=["select cells"]),
        "trace_events": [],
        "tool_calls": [],
    }
    state.update(asyncio.run(graph.call_tools(cast(Any, state))))
    assert state["calculations"][0].result == 8.72
    assert state["calculations"][0].unit == "percentage_points"
    assert [cell.year for cell in state["year_cells"]] == [2011, 2001]
    assert {call.tool_name for call in state["tool_calls"]} == {
        "select_year_cells",
        "calculate",
        "get_document_coverage",
    }
    # Cells retain their types and decimal values across a real node checkpoint boundary.
    state = dict(hydrate_agent_state(checkpoint_safe(state)))
    state["selected_evidence"] = state["retrieved_evidence"]
    state.update(asyncio.run(graph.synthesize(cast(Any, state))))
    draft = state["draft_answer"]
    assert [claim.year for claim in draft.claims] == [2011, 2001, None]
    result = validate_and_materialize_citations(
        draft,
        state["selected_evidence"],
        state["calculations"],
        asyncio.run(tools.table_cells(state["selected_evidence"])),
    )
    assert result.valid, result.errors
    assert set(result.claims[-1].citation_ids) == {
        *result.claims[0].citation_ids,
        *result.claims[1].citation_ids,
    }
