"""Structured table layer, tested on real pages and the live chunks that index them.

`fixtures/tables_real_pages.json` holds pages from all three reports with every indexed chunk on
those pages, copied from the running collection: Karnataka Statements 6, 17, and 19 (sex ratio,
Scheduled Tribe sex ratio, literacy), Odisha Statement 19 (whose residence sub-header sits on a
different row under each metric), and Madhya Pradesh Statement 16 (which writes the sex inside
the metric's own header cell).
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from backend.app.agent.errors import AgentOperationalError
from backend.app.agent.graph import AgentGraph
from backend.app.agent.models import AgentPlan, TaskClassification
from backend.app.agent.skills import SkillRegistry
from backend.app.agent.tools import AgentTools, TableLookup
from backend.app.execution.hydration import dataset_from_table_selection
from backend.app.execution.lineage import validate_dataset
from backend.app.ingestion.models import DocumentPage
from backend.app.retrieval.models import DocumentSummary, RetrievedEvidence
from backend.app.tables.extraction import IndexedChunk, extract_document_tables
from backend.app.tables.models import DocumentTables, table_records
from backend.app.tables.selection import TableSelectionError, select_column
from backend.app.tables.store import load_document_tables, tables_path, write_document_tables
from backend.tests.test_agent_ranking import make_graph, ranking_requirement
from backend.tests.test_agent_service import FakeTools

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "tables_real_pages.json").read_text(encoding="utf-8")
)
KARNATAKA = "census-2011-karnataka-pca-highlights"
ODISHA = "census-2011-odisha-pca-highlights"
MADHYA_PRADESH = "census-2011-madhya-pradesh-pca-highlights"


def pages(document_id: str) -> list[DocumentPage]:
    return [DocumentPage.model_validate(page) for page in FIXTURE[document_id]["pages"]]


def chunks(document_id: str) -> list[IndexedChunk]:
    return [
        IndexedChunk(item["chunk_id"], item["page_number"], item["text"])
        for item in FIXTURE[document_id]["chunks"]
    ]


def build(
    document_id: str,
    page_list: list[DocumentPage] | None = None,
    chunk_list: list[IndexedChunk] | None = None,
) -> DocumentTables:
    entry = FIXTURE[document_id]
    return extract_document_tables(
        document_id=document_id,
        region=entry["region"],
        source_checksum=entry["source_checksum"],
        pages=pages(document_id) if page_list is None else page_list,
        chunks=chunks(document_id) if chunk_list is None else chunk_list,
    )


def test_three_row_header_resolves_every_column_and_binds_every_row() -> None:
    tables = build(KARNATAKA)
    literacy = next(table for table in tables.tables if table.statement == "19")
    assert literacy.title == "Literates and Literacy Rate by residence : 2011 (Persons)"
    assert (literacy.population_group, literacy.entity_kind) == ("persons", "district")
    described = [
        (column.metric, column.year, column.residence) for column in literacy.value_columns()
    ]
    assert described == [
        ("Literates", 2011, "total"),
        ("Literates", 2011, "rural"),
        ("Literates", 2011, "urban"),
        ("Literacy Rate", 2001, "total"),
        ("Literacy Rate", 2001, "rural"),
        ("Literacy Rate", 2001, "urban"),
        ("Literacy Rate", 2011, "total"),
        ("Literacy Rate", 2011, "rural"),
        ("Literacy Rate", 2011, "urban"),
    ]
    assert {column.unit for column in literacy.value_columns()} == {"persons", "percent"}
    assert len(literacy.rows) == 31 and literacy.malformed_rows == 0
    assert [row.label for row in literacy.rows if row.aggregate] == ["KARNATAKA"]
    assert all(row.chunk_id for row in literacy.rows)
    udupi = next(row for row in literacy.rows if row.label == "Udupi")
    assert (udupi.code, udupi.cells[8]) == ("569", "86.24")


def test_ragged_header_keeps_residence_under_each_metric() -> None:
    # Odisha puts Total/Rural/Urban on the second header row under "Literates 2011" but on the
    # third row under "Literacy Rate", beneath the year.
    literacy = build(ODISHA).tables[0]
    described = [
        (column.metric, column.year, column.residence) for column in literacy.value_columns()
    ]
    assert described[0] == ("Literates", 2011, "total")
    assert described[3] == ("Literacy Rate", 2001, "total")
    assert described[6] == ("Literacy Rate", 2011, "total")
    selection = select_column(build(ODISHA), metric="literacy rate")
    top = max(selection.records, key=lambda record: record.value)
    # The trust benchmark's od-top-literacy case, which previously ended MODEL_OUTPUT_INVALID.
    assert (top.entity, top.raw_value) == ("Khordha", "86.9")
    assert selection.aggregate is not None and selection.aggregate.raw_value == "72.9"


@pytest.mark.parametrize(
    ("metric", "year", "residence", "population", "winner"),
    [
        ("literacy rate", None, None, None, ("Dakshina Kannada", "88.57")),
        ("effective literacy rate", 2011, "total", "persons", ("Dakshina Kannada", "88.57")),
        ("literacy", None, None, None, ("Dakshina Kannada", "88.57")),
        ("literacy rate", 2001, None, None, ("Dakshina Kannada", "83.35")),
        ("literacy rate", None, "urban", None, ("Udupi", "92.13")),
        ("literates", None, None, None, ("Bangalore", "75,12,276")),
        ("sex ratio", None, None, None, ("Udupi", "1,094")),
        # The model sometimes puts the sex-ratio unit in population_scope.
        ("sex ratio", None, None, "females per 1000 males", ("Udupi", "1,094")),
        ("sex ratio among scheduled tribes", None, None, None, ("Chikmagalur", "1,045")),
    ],
)
def test_selection_binds_the_requested_column(
    metric: str,
    year: int | None,
    residence: str | None,
    population: str | None,
    winner: tuple[str, str],
) -> None:
    selection = select_column(
        build(KARNATAKA), metric=metric, year=year, residence=residence, population=population
    )
    assert len(selection.records) == 30
    assert all(not record.aggregate for record in selection.records)
    top = max(selection.records, key=lambda record: record.value)
    assert (top.entity, top.raw_value) == winner


def test_subgroup_tables_never_answer_whole_population_questions() -> None:
    tables = build(KARNATAKA)
    plain = select_column(tables, metric="sex ratio")
    tribal = select_column(tables, metric="sex ratio among Scheduled Tribes")
    assert plain.statement == "6" and plain.qualifiers == []
    assert tribal.statement == "17" and tribal.qualifiers == ["scheduled tribes"]
    with pytest.raises(TableSelectionError, match="TABLE_NO_MATCHING_COLUMN"):
        select_column(tables, metric="child sex ratio")
    with pytest.raises(TableSelectionError, match="TABLE_NO_MATCHING_COLUMN"):
        select_column(tables, metric="literacy rate", population="female")


def test_sex_written_inside_the_metric_cell_is_a_population_group() -> None:
    # Madhya Pradesh Statement 16: "Scheduled Tribe population<br>Males | ... | ...<br>Females".
    tables = build(MADHYA_PRADESH)
    groups = {column.population_group for column in tables.tables[0].value_columns()}
    assert groups == {"males", "females"}
    with pytest.raises(TableSelectionError, match="TABLE_NO_MATCHING_COLUMN"):
        select_column(tables, metric="scheduled tribe population")
    females = select_column(tables, metric="scheduled tribe population", population="female")
    assert females.population_group == "females" and len(females.records) == 50


def test_row_without_exactly_one_indexed_chunk_blocks_the_column() -> None:
    literacy_chunks = [chunk for chunk in chunks(KARNATAKA) if "Udupi" not in chunk.text]
    tables = build(KARNATAKA, chunk_list=literacy_chunks)
    with pytest.raises(TableSelectionError, match="TABLE_COLUMN_INCOMPLETE"):
        select_column(tables, metric="literacy rate")
    duplicated = [
        *chunks(KARNATAKA),
        *[
            IndexedChunk(f"copy-{c.chunk_id}", c.page_number, c.text)
            for c in chunks(KARNATAKA)
            if "Udupi" in c.text
        ],
    ]
    with pytest.raises(TableSelectionError, match="TABLE_COLUMN_INCOMPLETE"):
        select_column(build(KARNATAKA, chunk_list=duplicated), metric="literacy rate")


def test_statement_context_ends_at_the_next_heading_or_annexure() -> None:
    [literacy] = [page for page in pages(KARNATAKA) if page.page_number == 50]
    table = literacy.text[literacy.text.index("| State / District Code") :]
    annexure = literacy.model_copy(
        update={"page_number": 51, "text": f"# Annexures\n\n## Annexure - I\n\n{table}"}
    )
    graph = literacy.model_copy(update={"page_number": 52, "text": f"Graph - 18\n\n{table}"})
    tables = build(KARNATAKA, page_list=[literacy, annexure, graph], chunk_list=[])
    assert [(table.statement, table.title) for table in tables.tables] == [
        ("19", "Literates and Literacy Rate by residence : 2011 (Persons)"),
        (None, "Annexure - I"),
        (None, None),
    ]


def test_identical_reprints_are_one_answer_but_conflicting_copies_refuse() -> None:
    # The Odisha report prints its first two chapters twice.
    [literacy] = [page for page in pages(KARNATAKA) if page.page_number == 50]
    reprint = literacy.model_copy(update={"page_number": 90})
    reprinted_chunks = [
        IndexedChunk(f"reprint-{chunk.chunk_id}", 90, chunk.text)
        for chunk in chunks(KARNATAKA)
        if chunk.page_number == 50
    ]
    tables = build(
        KARNATAKA, page_list=[literacy, reprint], chunk_list=[*chunks(KARNATAKA), *reprinted_chunks]
    )
    selection = select_column(tables, metric="literacy rate")
    assert {record.page_number for record in selection.records} == {50}

    altered = reprint.model_copy(update={"text": reprint.text.replace("86.24", "96.24")})
    altered_chunks = [
        IndexedChunk(chunk.chunk_id, 90, chunk.text.replace("86.24", "96.24"))
        for chunk in reprinted_chunks
    ]
    conflicting = build(
        KARNATAKA, page_list=[literacy, altered], chunk_list=[*chunks(KARNATAKA), *altered_chunks]
    )
    with pytest.raises(TableSelectionError, match="TABLE_AMBIGUOUS_COLUMN"):
        select_column(conflicting, metric="literacy rate")


def test_table_carried_onto_the_next_page_is_one_table() -> None:
    [literacy] = [page for page in pages(KARNATAKA) if page.page_number == 50]
    lines = literacy.text.splitlines()
    split = next(index for index, line in enumerate(lines) if "| Udupi" in line)
    header_end = next(index for index, line in enumerate(lines) if line.startswith("| 1 "))
    header = lines[lines.index(next(line for line in lines if "State / District Code" in line)) :]
    header = header[: header.index(lines[header_end]) + 1]
    first = literacy.model_copy(update={"text": "\n".join(lines[:split])})
    second = literacy.model_copy(
        update={"page_number": 51, "text": "\n".join([*header, *lines[split:]])}
    )
    [table] = build(KARNATAKA, page_list=[first, second], chunk_list=[]).tables
    assert table.page_numbers == [50, 51] and len(table.rows) == 31


def test_records_describe_every_cell() -> None:
    records = [
        record
        for record in table_records(build(ODISHA))
        if record.entity == "Khordha" and record.metric == "Literacy Rate"
    ]
    assert [(r.year, r.residence, r.raw_value) for r in records] == [
        (2001, "total", "79.6"),
        (2001, "rural", "74.1"),
        (2001, "urban", "86.7"),
        (2011, "total", "86.9"),
        (2011, "rural", "83.0"),
        (2011, "urban", "91.0"),
    ]
    assert {(r.region, r.population_group, r.unit, r.page_number) for r in records} == {
        ("Odisha", "persons", "percent", 82)
    }


def test_store_round_trip_rejects_unsafe_ids_and_old_versions(tmp_path: Path) -> None:
    tables = build(ODISHA)
    path = write_document_tables(tmp_path, tables)
    assert path == tmp_path / "processed" / "tables" / f"{ODISHA}.json"
    assert load_document_tables(tmp_path, ODISHA) == tables
    assert load_document_tables(tmp_path, "missing-document") is None
    with pytest.raises(ValueError):
        tables_path(tmp_path, "../../etc/passwd")
    path.write_text(tables.model_copy(update={"version": 0}).model_dump_json())
    assert load_document_tables(tmp_path, ODISHA) is None


class FixtureQdrant:
    """Serves the fixture's live payloads the way QdrantClient.retrieve/scroll do."""

    def __init__(self, document_id: str, edit: Any = None) -> None:
        self.payloads = [dict(item) for item in FIXTURE[document_id]["chunks"]]
        if edit is not None:
            for payload in self.payloads:
                edit(payload)

    def retrieve(self, collection_name: str, ids: list[str], **_: Any) -> list[Any]:
        by_id = {payload["chunk_id"]: payload for payload in self.payloads}
        return [SimpleNamespace(id=i, payload=by_id[i]) for i in ids if i in by_id]

    def scroll(self, collection_name: str, **_: Any) -> tuple[list[Any], None]:
        return [SimpleNamespace(payload=payload) for payload in self.payloads], None


def fixture_tools(tmp_path: Path, document_id: str, edit: Any = None) -> AgentTools:
    write_document_tables(tmp_path, build(document_id))
    store = SimpleNamespace(client=FixtureQdrant(document_id, edit), collection_name="c")
    return AgentTools(
        cast(Any, None), cast(Any, store), SkillRegistry(tmp_path / "skills"), tmp_path
    )


def test_tool_returns_selection_with_its_live_evidence(tmp_path: Path) -> None:
    tools = fixture_tools(tmp_path, KARNATAKA)
    requirement = ranking_requirement(metric="literacy rate", residence_scope=None)
    lookup = asyncio.run(tools.select_table_column(KARNATAKA, requirement))
    assert lookup.reason is None and lookup.selection is not None
    cited = {record.chunk_id for record in lookup.selection.records}
    assert {item.chunk_id for item in lookup.evidence} == cited
    missing = asyncio.run(tools.select_table_column("census-2011-goa", requirement))
    assert missing.reason == "TABLE_STORE_MISSING"
    unknown = asyncio.run(
        tools.select_table_column(KARNATAKA, requirement.model_copy(update={"metric": "gdp"}))
    )
    assert unknown.reason == "TABLE_NO_MATCHING_COLUMN"


@pytest.mark.parametrize(
    "edit",
    [
        lambda payload: payload.update(text=payload["text"].replace("86.24", "86.25")),
        lambda payload: payload.update(source_checksum="b" * 64),
        lambda payload: payload.update(page_number=payload["page_number"] + 1),
    ],
)
def test_tool_refuses_a_store_that_no_longer_matches_the_index(tmp_path: Path, edit: Any) -> None:
    tools = fixture_tools(tmp_path, KARNATAKA, edit)
    requirement = ranking_requirement(metric="literacy rate", residence_scope=None)
    lookup = asyncio.run(tools.select_table_column(KARNATAKA, requirement))
    assert (lookup.selection, lookup.reason) == (None, "TABLE_STORE_STALE")


def table_lookup(tmp_path: Path, metric: str = "literacy rate") -> TableLookup:
    tools = fixture_tools(tmp_path, KARNATAKA)
    requirement = ranking_requirement(metric=metric, residence_scope=None)
    return asyncio.run(tools.select_table_column(KARNATAKA, requirement))


def test_dataset_cites_every_row_through_its_line(tmp_path: Path) -> None:
    lookup = table_lookup(tmp_path)
    assert lookup.selection is not None
    requirement = ranking_requirement(metric="literacy rate", residence_scope=None)
    dataset = dataset_from_table_selection(lookup.selection, requirement, lookup.evidence, "q")
    validate_dataset(dataset, lookup.evidence, requirement)
    assert len(dataset.rows) == 30 and dataset.units == {"value": "percent"}
    assert dataset.title == "Literacy Rate by district in Karnataka, 2011"
    udupi = next(source for source in dataset.source_records if source.region == "Udupi")
    assert udupi.exact_supporting_quote.endswith(
        next(record.line for record in lookup.selection.records if record.entity == "Udupi")
    )
    assert (udupi.raw_value, udupi.year, udupi.residence_scope) == ("86.24", 2011, "total")


def test_dataset_title_names_a_non_default_population_and_residence(tmp_path: Path) -> None:
    # Live: "Bhopal recorded the highest Literacy Rate (74.9 percent) among Literacy Rate by
    # district in Madhya Pradesh" answered a female-literacy question without saying so.
    requirement = ranking_requirement(
        metric="scheduled tribe population", residence_scope="urban", population_scope="female"
    )
    tools = fixture_tools(tmp_path, MADHYA_PRADESH)
    lookup = asyncio.run(tools.select_table_column(MADHYA_PRADESH, requirement))
    assert lookup.selection is not None
    dataset = dataset_from_table_selection(lookup.selection, requirement, lookup.evidence, "q")
    validate_dataset(dataset, lookup.evidence, requirement)
    assert dataset.title == (
        "Scheduled Tribe population by district in Madhya Pradesh (urban, females), 2011"
    )


class NoProposalModel:
    async def propose_artifact_dataset(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a verified table column needs no model proposal")


def ranking_state(lookup: TableLookup, **requirement: Any) -> dict[str, Any]:
    req = ranking_requirement(metric="literacy rate", residence_scope=None, **requirement)
    return {
        "session_id": "11111111-1111-4111-8111-111111111111",
        "run_id": "22222222-2222-4222-8222-222222222222",
        "resolved_query": "Which district of Karnataka had the highest literacy rate?",
        "task_type": "artifact_table",
        "classification": TaskClassification(
            task_type="artifact_table", regions=["Karnataka"], reason="ranking"
        ),
        "selected_evidence": lookup.evidence,
        "table_selection": lookup.selection,
        "artifact_requirement": req,
        "trace_events": [],
        "artifact_attempt": 0,
        "artifact_errors": [],
        "plan": AgentPlan(steps=["artifact"]),
    }


@pytest.mark.parametrize(("direction", "winner"), [("max", "Dakshina Kannada"), ("min", "Yadgir")])
def test_ranking_uses_the_table_without_a_model_proposal(
    tmp_path: Path, direction: str, winner: str
) -> None:
    lookup = table_lookup(tmp_path)
    graph = make_graph(cast(Any, NoProposalModel()), tmp_path)
    result: dict[str, Any] = asyncio.run(
        graph.prepare_artifact(cast(Any, ranking_state(lookup, rank_direction=direction)))
    )
    assert result["ranking_winner"].region == winner
    assert len(result["artifact_dataset"].rows) == 30
    events = {event.event: event.details for event in result["trace_events"]}
    assert "artifact_proposal_validation" not in events
    assert events["table_dataset"]["statement"] == "19"
    assert events["ranking_winner_selected"]["detected_row_count"] == 30


def test_ranking_fails_closed_when_table_evidence_is_missing(tmp_path: Path) -> None:
    lookup = table_lookup(tmp_path)
    state = ranking_state(lookup)
    state["selected_evidence"] = lookup.evidence[1:]
    graph = make_graph(cast(Any, NoProposalModel()), tmp_path)
    with pytest.raises(AgentOperationalError) as caught:
        asyncio.run(graph.prepare_artifact(cast(Any, state)))
    assert caught.value.code == "INTERNAL_PROVENANCE_INVALID"
    assert caught.value.diagnostics["reason_code"] == "UNKNOWN_EVIDENCE_ID"


class RankingTools(FakeTools):
    def __init__(self, skills: SkillRegistry, lookup: TableLookup) -> None:
        super().__init__(skills)
        self.lookup = lookup
        self.scanned: list[str] = []

    async def list_documents(self) -> list[Any]:
        return [
            DocumentSummary(
                document_id=KARNATAKA,
                title="Karnataka",
                region="Karnataka",
                source_checksum="a" * 64,
            )
        ]

    async def select_table_column(self, document_id: str, requirement: Any) -> TableLookup:
        return self.lookup

    async def collect_metric_table_rows(
        self, document_id: str, metric: str
    ) -> list[RetrievedEvidence]:
        self.scanned.append(document_id)
        return []


def call_tools_state(lookup: TableLookup) -> dict[str, Any]:
    state = ranking_state(lookup)
    state.update(tool_calls=[], errors=[], table_selection=None, selected_evidence=[])
    return state


def test_call_tools_prefers_the_table_and_falls_back_to_scanning(tmp_path: Path) -> None:
    lookup = table_lookup(tmp_path)
    tools = RankingTools(SkillRegistry(tmp_path / "skills"), lookup)
    graph = AgentGraph(cast(Any, NoProposalModel()), cast(Any, tools))
    found: dict[str, Any] = asyncio.run(graph.call_tools(cast(Any, call_tools_state(lookup))))
    assert found["table_selection"] == lookup.selection and tools.scanned == []
    assert {item.chunk_id for item in found["retrieved_evidence"]} == {
        item.chunk_id for item in lookup.evidence
    }

    tools.lookup = TableLookup(reason="TABLE_STORE_MISSING")
    fallback: dict[str, Any] = asyncio.run(graph.call_tools(cast(Any, call_tools_state(lookup))))
    assert fallback["table_selection"] is None and tools.scanned == [KARNATAKA]
    [event] = [e for e in fallback["trace_events"] if e.event == "table_lookup"]
    assert event.details["reason"] == "TABLE_STORE_MISSING"
