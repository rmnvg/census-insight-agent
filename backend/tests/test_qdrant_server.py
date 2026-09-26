"""Withdrawn-document exclusion against a real Qdrant server; skipped unless TEST_QDRANT_URL is set.

Local-mode Qdrant evaluates filters in Python and does not reproduce server query planning:
Qdrant 1.15 returns points that `must_not` excludes when `must` matches the same `document_id`.
"""

import asyncio
import os
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient

from backend.app.agent.skills import SkillRegistry
from backend.app.agent.tools import AgentTools
from backend.app.ingestion.uploads import withdrawn_document_ids
from backend.app.retrieval.models import SparseVectorData
from backend.app.retrieval.service import HybridRetrievalService
from backend.tests.test_qdrant_retrieval import DenseProvider, SparseEncoder, make_chunk, make_store

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_QDRANT_URL"),
    reason="TEST_QDRANT_URL is not set; server filter tests need a real Qdrant",
)

WITHDRAWN = "upload-kerala-0123456789"


def test_withdrawn_documents_are_excluded_by_a_real_server(tmp_path: Path) -> None:
    client = QdrantClient(url=os.environ["TEST_QDRANT_URL"])
    name = f"withdrawn_{uuid4().hex}"
    store = make_store(client, name)
    store.ensure_collection()
    try:
        store.upsert(
            [
                make_chunk(
                    str(uuid4()),
                    document_id=document_id,
                    region="Kerala",
                    page_number=1,
                    text="Kerala literacy is 94.00",
                )
                for document_id in ("doc-a", WITHDRAWN)
            ],
            [[1.0, 0.0, 0.0]] * 2,
            [SparseVectorData(indices=[1], values=[1.0])] * 2,
        )
        marker = tmp_path / "processed" / "uploads" / "withdrawn" / f"{WITHDRAWN}.json"
        marker.parent.mkdir(parents=True)
        marker.write_text("{}", encoding="utf-8")
        service = HybridRetrievalService(
            client=client,
            collection_name=name,
            dense_provider=cast(Any, DenseProvider()),
            sparse_encoder=cast(Any, SparseEncoder()),
            excluded_document_ids=lambda: withdrawn_document_ids(tmp_path),
        )

        for document_ids, expected in (
            (None, {"doc-a"}),
            (["doc-a", WITHDRAWN], {"doc-a"}),
            ([WITHDRAWN], set()),
        ):
            found = service.search("Kerala literacy", document_ids, None, 10)
            assert {item.document_id for item in found} == expected, document_ids

        tools = AgentTools(service, store, SkillRegistry(tmp_path / "skills"), tmp_path)
        assert asyncio.run(tools.collect_summary_evidence(WITHDRAWN)) == []
        assert len(asyncio.run(tools.collect_summary_evidence("doc-a"))) == 1
    finally:
        client.delete_collection(name)
