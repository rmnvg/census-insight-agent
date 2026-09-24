import asyncio
from typing import Any, cast

from backend.app.agent.models import ResolvedQuery, TaskClassification
from backend.app.agent.provider import GeminiAgentModel
from backend.app.retrieval.models import DocumentSummary


class RecordingChat:
    def __init__(self) -> None:
        self.schemas: list[type[Any]] = []
        self.messages: list[Any] = []

    def with_structured_output(self, schema: type[Any]) -> "RecordingChat":
        self.schemas.append(schema)
        return self

    async def ainvoke(self, messages: Any) -> dict[str, object]:
        self.messages.append(messages)
        if self.schemas[-1] is ResolvedQuery:
            return {"query": "literacy rate of Northvale in 2011"}
        return {"task_type": "lookup", "regions": ["Northvale"], "reason": "named region"}


def human_prompt(chat: RecordingChat) -> str:
    return str(cast(list[Any], chat.messages[-1])[-1].content)


def test_classifier_and_resolver_see_the_supplied_document_catalog() -> None:
    async def catalog() -> list[DocumentSummary]:
        return [
            DocumentSummary(
                document_id="upload-northvale-highlights-0123456789",
                title="Northvale Census Highlights",
                region="Northvale",
                source_checksum="0" * 64,
            )
        ]

    chat = RecordingChat()
    model = GeminiAgentModel(chat, max_retries=0, catalog=catalog)  # type: ignore[arg-type]
    result = asyncio.run(model.classify("What was the literacy rate of Northvale?", []))
    assert isinstance(result, TaskClassification)
    prompt = human_prompt(chat)
    assert "Supplied documents" in prompt
    assert "upload-northvale-highlights-0123456789: Northvale Census Highlights" in prompt
    assert "(region: Northvale)" in prompt

    asyncio.run(model.resolve("And the sex ratio?", []))
    assert "(region: Northvale)" in human_prompt(chat)


def test_catalog_failure_never_blocks_classification() -> None:
    async def broken() -> list[DocumentSummary]:
        raise OSError("manifest directory unavailable")

    chat = RecordingChat()
    model = GeminiAgentModel(chat, max_retries=0, catalog=broken)  # type: ignore[arg-type]
    asyncio.run(model.classify("What was the literacy rate of Karnataka?", []))
    assert "Supplied documents" not in human_prompt(chat)
    assert human_prompt(chat).startswith("Recent conversation:")
