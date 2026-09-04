from langchain_core.messages import AIMessage

from backend.app.providers.verification import verify_tool_call


def test_structured_tool_call_is_parsed() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_test_metric",
                "args": {"region": "Karnataka"},
                "id": "test-call",
                "type": "tool_call",
            }
        ],
    )

    result = verify_tool_call(
        message,
        expected_name="get_test_metric",
        expected_region="Karnataka",
    )

    assert result.passed is True
    assert result.tool_name == "get_test_metric"
    assert result.region == "Karnataka"


def test_missing_tool_call_fails_verification() -> None:
    result = verify_tool_call(
        AIMessage(content="No tool needed"),
        expected_name="get_test_metric",
        expected_region="Karnataka",
    )

    assert result.passed is False
