from langchain_core.messages import AIMessage

from backend.app.providers.contracts import ToolCallVerificationResult


def verify_tool_call(
    message: AIMessage,
    *,
    expected_name: str,
    expected_region: str,
) -> ToolCallVerificationResult:
    """Inspect, but do not execute, an expected structured tool call."""
    if not message.tool_calls:
        return ToolCallVerificationResult(passed=False, detail="Gemini returned no tool call")

    call = message.tool_calls[0]
    name = call.get("name")
    args = call.get("args", {})
    region = args.get("region") if isinstance(args, dict) else None
    passed = name == expected_name and region == expected_region
    detail = "Structured tool call matched" if passed else "Structured tool call did not match"
    return ToolCallVerificationResult(
        passed=passed,
        tool_name=name,
        region=region if isinstance(region, str) else None,
        detail=detail,
    )
