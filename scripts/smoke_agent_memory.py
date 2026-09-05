import argparse
import json
import re
import sys
import urllib.error
import urllib.request

KARNATAKA_DOCUMENT = "census-2011-karnataka-pca-highlights"
ODISHA_DOCUMENT = "census-2011-odisha-pca-highlights"


class SmokeFailure(RuntimeError):
    pass


def _contains_number(text: str, expected: float, *, tolerance: float = 0.005) -> bool:
    values = [float(value) for value in re.findall(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)", text)]
    return any(abs(value - expected) <= tolerance for value in values)


def _trusted_citations(trace: dict) -> dict[str, dict]:
    citations: dict[str, dict] = {}
    for event in trace.get("events", []):
        if event.get("event") == "validated_citations":
            for item in event.get("details", {}).get("citations", []):
                citations[item["citation_id"]] = item
    return citations


def assert_common(result: dict, trace: dict, session_id: str) -> None:
    if trace.get("session_id") != session_id:
        raise SmokeFailure("Trace session mismatch")
    trusted = _trusted_citations(trace)
    for claim in result.get("claims", []):
        if claim.get("document_derived") and not claim.get("citation_ids"):
            raise SmokeFailure("Returned an uncited factual claim")
    for citation in result.get("citations", []):
        validated = trusted.get(citation.get("citation_id"))
        if validated is None:
            raise SmokeFailure("Citation does not map to current-run validated evidence")
        for field in ("chunk_id", "document_id", "page_number", "snippet"):
            if citation.get(field) != validated.get(field):
                raise SmokeFailure(f"Citation has untrusted {field}")


def assert_answerable(result: dict, *, turn: int) -> None:
    if result.get("refusal"):
        raise SmokeFailure(f"Turn {turn} unexpectedly refused")
    if not result.get("claims"):
        raise SmokeFailure(f"Turn {turn} returned zero factual claims")


def assert_turn_1(result: dict) -> None:
    assert_answerable(result, turn=1)
    if not _contains_number(result["answer"], 75.36):
        raise SmokeFailure("Turn 1 omitted Karnataka value 75.36")
    document_matches = [
        item for item in result["citations"] if item["document_id"] == KARNATAKA_DOCUMENT
    ]
    if not document_matches:
        raise SmokeFailure("Turn 1 citation document match failed")
    page_matches = [item for item in document_matches if item["page_number"] in {10, 50}]
    if not page_matches:
        raise SmokeFailure("Turn 1 citation page match failed; expected physical PDF page 10 or 50")
    value_matches = [item for item in page_matches if _contains_number(item["snippet"], 75.36)]
    if not value_matches:
        raise SmokeFailure("Turn 1 citation quote does not visibly contain 75.36")
    cited_ids = {value for claim in result["claims"] for value in claim["citation_ids"]}
    if not any(item["citation_id"] in cited_ids for item in value_matches):
        raise SmokeFailure("Turn 1 value-bearing citation is not mapped to the factual claim")


def assert_turn_2(result: dict, trace: dict) -> None:
    assert_answerable(result, turn=2)
    for value in (75.36, 72.9, 2.46):
        if not _contains_number(result["answer"], value):
            raise SmokeFailure(f"Turn 2 omitted expected value {value}")
    documents = {item["document_id"] for item in result["citations"]}
    if not {KARNATAKA_DOCUMENT, ODISHA_DOCUMENT} <= documents:
        raise SmokeFailure("Turn 2 lacks citations from both comparison documents")
    resolutions = [
        event["details"].get("query", "")
        for event in trace["events"]
        if event["event"] == "resolved_query"
    ]
    if not resolutions or not all(
        term in resolutions[-1].casefold() for term in ("literacy", "karnataka", "odisha")
    ):
        raise SmokeFailure("Turn 2 did not resolve the prior literacy context")


def assert_turn_3(result: dict) -> None:
    assert_answerable(result, turn=3)
    documents = {item["document_id"] for item in result["citations"]}
    if not {KARNATAKA_DOCUMENT, ODISHA_DOCUMENT} <= documents:
        raise SmokeFailure("Turn 3 lacks source support for both values")


def assert_turn_4(result: dict) -> None:
    if not result.get("refusal"):
        raise SmokeFailure("Turn 4 did not refuse an out-of-scope request")
    if result.get("claims") or result.get("citations"):
        raise SmokeFailure("Turn 4 fabricated an out-of-scope claim or citation")


def request_json(
    base_url: str, method: str, path: str, payload: dict[str, str] | None = None
) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        method=method,
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=210) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"error_code": "HTTP_ERROR", "message": "Non-JSON HTTP error"}
        safe = {
            key: payload.get(key)
            for key in ("error_code", "message", "session_id", "trace_id", "retryable")
            if key in payload
        }
        print(json.dumps({"http_status": error.code, "error": safe}, indent=2), file=sys.stderr)
        raise SmokeFailure(f"HTTP {error.code}: {safe.get('error_code', 'HTTP_ERROR')}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual live persistent-memory agent smoke test")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--session-id")
    parser.add_argument("--start-turn", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    if args.session_id:
        session_id = args.session_id
        request_json(args.base_url, "GET", f"/sessions/{session_id}")
    else:
        if args.start_turn != 1:
            raise SmokeFailure("--start-turn after 1 requires --session-id")
        session = request_json(args.base_url, "POST", "/sessions")
        session_id = session["session_id"]
    print(json.dumps({"session_id": session_id, "start_turn": args.start_turn}, indent=2))
    if args.start_turn >= 2:
        context = request_json(args.base_url, "GET", f"/sessions/{session_id}/context")
        required_turns = 1 if args.start_turn == 2 else 2
        if context.get("validated_turn_count", 0) < required_turns:
            raise SmokeFailure("Session lacks the successful validated turns required to resume")
        if args.start_turn >= 3 and not context.get("has_validated_comparison"):
            raise SmokeFailure("Session lacks a validated comparison required for Turn 3")
    turns = [
        "What was the literacy rate in Karnataka in 2011?",
        "How does that compare with Odisha?",
        "Which source pages support those values?",
        "What was the unemployment rate in France in 2011?",
    ]
    trace_ids: set[str] = set()
    results: list[dict] = []
    for number, message in enumerate(turns, start=1):
        if number < args.start_turn:
            continue
        result = request_json(
            args.base_url,
            "POST",
            "/chat",
            {"session_id": session_id, "message": message},
        )
        trace = request_json(args.base_url, "GET", f"/runs/{result['trace_id']}/trace")
        print(json.dumps({"turn": number, "message": message, "response": result}, indent=2))
        if result["trace_id"] in trace_ids:
            raise SmokeFailure("Trace IDs are not unique")
        trace_ids.add(result["trace_id"])
        assert_common(result, trace, session_id)
        if number == 1:
            assert_turn_1(result)
        elif number == 2:
            assert_turn_2(result, trace)
        elif number == 3:
            assert_turn_3(result)
        else:
            assert_turn_4(result)
        results.append(result)
    if all(item.get("refusal") for item in results):
        raise SmokeFailure("All-refusal run cannot pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
