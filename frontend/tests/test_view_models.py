import pytest

from frontend.models import Citation, Claim, EvidenceSpan, TraceResponse
from frontend.view_models import (
    VISUAL_LIMITATION,
    citations_for_claim,
    ordered_citations,
    parse_csv,
    sanitize_trace,
    validate_png,
    validate_source_manifest,
)


def citation(identifier: str = "citation-1") -> Citation:
    snippet = "| State | Literacy |\n| Karnataka | 75.36 | <script>alert(1)</script>"
    return Citation(
        citation_id=identifier,
        document_id="doc-karnataka",
        document_title="Karnataka Census",
        page_number=50,
        snippet=snippet,
        chunk_id="chunk-1",
        section_path=["Literacy", "2011"],
        evidence_span=EvidenceSpan(evidence_id="chunk-1", start_offset=0, end_offset=len(snippet)),
    )


def test_citation_deduplication_mapping_and_exact_source_preservation() -> None:
    value = citation()
    claim = Claim(claim_id="claim-1", text="Literacy was 75.36%.", citation_ids=[value.citation_id])
    assert ordered_citations([value, value]) == [value]
    assert citations_for_claim(claim, [value]) == [value]
    assert citation().snippet.endswith("<script>alert(1)</script>")
    assert citation().page_number == 50


def test_missing_claim_citation_is_ignored_safely() -> None:
    claim = Claim(claim_id="claim-1", text="Claim", citation_ids=["missing"])
    assert citations_for_claim(claim, [citation()]) == []


def test_trace_allowlist_ignores_unknown_sensitive_fields() -> None:
    trace = TraceResponse.model_validate(
        {
            "session_id": "session",
            "run_id": "run",
            "events": [
                {
                    "timestamp": "2026-01-01T00:00:00Z",
                    "event": "evidence_assessment",
                    "node": "assess_evidence",
                    "details": {
                        "candidate_count": 2,
                        "credentials": "secret",
                        "embedding": [0.1, 0.2],
                        "prompt": "hidden",
                        "source_checksum": "a" * 64,
                    },
                    "unknown": "ignored",
                },
                {
                    "timestamp": "2026-01-01T00:00:00Z",
                    "event": "unknown_internal_event",
                    "details": {"status": "hidden"},
                },
            ],
            "unknown": "ignored",
        }
    )
    safe = sanitize_trace(trace)
    assert len(safe) == 1
    assert safe[0].details == {"candidate_count": 2}


def test_artifact_content_validation() -> None:
    assert validate_png(b"\x89PNG\r\n\x1a\nrest")
    with pytest.raises(ValueError, match="PNG"):
        validate_png(b"not-png")
    frame = parse_csv(b"region,value\nKarnataka,75.36\nOdisha,72.9\n")
    assert list(frame.columns) == ["region", "value"]
    assert frame.shape == (2, 2)
    manifest = validate_source_manifest(b'{"source_records": []}')
    assert manifest == {"source_records": []}


@pytest.mark.parametrize(
    "content",
    [b"", b"only-header\n", b"a,b\n1\n", b"\xff\xfe"],
)
def test_invalid_csv_handling(content: bytes) -> None:
    with pytest.raises(ValueError, match="CSV"):
        parse_csv(content)


@pytest.mark.parametrize("content", [b"[]", b"{}", b"not-json"])
def test_invalid_source_manifest(content: bytes) -> None:
    with pytest.raises(ValueError, match="manifest"):
        validate_source_manifest(content)


def test_excluded_visual_wording_does_not_claim_absence() -> None:
    assert "excluded" in VISUAL_LIMITATION
    assert "absent" not in VISUAL_LIMITATION.casefold()
    assert "not in the pdf" not in VISUAL_LIMITATION.casefold()
