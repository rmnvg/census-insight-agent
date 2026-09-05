from backend.app.agent.evidence import compatible_rounding


def test_compatible_rounding_distinguishes_rounding_from_conflict() -> None:
    assert compatible_rounding("75.36", "75.4")
    assert compatible_rounding("72.90", "72.9")
    assert not compatible_rounding("75.36", "74.4")
