from pathlib import Path

import pytest

from backend.app.agent.skills import SkillRegistry


def test_relevant_skill_is_loaded_and_lookup_has_none(tmp_path: Path) -> None:
    (tmp_path / "summary.md").write_text(
        "---\nname: summary\ndescription: Safe summary\ntask_types: [summary]\n---\nCite facts.\n"
    )
    registry = SkillRegistry(tmp_path)
    assert registry.for_task("summary").metadata.name == "summary"  # type: ignore[union-attr]
    assert registry.for_task("lookup") is None


def test_skill_path_traversal_and_absolute_paths_are_rejected(tmp_path: Path) -> None:
    registry = SkillRegistry(tmp_path)
    with pytest.raises(ValueError, match="Invalid skill name"):
        registry.read_skill("../secret")
    with pytest.raises(ValueError, match="Invalid skill name"):
        registry.read_skill("/tmp/secret")


def test_skill_reference_programs_pass_the_executor_code_policy() -> None:
    # The table/chart skills teach a known-good program; it must never drift into code the
    # sandbox rejects (live failures: `.rename()`, `__main__` guards, runtime crashes).
    import re
    from pathlib import Path

    from executor.policy import validate_code

    for name in ("table", "chart"):
        text = (Path("skills") / f"{name}.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        assert len(blocks) == 1, name
        result = validate_code(blocks[0])
        assert result.valid, (name, result.errors)
