import re
from pathlib import Path

from pydantic import BaseModel

from backend.app.agent.models import TaskType

SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class SkillMetadata(BaseModel):
    name: str
    description: str
    task_types: list[TaskType]


class RuntimeSkill(BaseModel):
    metadata: SkillMetadata
    instructions: str


class SkillRegistry:
    """Read-only registry for configured Markdown instruction files."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()

    def list_skills(self) -> list[SkillMetadata]:
        return [self._parse(path).metadata for path in sorted(self.directory.glob("*.md"))]

    def read_skill(self, skill_name: str) -> RuntimeSkill:
        if not SAFE_NAME.fullmatch(skill_name):
            raise ValueError("Invalid skill name")
        path = (self.directory / f"{skill_name}.md").resolve()
        if path.parent != self.directory or not path.is_file():
            raise ValueError("Unknown skill")
        return self._parse(path)

    @staticmethod
    def _parse(path: Path) -> RuntimeSkill:
        raw = path.read_text(encoding="utf-8")
        if not raw.startswith("---\n") or "\n---\n" not in raw[4:]:
            raise ValueError(f"Skill {path.name} has invalid front matter")
        header, instructions = raw[4:].split("\n---\n", 1)
        values: dict[str, str] = {}
        for line in header.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                values[key.strip()] = value.strip()
        task_types = [
            value.strip()
            for value in values.get("task_types", "").strip("[]").split(",")
            if value.strip()
        ]
        return RuntimeSkill(
            metadata=SkillMetadata.model_validate(
                {
                    "name": values.get("name"),
                    "description": values.get("description"),
                    "task_types": task_types,
                }
            ),
            instructions=instructions.strip(),
        )

    def for_task(self, task_type: TaskType) -> RuntimeSkill | None:
        for metadata in self.list_skills():
            if task_type in metadata.task_types:
                return self.read_skill(metadata.name)
        return None
