import csv
import hashlib
import json
import stat
import warnings
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from backend.app.execution.contracts import (
    ExpectedArtifact,
    ProducedArtifact,
    SourceManifest,
)

MAX_FILES = 8
MAX_TOTAL_BYTES = 20 * 1024 * 1024
MAX_CSV_ROWS = 10_000
ALLOWED_SUFFIXES = {".png", ".csv", ".md", ".json"}


class ArtifactValidationError(ValueError):
    def __init__(self, message: str, *, missing: bool = False, output_limit: bool = False) -> None:
        super().__init__(message)
        self.missing = missing
        self.output_limit = output_limit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_png(path: Path) -> None:
    if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise ArtifactValidationError(f"Invalid PNG signature: {path.name}")
    Image.MAX_IMAGE_PIXELS = 25_000_000
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                if width < 10 or height < 10:
                    raise ArtifactValidationError(f"PNG dimensions are not meaningful: {path.name}")
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
    ) as error:
        raise ArtifactValidationError(f"Unsafe or invalid PNG: {path.name}") from error


def _validate_csv(path: Path, expected_columns: list[str]) -> None:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != expected_columns:
                raise ArtifactValidationError(f"CSV schema mismatch: {path.name}")
            for count, _ in enumerate(reader, start=1):
                if count > MAX_CSV_ROWS:
                    raise ArtifactValidationError(
                        f"CSV row limit exceeded: {path.name}", output_limit=True
                    )
    except UnicodeDecodeError as error:
        raise ArtifactValidationError(f"CSV is not UTF-8: {path.name}") from error


def _validate_json(path: Path) -> None:
    try:
        SourceManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise ArtifactValidationError(f"Invalid source manifest: {path.name}") from error


def validate_outputs(output_dir: Path, expected: list[ExpectedArtifact]) -> list[ProducedArtifact]:
    declared = {item.filename: item for item in expected}
    entries = list(output_dir.iterdir()) if output_dir.is_dir() else []
    if len(entries) > MAX_FILES:
        raise ArtifactValidationError("Artifact file count exceeded", output_limit=True)
    produced_names = {item.name for item in entries}
    if missing := set(declared) - produced_names:
        raise ArtifactValidationError(
            f"Expected artifacts were not created: {', '.join(sorted(missing))}", missing=True
        )
    if unexpected := produced_names - set(declared):
        raise ArtifactValidationError(
            f"Unexpected artifacts were created: {', '.join(sorted(unexpected))}"
        )
    total = 0
    artifacts: list[ProducedArtifact] = []
    for path in entries:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ArtifactValidationError(f"Non-regular artifact rejected: {path.name}")
        if path.name.startswith(".") or path.suffix.casefold() not in ALLOWED_SUFFIXES:
            raise ArtifactValidationError(f"Artifact extension rejected: {path.name}")
        total += metadata.st_size
        if total > MAX_TOTAL_BYTES:
            raise ArtifactValidationError("Artifact byte limit exceeded", output_limit=True)
        specification = declared[path.name]
        if path.suffix.casefold() == ".png":
            _validate_png(path)
        elif path.suffix.casefold() == ".csv":
            _validate_csv(path, specification.expected_columns)
        elif path.suffix.casefold() == ".json":
            _validate_json(path)
        else:
            try:
                path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as error:
                raise ArtifactValidationError(f"Invalid Markdown: {path.name}") from error
        artifacts.append(
            ProducedArtifact(
                filename=path.name,
                media_type=specification.media_type,
                byte_size=metadata.st_size,
                sha256=_sha256(path),
            )
        )
    return sorted(artifacts, key=lambda item: item.filename)
