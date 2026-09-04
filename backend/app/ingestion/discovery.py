import hashlib
import json
import re
import unicodedata
import uuid
from pathlib import Path

from backend.app.ingestion.errors import DocumentPairingError, SourcePathError
from backend.app.ingestion.models import DocumentPair, ManifestOverride


def normalize_filename(filename: str) -> str:
    """Normalize a filename stem for exact, deterministic pairing."""
    stem = unicodedata.normalize("NFKC", Path(filename).stem).casefold()
    return re.sub(r"[^a-z0-9]+", "-", stem).strip("-")


def ensure_within(path: Path, root: Path) -> Path:
    """Resolve a path and reject traversal outside an approved root."""
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise SourcePathError(
            f"Path is outside configured source directory: {path.name}"
        ) from error
    return resolved


def source_checksum(pdf_path: Path) -> str:
    digest = hashlib.sha256()
    with pdf_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_overrides(manifest_dir: Path, source_root: Path) -> list[ManifestOverride]:
    if not manifest_dir.exists():
        return []
    overrides: list[ManifestOverride] = []
    for path in sorted(manifest_dir.glob("*.override.json")):
        ensure_within(path, manifest_dir)
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw if isinstance(raw, list) else [raw]
        overrides.extend(ManifestOverride.model_validate(item) for item in items)
    filenames = {override.pdf_filename for override in overrides}
    if len(filenames) != len(overrides):
        raise DocumentPairingError("Manifest overrides contain duplicate PDF filenames")
    for override in overrides:
        ensure_within(source_root / "pdf" / override.pdf_filename, source_root)
        if override.markdown_filename:
            ensure_within(source_root / "markdown" / override.markdown_filename, source_root)
    return overrides


def discover_documents(source_root: Path, manifest_dir: Path) -> list[DocumentPair]:
    """Pair PDFs and Markdown by explicit override or exact normalized stem."""
    root = source_root.resolve()
    pdf_dir = ensure_within(root / "pdf", root)
    markdown_dir = ensure_within(root / "markdown", root)
    pdfs = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    markdown_files = sorted(markdown_dir.glob("*.md")) if markdown_dir.exists() else []
    overrides = {item.pdf_filename: item for item in _load_overrides(manifest_dir, root)}
    missing_override_pdfs = sorted(set(overrides) - {path.name for path in pdfs})
    if missing_override_pdfs:
        raise DocumentPairingError(
            f"Manifest overrides reference missing PDFs: {', '.join(missing_override_pdfs)}"
        )

    markdown_by_key: dict[str, Path] = {}
    duplicate_keys: set[str] = set()
    for path in markdown_files:
        key = normalize_filename(path.name)
        if key in markdown_by_key:
            duplicate_keys.add(key)
        markdown_by_key[key] = path
    if duplicate_keys:
        duplicates = ", ".join(sorted(duplicate_keys))
        raise DocumentPairingError(f"Ambiguous normalized Markdown filenames: {duplicates}")

    pairs: list[DocumentPair] = []
    used_markdown: set[Path] = set()
    for pdf_path in pdfs:
        override = overrides.get(pdf_path.name)
        key = normalize_filename(pdf_path.name)
        if override:
            markdown_path = (
                markdown_dir / override.markdown_filename if override.markdown_filename else None
            )
            if markdown_path and not markdown_path.is_file():
                raise DocumentPairingError(
                    f"Override Markdown does not exist for {pdf_path.name}: {markdown_path.name}"
                )
            pair = DocumentPair(
                document_id=override.document_id,
                title=override.title,
                region=override.region,
                pdf_path=pdf_path,
                markdown_path=markdown_path,
            )
        else:
            markdown_path = markdown_by_key.get(key)
            pair = DocumentPair(
                document_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"census-document:{key}")),
                title=Path(pdf_path.name).stem.replace("_", " ").replace("-", " ").strip(),
                region="unspecified",
                pdf_path=pdf_path,
                markdown_path=markdown_path,
            )
        if pair.markdown_path:
            if pair.markdown_path in used_markdown:
                raise DocumentPairingError(
                    f"Markdown paired to more than one PDF: {pair.markdown_path.name}"
                )
            used_markdown.add(pair.markdown_path)
        pairs.append(pair)
    orphaned_markdown = sorted(path.name for path in set(markdown_files) - used_markdown)
    if orphaned_markdown:
        raise DocumentPairingError(
            f"Markdown has no deterministic PDF pairing: {', '.join(orphaned_markdown)}"
        )
    return pairs
