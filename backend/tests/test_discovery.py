import json
from pathlib import Path

import pytest

from backend.app.ingestion.discovery import discover_documents, ensure_within, normalize_filename
from backend.app.ingestion.errors import DocumentPairingError, SourcePathError


def test_filename_normalization() -> None:
    assert normalize_filename("  Karnataka_Census—2021.PDF") == "karnataka-census-2021"


def test_pdf_markdown_pairing_uses_exact_normalized_stem(tmp_path: Path) -> None:
    source = tmp_path / "source"
    pdf_dir = source / "pdf"
    markdown_dir = source / "markdown"
    manifest_dir = tmp_path / "manifests"
    pdf_dir.mkdir(parents=True)
    markdown_dir.mkdir()
    manifest_dir.mkdir()
    (pdf_dir / "Karnataka Census.pdf").write_bytes(b"synthetic")
    markdown = markdown_dir / "karnataka_census.md"
    markdown.write_text("content", encoding="utf-8")

    pairs = discover_documents(source, manifest_dir)

    assert len(pairs) == 1
    assert pairs[0].markdown_path == markdown


def test_manifest_override_pairs_different_filenames(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "pdf").mkdir(parents=True)
    (source / "markdown").mkdir()
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (source / "pdf" / "official.pdf").write_bytes(b"synthetic")
    preferred = source / "markdown" / "human-edited.md"
    preferred.write_text("content", encoding="utf-8")
    (manifests / "pair.override.json").write_text(
        json.dumps(
            {
                "document_id": "karnataka",
                "title": "Karnataka Census",
                "region": "Karnataka",
                "pdf_filename": "official.pdf",
                "markdown_filename": "human-edited.md",
            }
        ),
        encoding="utf-8",
    )

    pair = discover_documents(source, manifests)[0]

    assert pair.document_id == "karnataka"
    assert pair.markdown_path == preferred


def test_source_path_traversal_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()

    with pytest.raises(SourcePathError, match="outside configured source"):
        ensure_within(tmp_path / "outside.pdf", root)


def test_pairing_does_not_use_substring_guess(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "pdf").mkdir(parents=True)
    (source / "markdown").mkdir()
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (source / "pdf" / "census.pdf").write_bytes(b"synthetic")
    (source / "markdown" / "census-notes.md").write_text("notes", encoding="utf-8")

    with pytest.raises(DocumentPairingError, match="no deterministic PDF pairing"):
        discover_documents(source, manifests)
