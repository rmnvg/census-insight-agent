import os
import re
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from backend.app.tables.models import TABLE_STORE_VERSION, DocumentTables

_DOCUMENT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")


def tables_path(data_root: Path, document_id: str) -> Path:
    if not _DOCUMENT_ID.fullmatch(document_id):
        raise ValueError(f"Invalid document ID: {document_id!r}")
    return data_root / "processed" / "tables" / f"{document_id}.json"


def write_document_tables(data_root: Path, tables: DocumentTables) -> Path:
    path = tables_path(data_root, tables.document_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(tables.model_dump_json(indent=1), encoding="utf-8")
    os.replace(temporary, path)
    return path


def load_document_tables(data_root: Path, document_id: str) -> DocumentTables | None:
    """The document's table store, or None when it is missing, unreadable, or outdated."""
    try:
        path = tables_path(data_root, document_id)
        tables = DocumentTables.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError):
        return None
    if tables.version != TABLE_STORE_VERSION or tables.document_id != document_id:
        return None
    return tables
