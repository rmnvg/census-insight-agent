"""Structured Census statement tables derived from the indexed Markdown.

Qdrant stays the only vector store. A table store is a derived JSON file per document under
`data/processed/tables/`: every statement table with its multi-level header resolved into column
semantics, and every row bound to the indexed chunk that contains it, so a value selected here is
still cited from Qdrant evidence.
"""
