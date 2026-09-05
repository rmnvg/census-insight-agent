# AGENTS.md

These instructions apply to the entire repository.

## Runtime and tooling

- Use Python 3.12 and `uv` for dependency management and command execution.
- Keep `uv.lock` committed and synchronized with `pyproject.toml`.
- Run `make check` before considering a change complete.

## Architecture decisions

- All Gemini model calls go through Vertex AI on Google Cloud.
- Authenticate to Google Cloud exclusively with Application Default Credentials (ADC).
- Never introduce `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
- Use `langchain_google_genai.ChatGoogleGenerativeAI` configured for Vertex AI. Do not use the deprecated `ChatVertexAI` integration.
- Qdrant is the only vector database.
- The executor is isolated and must never receive GCP credentials, credential mounts, or Google authentication environment variables.
- Original PDFs are the authoritative source documents.
- When supplied, Markdown is the preferred extracted representation of a PDF.
- Use PyMuPDF4LLM only when Markdown pages are missing or when processing newly supplied PDFs.
- Keep unverified OCR quarantined under `data/processed/page-review/`; it is never evidence and
  must not enter chunking, embeddings, Qdrant, retrieval, or answer generation.
- Excluded visual pages indicate an indexing limitation, not that information is absent from the
  authoritative PDF.

## Current scope

The repository contains Vertex AI providers, citation-safe ingestion, Qdrant hybrid retrieval, and
the citation-grounded LangGraph conversational agent. Artifact execution remains deferred.
