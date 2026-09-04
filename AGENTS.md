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

## Current scope

The initial repository contains health endpoints and infrastructure only. Do not implement ingestion, retrieval, or LangGraph workflows until that work is explicitly requested.
