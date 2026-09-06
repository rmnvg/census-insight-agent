# Architecture Decisions

## AI platform and authentication

Gemini requests are sent through Vertex AI on Google Cloud. Services authenticate with Application Default Credentials (ADC), using `GOOGLE_APPLICATION_CREDENTIALS` when a service-account credential file is mounted. API-key authentication through `GEMINI_API_KEY` or `GOOGLE_API_KEY` is prohibited.

LangChain integrations must use `langchain_google_genai.ChatGoogleGenerativeAI` in Vertex AI mode. The deprecated `ChatVertexAI` integration is not used.

## Data and retrieval

Qdrant is the project's only vector database.

Original PDFs remain authoritative. Provided Markdown is the preferred extracted representation. PyMuPDF4LLM is reserved for pages without supplied Markdown and for newly received PDFs.

## Trust boundary

The executor is an isolated service. It must never receive Google Cloud credentials, credential volumes, or Google authentication environment variables. Any future executor protocol must pass only the minimum task data required.

## Ingestion and retrieval

PDF identity and page numbering remain authoritative. Supplied Markdown is preferred only after explicit-marker parsing or confident deterministic page alignment. Missing pages use PyMuPDF4LLM. Chunks never cross page boundaries and citation snippets remain verbatim.

Retrieval combines 768-dimensional Gemini dense vectors and local FastEmbed BM25 sparse vectors in one Qdrant collection. Qdrant performs reciprocal-rank fusion, and every result must carry valid page provenance.

## Deferred work

Deliberately out of scope for this submission: document upload/re-ingestion via the UI (the brief
supplies documents on disk), streamed token output, URL-restorable browser sessions, a durable job
broker in place of the filesystem executor queue, and microVM-grade executor isolation. See
`DESIGN.md`'s "Tradeoffs, alternatives, and intentionally skipped work" for the full list and
reasoning.
