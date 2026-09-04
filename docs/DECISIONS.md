# Architecture Decisions

## AI platform and authentication

Gemini requests are sent through Vertex AI on Google Cloud. Services authenticate with Application Default Credentials (ADC), using `GOOGLE_APPLICATION_CREDENTIALS` when a service-account credential file is mounted. API-key authentication through `GEMINI_API_KEY` or `GOOGLE_API_KEY` is prohibited.

LangChain integrations must use `langchain_google_genai.ChatGoogleGenerativeAI` in Vertex AI mode. The deprecated `ChatVertexAI` integration is not used.

## Data and retrieval

Qdrant is the project's only vector database.

Original PDFs remain authoritative. Provided Markdown is the preferred extracted representation. PyMuPDF4LLM is reserved for pages without supplied Markdown and for newly received PDFs.

## Trust boundary

The executor is an isolated service. It must never receive Google Cloud credentials, credential volumes, or Google authentication environment variables. Any future executor protocol must pass only the minimum task data required.

## Deferred work

Ingestion, retrieval, and LangGraph orchestration are intentionally deferred from the initial scaffold.
