# Recorded assignment demo

Recorded September 7, 2026 from the live local Docker application. Synthetic English narration. Model waiting time is edited out; selected frames are held during explanations. The comparison refusal is the actual observed result.

## 0:00 — Project overview

Welcome to Census Insight Agent. This project answers questions about the 2011 India Census reports for Karnataka, Odisha, and Madhya Pradesh. Its goal is to make census analysis useful and auditable: factual claims must connect to evidence. This video shows the real running application, with waiting time edited for length.

## 0:20 — Architecture

The interface is built with Streamlit and calls a Fast API backend. LangGraph coordinates task classification, retrieval, evidence assessment, and answer validation. Gemini runs through Vertex AI with Application Default Credentials. Qdrant combines dense retrieval with local B M 25 sparse search. For charts, an isolated executor receives verified data through a filesystem queue, without network access or cloud credentials.

## 0:48 — Startup and checks

I started the project with Docker Compose. All four application services became healthy, and the existing Qdrant collection contained two thousand and fifty eight indexed points. The project quality gate also passed: three hundred and five tests, formatting, linting, type checks, an offline interface smoke test, a security audit, and a secret scan.

## 1:10 — Live application

Here is the running application at localhost, port eighty five oh one. The sidebar displays service health and source report coverage. A conversation keeps related questions together. I will demonstrate a lookup, a follow up comparison, a generated chart, and an out of scope question.

## 1:28 — 1 / Census lookup

First, I ask: What was Karnataka's literacy rate in 2011? The live answer reports seventy five point three six percent. It includes verified claims and source references to the Karnataka report. The answer is useful because we can inspect the evidence behind the number.

## 1:45 — 2 / Inspect the citation

Expanding a source citation reveals the original excerpt and the physical PDF page. Citation validation uses document identity, page provenance, and exact source text. Retrieval only finds candidate evidence; application code must still validate the claim. If that validation fails, the system can refuse instead of guessing.

## 2:06 — 3 / Conversational comparison

Next, I ask: How does that compare with Odisha? In this live run, the application found relevant evidence but could not produce a response that passed citation validation. It therefore refused. This demonstrates the validation boundary, but also a current reliability limitation: a supported question does not always result in a usable answer. This recording preserves the actual outcome.

## 2:30 — 4 / Generated chart

Now I request a bar chart comparing total persons literacy rates for Karnataka and Odisha in 2011. The model proposes data, and application code verifies the cells against trusted evidence before execution. The isolated worker produces the chart and companion files. The interface exposes the chart, plotted data, and source manifest for review and download.

## 2:53 — 5 / Safe refusal

Finally, I ask about France's unemployment rate in 2011. This question is outside the supplied Indian census corpus. The application refuses to invent an answer. That boundary matters: missing evidence should remain visible to the user.

## 3:08 — Limitations and handoff

The original PDFs remain the authoritative sources, with supplied Markdown preferred for extraction. Twelve unverified Karnataka visual pages are excluded from indexing; that is a coverage limitation, not proof that information is absent from the PDF. Docker isolation is designed for development, and production needs stronger sandboxing. The README explains local setup, source files, and Vertex authentication. This completes the project walkthrough.
