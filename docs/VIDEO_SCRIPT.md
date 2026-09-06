# Five-minute Video Script

## 0:00–0:30 — Problem and architecture

Open `README.md` at the Mermaid diagram. Say: “This assistant answers from supplied Census reports, but retrieval alone is not evidence. The backend validates claims, exact quotes, physical PDF pages, and artifact lineage across four separated services.”

## 0:30–1:20 — Start and UI

Show `docker compose ps`, then open <http://localhost:8501>. Point out four healthy services and the source-report coverage cards. Say that first startup may wait for image and local sparse-model loading.

## 1:20–2:10 — Lookup, memory, citations

Ask exactly:

1. “What was Karnataka’s literacy rate in 2011?”
2. “How does that compare with Odisha?”
3. “Which source pages support those values?”

Expand one citation. Show the exact quote and physical page. Explain that the follow-up uses validated structured claim history, not copied conversation prose.

## 2:10–3:10 — Artifacts

Ask: “Create a bar chart comparing the 2011 total persons literacy rates for Karnataka and Odisha.” Show the rendered chart and download `chart.png`, `plotted-data.csv`, and `source-manifest.json`.

Then ask: “Create a table comparing the 2011 total, rural, and urban literacy rates for Karnataka and Odisha.” Show `table.csv`, `table.md`, and its manifest. Explain deterministic evidence hydration and companion-data validation.

## 3:10–4:00 — Trace and skill

Expand “Execution details,” then open `skills/chart.md` and `backend/app/agent/graph.py`. Say: “The trace exposes node/status/count/latency data, never prompts, reasoning, vectors, credentials, or raw chunks. Skills are runtime instructions; they cannot bypass citation or executor policy.”

## 4:00–4:40 — Repository and security

Open `docker-compose.yml`, `scripts/verify_security.py`, and `.github/workflows/ci.yml`. Highlight that only the backend receives read-only ADC; the executor is non-root and network-disabled; the frontend is read-only and sees only FastAPI. Show the latest `make verify-offline` result.

## 4:40–5:00 — Limitations and close

Open `FAILURE_ANALYSIS.md`. Say: “Unsafe visual OCR is intentionally excluded, Docker is not a hardened hostile sandbox, and live model evaluation is opt-in and billable. The core outcome is an auditable path from PDF page to claim or artifact, with safe failure when that path cannot be proven.”
