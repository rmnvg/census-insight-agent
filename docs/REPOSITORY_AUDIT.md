# Repository Readiness Audit

Audit date: 2026-09-06. This is a concise, value-free record of the Prompt 7 inspection.

## Findings and actions

| Area | Finding | Action/status |
|---|---|---|
| Structure | Backend, frontend, executor, skills, evals, scripts, data contracts, workspace, and docs are separated. | Retained. |
| Worktree | The audit started from a clean worktree at `556825b`. | Prompt 7 changes remain uncommitted for review. |
| Large files | No tracked file exceeded 254 KB. The local ignored corpus is about 115 MB; local generated data is about 51 MB. | Corpus remains ignored with explicit setup instructions. |
| Git history | No credential-like value or historical large corpus blob was detected. Commit metadata contains the developer's Git author email. | No history rewrite; author metadata is normal repository provenance. |
| Generated state | Checkpoints, traces, queue files, artifacts, embedding caches, reports, and generated manifests exist locally. | Ignored; none is required by source code. |
| Secrets | No private key, Google API key, bearer token, service-account secret field, database password, or personal absolute path was found in tracked blobs/history. | Reproducible scanner added. `.env.example` contains placeholders only. |
| Dependencies | Direct production/tool dependencies used ranges even though the lockfile was exact. Qdrant client 1.19 exceeded the server 1.15 line. | Direct dependencies pinned; client 1.16 is within one minor of server 1.15 and retains required metadata support; lock updated. |
| Images | Dockerfiles and Qdrant used mutable tags. | Existing reviewed tags are now paired with content digests. |
| Documentation | Setup was extensive but lacked a clean-clone boundary, final reviewer notes, and a guarded ten-case harness. | README reorganized; readiness, interview, video, and checklist docs added. |
| Commands | Old one-off replay commands are intentionally retained as regression diagnostics; production `pytest` is not directly on `PATH`. | Canonical `make check`/`make verify-offline` added; docs use `uv run --frozen`. |
| Links | Repository-relative Markdown links resolve in the submitted tree. | Covered by final audit. |
| Dead/duplicate code | Replay scripts overlap by scenario, not implementation contract, and encode distinct historical regressions. | No blind deletion. Reassess after submission. |

## Known third-party warnings

- The live LangChain tool-call probe may emit the Google GenAI recommendation about chat-based automatic function calling. Application code already uses `ChatGoogleGenerativeAI` configured for Vertex AI; changing provider behavior solely to suppress a dependency-layer warning was judged higher risk. The structured tool-call result remains validated.
- Starlette 1.6.0 references an AnyIO alias that AnyIO 4.15 deprecates. The warning originates in Starlette's test client, not application code. The compatible locked stack is retained until an upstream release removes the alias.
- Embedded/local Qdrant tests warn that payload indexes have no effect locally. Production uses the pinned Qdrant server, where those indexes do apply.

## Runtime-only files that must not be committed

`.env`, ADC/service-account JSON, `workspace/checkpoints.sqlite*`, `workspace/sessions/`, `workspace/execution-queue/`, generated artifacts, `data/processed/`, generated manifests, source corpus files unless licensing/size policy changes, Python/tool caches, local Qdrant storage, and live evaluation reports.
