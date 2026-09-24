# Census Insight Agent — web client

Next.js (App Router) + TypeScript + Tailwind CSS client for the Census Insight Agent API.

- `src/app/api/[...path]/route.ts`: same-origin proxy to FastAPI, restricted by the allowlist in
  `src/lib/proxy-rules.ts`.
- `src/components/chat-provider.tsx`: session list, per-session threads, and in-flight streams
  (kept in the root layout so a running answer survives switching chats).
- `src/components/*`: chat view, citation → PDF page viewer, artifact cards, trace viewer, library
  and upload dialog.

```shell
npm ci
CENSUS_API_BASE_URL=http://127.0.0.1:8000 npm run dev   # against a running backend
npm run check                                          # typecheck + lint + unit tests
npm run build
```

In Docker Compose this runs as the `web` service on <http://localhost:3000>. See the repository
[README](../README.md) and [DESIGN.md](../DESIGN.md#nextjs-ui-boundary).
