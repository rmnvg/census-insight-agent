import { SseParser } from "./sse";
import type {
  ApiError,
  ChatResponse,
  CoverageSummary,
  DocumentSummary,
  RunTrace,
  SessionRecord,
  SessionSummary,
  SessionTranscript,
  UploadJob,
} from "./types";

export class ApiRequestError extends Error {
  constructor(
    public readonly error: ApiError,
    public readonly status: number,
  ) {
    super(error.message);
  }
}

async function toError(response: Response): Promise<ApiRequestError> {
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    // Non-JSON error bodies fall through to the generic message below.
  }
  const value = (body ?? {}) as Partial<ApiError> & { detail?: unknown };
  if (value.error_code && value.message) return new ApiRequestError(value as ApiError, response.status);
  const detail = typeof value.detail === "string" ? value.detail : "The Census service returned an error.";
  return new ApiRequestError(
    { error_code: `HTTP_${response.status}`, message: detail, retryable: response.status >= 500 },
    response.status,
  );
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api/${path}`, { cache: "no-store", ...init });
  } catch {
    throw new ApiRequestError(
      { error_code: "NETWORK_ERROR", message: "Could not reach the Census service.", retryable: true },
      0,
    );
  }
  if (!response.ok) throw await toError(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

const json = (body: unknown): RequestInit => ({
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  listSessions: () => request<SessionSummary[]>("sessions"),
  createSession: () => request<SessionRecord>("sessions", { method: "POST" }),
  transcript: (id: string) => request<SessionTranscript>(`sessions/${id}/messages`),
  renameSession: (id: string, title: string) =>
    request<SessionRecord>(`sessions/${id}`, { method: "PATCH", ...json({ title }) }),
  deleteSession: (id: string) => request<void>(`sessions/${id}`, { method: "DELETE" }),
  trace: (runId: string) => request<RunTrace>(`runs/${runId}/trace`),
  documents: () => request<DocumentSummary[]>("documents"),
  coverage: (id: string) => request<CoverageSummary>(`documents/${encodeURIComponent(id)}/coverage`),
  uploads: () => request<UploadJob[]>("documents/uploads"),
  upload: (id: string) => request<UploadJob>(`documents/uploads/${id}`),
  deleteDocument: (id: string) =>
    request<void>(`documents/${encodeURIComponent(id)}`, { method: "DELETE" }),
  uploadDocument: (file: File, title: string, region: string) => {
    const form = new FormData();
    form.set("file", file);
    form.set("title", title);
    form.set("region", region);
    return request<UploadJob>("documents/upload", { method: "POST", body: form });
  },
  health: async (path: "health" | "health/qdrant" | "health/executor") => {
    try {
      const value = await request<{ status: string }>(path);
      return value.status === "ok" ? "ok" : "error";
    } catch {
      return "error";
    }
  },
};

export function artifactFileUrl(sessionId: string, artifactId: string, filename: string): string {
  return `/api/sessions/${sessionId}/artifacts/${artifactId}/files/${filename}`;
}

export function pageImageUrl(documentId: string, page: number, highlight?: string): string {
  const query = highlight ? `?highlight=${encodeURIComponent(highlight.slice(0, 1800))}` : "";
  return `/api/documents/${encodeURIComponent(documentId)}/pages/${page}${query}`;
}

export type StreamHandlers = {
  onProgress: (node: string, label: string) => void;
};

/** POST /chat/stream and resolve with the final response, reporting progress on the way. */
export async function streamChat(
  sessionId: string,
  message: string,
  handlers: StreamHandlers,
): Promise<ChatResponse> {
  let response: Response;
  try {
    response = await fetch("/api/chat/stream", {
      method: "POST",
      ...json({ session_id: sessionId, message }),
    });
  } catch {
    throw new ApiRequestError(
      { error_code: "NETWORK_ERROR", message: "Could not reach the Census service.", retryable: true },
      0,
    );
  }
  if (!response.ok || !response.body) throw await toError(response);
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  const parser = new SseParser();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    for (const event of parser.push(value)) {
      const data = JSON.parse(event.data);
      if (event.event === "progress") handlers.onProgress(data.node, data.label);
      else if (event.event === "result") return data as ChatResponse;
      else if (event.event === "error") throw new ApiRequestError(data as ApiError, 200);
    }
  }
  throw new ApiRequestError(
    {
      error_code: "STREAM_INTERRUPTED",
      message: "The connection closed before the answer arrived. It will appear in this chat once it finishes.",
      retryable: true,
    },
    0,
  );
}
