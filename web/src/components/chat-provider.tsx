"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { api, ApiRequestError, streamChat } from "@/lib/api";
import type {
  ApiError,
  DocumentSummary,
  ProgressStep,
  SessionRecord,
  SessionSummary,
  TranscriptEntry,
} from "@/lib/types";

export type Pending = { text: string; steps: ProgressStep[]; startedAt: number };

export type Thread = {
  status: "loading" | "ready" | "missing" | "error";
  session: SessionRecord | null;
  messages: TranscriptEntry[];
  pending: Pending | null;
};

type ChatContextValue = {
  sessions: SessionSummary[];
  sessionsLoaded: boolean;
  threads: Record<string, Thread>;
  documents: DocumentSummary[];
  refreshSessions: () => Promise<void>;
  refreshDocuments: () => Promise<void>;
  loadThread: (id: string, options?: { quiet?: boolean }) => Promise<void>;
  send: (sessionId: string | null, text: string) => Promise<string | null>;
  rename: (id: string, title: string) => Promise<void>;
  remove: (id: string) => Promise<void>;
};

const ChatContext = createContext<ChatContextValue | null>(null);

export function useChat(): ChatContextValue {
  const value = useContext(ChatContext);
  if (!value) throw new Error("useChat must be used inside ChatProvider");
  return value;
}

function localEntry(role: "user" | "assistant", fields: Partial<TranscriptEntry>): TranscriptEntry {
  return {
    message_id: `local-${crypto.randomUUID()}`,
    role,
    created_at: new Date().toISOString(),
    content: null,
    response: null,
    error: null,
    ...fields,
  };
}

export function ChatProvider({ children }: { children: ReactNode }) {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sessionsLoaded, setSessionsLoaded] = useState(false);
  const [threads, setThreads] = useState<Record<string, Thread>>({});
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const threadsRef = useRef(threads);
  useEffect(() => {
    threadsRef.current = threads;
  }, [threads]);

  const patchThread = useCallback((id: string, patch: (thread: Thread) => Thread) => {
    setThreads((current) => {
      const existing = current[id] ?? { status: "ready", session: null, messages: [], pending: null };
      return { ...current, [id]: patch(existing) };
    });
  }, []);

  const refreshSessions = useCallback(async () => {
    try {
      setSessions(await api.listSessions());
    } catch {
      // The sidebar keeps its last list; health indicators surface the outage.
    } finally {
      setSessionsLoaded(true);
    }
  }, []);

  const refreshDocuments = useCallback(async () => {
    try {
      setDocuments(await api.documents());
    } catch {
      // Library dialog shows its own error state.
    }
  }, []);

  useEffect(() => {
    // Initial load of external data into state.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refreshSessions();
    void refreshDocuments();
  }, [refreshSessions, refreshDocuments]);

  const loadThread = useCallback(
    async (id: string, options?: { quiet?: boolean }) => {
      if (!options?.quiet) {
        patchThread(id, (thread) => ({
          ...thread,
          status: thread.messages.length || thread.pending ? thread.status : "loading",
        }));
      }
      try {
        const transcript = await api.transcript(id);
        patchThread(id, (thread) => ({
          ...thread,
          status: "ready",
          session: transcript.session,
          // An in-flight turn owns the tail of the conversation until its stream resolves.
          messages: thread.pending ? thread.messages : transcript.messages,
        }));
      } catch (error) {
        const missing = error instanceof ApiRequestError && [404, 422].includes(error.status);
        patchThread(id, (thread) => ({ ...thread, status: missing ? "missing" : "error" }));
      }
    },
    [patchThread],
  );

  const send = useCallback(
    async (sessionId: string | null, text: string) => {
      let id = sessionId;
      if (id && threadsRef.current[id]?.pending) return id;
      if (!id) {
        try {
          const session = await api.createSession();
          id = session.session_id;
          patchThread(id, () => ({ status: "ready", session, messages: [], pending: null }));
          window.history.replaceState(null, "", `/c/${id}`);
        } catch (error) {
          const detail = error instanceof ApiRequestError ? error.error.message : "Could not start a chat.";
          window.alert(detail);
          return null;
        }
      }
      const threadId = id;
      patchThread(threadId, (thread) => ({
        ...thread,
        pending: { text, steps: [], startedAt: Date.now() },
      }));
      const userEntry = localEntry("user", { content: text });
      try {
        const response = await streamChat(threadId, text, {
          onProgress: (node, label) =>
            patchThread(threadId, (thread) =>
              thread.pending
                ? {
                    ...thread,
                    pending: {
                      ...thread.pending,
                      steps: [...thread.pending.steps, { node, label, at: Date.now() }],
                    },
                  }
                : thread,
            ),
        });
        patchThread(threadId, (thread) => ({
          ...thread,
          pending: null,
          messages: [...thread.messages, userEntry, localEntry("assistant", { response })],
        }));
      } catch (error) {
        const apiError: ApiError =
          error instanceof ApiRequestError
            ? error.error
            : { error_code: "CLIENT_ERROR", message: "Something went wrong.", retryable: true };
        if (apiError.error_code === "STREAM_INTERRUPTED" || apiError.error_code === "NETWORK_ERROR") {
          // The backend keeps working after a dropped connection; reload the saved transcript.
          patchThread(threadId, (thread) => ({ ...thread, pending: null }));
          await loadThread(threadId, { quiet: true });
        } else {
          patchThread(threadId, (thread) => ({
            ...thread,
            pending: null,
            messages: [...thread.messages, userEntry, localEntry("assistant", { error: apiError })],
          }));
        }
      }
      void refreshSessions();
      return threadId;
    },
    [loadThread, patchThread, refreshSessions],
  );

  const rename = useCallback(
    async (id: string, title: string) => {
      const session = await api.renameSession(id, title);
      setSessions((current) =>
        current.map((item) => (item.session_id === id ? { ...item, title: session.title } : item)),
      );
      patchThread(id, (thread) => ({ ...thread, session }));
    },
    [patchThread],
  );

  const remove = useCallback(async (id: string) => {
    await api.deleteSession(id);
    setSessions((current) => current.filter((item) => item.session_id !== id));
    setThreads((current) => {
      const next = { ...current };
      delete next[id];
      return next;
    });
  }, []);

  const value = useMemo(
    () => ({
      sessions,
      sessionsLoaded,
      threads,
      documents,
      refreshSessions,
      refreshDocuments,
      loadThread,
      send,
      rename,
      remove,
    }),
    [sessions, sessionsLoaded, threads, documents, refreshSessions, refreshDocuments, loadThread, send, rename, remove],
  );

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}
