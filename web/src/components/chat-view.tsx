"use client";

import {
  ArrowUp,
  BarChart3,
  Calculator,
  FileSearch,
  Library,
  ShieldCheck,
  ShieldX,
  Table2,
  Trophy,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

import type { TranscriptEntry } from "@/lib/types";

import { AssistantMessage, ErrorMessage } from "./assistant-message";
import { useChat } from "./chat-provider";
import { LogoMark } from "./logo";
import { ProgressCard } from "./progress-card";

const MAX_MESSAGE = 4000;

const SUGGESTIONS = [
  { icon: FileSearch, label: "Lookup", prompt: "What was the literacy rate of Karnataka in 2011?" },
  { icon: Calculator, label: "Compare", prompt: "Compare the sex ratio of Odisha and Madhya Pradesh." },
  {
    icon: BarChart3,
    label: "Chart",
    prompt: "Create a bar chart comparing the literacy rates of Karnataka and Odisha.",
  },
  { icon: Trophy, label: "Rank", prompt: "Which district of Madhya Pradesh had the highest sex ratio?" },
  { icon: Table2, label: "Table", prompt: "Make a table of the total population of Odisha by rural and urban areas." },
  { icon: ShieldX, label: "Out of scope", prompt: "What was the GDP of France in 2011?" },
];

type Props = { sessionId: string | null; onOpenLibrary: () => void };

export function ChatView({ sessionId, onOpenLibrary }: Props) {
  const { threads, loadThread, send, documents, sessions } = useChat();
  const thread = sessionId ? threads[sessionId] : undefined;
  const [draft, setDraft] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const messages = thread?.messages ?? [];
  const pending = thread?.pending ?? null;
  // A reload during a running turn leaves a trailing user message with no answer yet.
  const awaiting = !pending && messages.at(-1)?.role === "user";

  useEffect(() => {
    if (sessionId && !threads[sessionId]) void loadThread(sessionId);
  }, [sessionId, threads, loadThread]);

  useEffect(() => {
    if (!sessionId || !awaiting) return;
    let attempts = 0;
    const timer = window.setInterval(() => {
      attempts += 1;
      if (attempts > 75) window.clearInterval(timer);
      void loadThread(sessionId, { quiet: true });
    }, 4000);
    return () => window.clearInterval(timer);
  }, [sessionId, awaiting, loadThread]);

  const stepCount = pending?.steps.length ?? -1;
  useEffect(() => {
    const element = scrollRef.current;
    if (element) element.scrollTo({ top: element.scrollHeight, behavior: "smooth" });
  }, [messages.length, stepCount]);

  const submit = async (text: string) => {
    const value = text.trim();
    if (!value || pending) return;
    setDraft("");
    await send(sessionId, value);
  };

  const lastUserPrompt = [...messages].reverse().find((entry) => entry.role === "user")?.content;
  const empty = !messages.length && !pending && thread?.status !== "loading";

  if (thread?.status === "missing") {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-2 p-8 text-center">
        <p className="text-lg font-medium">This chat doesn’t exist</p>
        <p className="text-sm text-zinc-500">It may have been deleted. Start a new chat from the sidebar.</p>
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <header className="flex h-14 shrink-0 items-center gap-3 border-b border-zinc-200/70 px-4 pl-14 md:pl-6 dark:border-zinc-800/70">
        <h1 className="min-w-0 truncate text-sm font-medium">
          {thread?.session?.title ??
            sessions.find((item) => item.session_id === sessionId)?.title ??
            messages.find((entry) => entry.role === "user")?.content ??
            pending?.text ??
            "New chat"}
        </h1>
        <span className="ml-auto hidden items-center gap-1.5 rounded-full bg-emerald-50 px-2.5 py-1 text-[11px] font-medium text-emerald-700 sm:flex dark:bg-emerald-950/50 dark:text-emerald-400">
          <ShieldCheck className="size-3.5" />
          Citations verified in code
        </span>
      </header>

      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        {empty ? (
          <EmptyState documents={documents.map((doc) => doc.region)} onPick={submit} onOpenLibrary={onOpenLibrary} />
        ) : (
          <div className="mx-auto w-full max-w-3xl space-y-8 px-4 py-8 md:px-6">
            {thread?.status === "loading" && <ThreadSkeleton />}
            {messages.map((entry, index) => (
              <Entry
                key={entry.message_id}
                entry={entry}
                onRetry={
                  entry.error?.retryable && index === messages.length - 1 && lastUserPrompt
                    ? () => submit(lastUserPrompt)
                    : undefined
                }
              />
            ))}
            {pending && (
              <>
                <UserBubble text={pending.text} />
                <ProgressCard pending={pending} />
              </>
            )}
            {awaiting && <ProgressCard pending={null} />}
          </div>
        )}
      </div>

      <Composer
        value={draft}
        onChange={setDraft}
        onSubmit={() => submit(draft)}
        disabled={Boolean(pending) || awaiting}
      />
    </div>
  );
}

function Entry({ entry, onRetry }: { entry: TranscriptEntry; onRetry?: () => void }) {
  if (entry.role === "user") return <UserBubble text={entry.content ?? ""} />;
  if (entry.response) return <AssistantMessage response={entry.response} />;
  if (entry.error) return <ErrorMessage error={entry.error} onRetry={onRetry} />;
  return null;
}

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-zinc-100 px-4 py-2.5 text-[15px] leading-relaxed dark:bg-zinc-800">
        {text}
      </div>
    </div>
  );
}

function ThreadSkeleton() {
  return (
    <div className="space-y-6">
      <div className="ml-auto h-10 w-2/3 animate-pulse rounded-2xl bg-zinc-100 dark:bg-zinc-800" />
      <div className="space-y-2">
        <div className="h-4 w-full animate-pulse rounded bg-zinc-100 dark:bg-zinc-800" />
        <div className="h-4 w-5/6 animate-pulse rounded bg-zinc-100 dark:bg-zinc-800" />
        <div className="h-4 w-2/3 animate-pulse rounded bg-zinc-100 dark:bg-zinc-800" />
      </div>
    </div>
  );
}

function EmptyState({
  documents,
  onPick,
  onOpenLibrary,
}: {
  documents: string[];
  onPick: (prompt: string) => void;
  onOpenLibrary: () => void;
}) {
  const regions = [...new Set(documents)];
  return (
    <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col justify-center px-4 py-10 md:px-6">
      <div className="mb-8 flex flex-col items-center text-center">
        <LogoMark className="mb-5 size-12" />
        <h2 className="text-2xl font-semibold tracking-tight md:text-3xl">
          Ask the Census, get answers you can verify
        </h2>
        <p className="mt-3 max-w-xl text-sm leading-6 text-zinc-500 dark:text-zinc-400">
          Every figure is traced to an exact page of the source PDF, arithmetic is computed in code, and the
          agent declines when the evidence isn’t there.
        </p>
        <button
          type="button"
          onClick={onOpenLibrary}
          className="mt-4 flex flex-wrap items-center justify-center gap-1.5 text-xs text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"
        >
          <Library className="size-3.5" />
          {regions.length ? (
            <>
              Searching {regions.length} report{regions.length === 1 ? "" : "s"}:
              {regions.map((region) => (
                <span key={region} className="rounded-full bg-zinc-100 px-2 py-0.5 dark:bg-zinc-800">
                  {region}
                </span>
              ))}
            </>
          ) : (
            "Open the document library"
          )}
        </button>
      </div>
      <div className="grid gap-2.5 sm:grid-cols-2">
        {SUGGESTIONS.map(({ icon: Icon, label, prompt }) => (
          <button
            key={prompt}
            type="button"
            onClick={() => onPick(prompt)}
            className="group flex items-start gap-3 rounded-xl border border-zinc-200 p-3.5 text-left transition hover:border-teal-600/40 hover:bg-teal-50/40 dark:border-zinc-800 dark:hover:border-teal-500/40 dark:hover:bg-teal-950/20"
          >
            <span className="rounded-lg bg-zinc-100 p-1.5 text-zinc-600 transition group-hover:bg-teal-100 group-hover:text-teal-700 dark:bg-zinc-800 dark:text-zinc-300 dark:group-hover:bg-teal-900/50 dark:group-hover:text-teal-300">
              <Icon className="size-4" />
            </span>
            <span className="min-w-0">
              <span className="block text-[11px] font-medium uppercase tracking-wider text-zinc-400">{label}</span>
              <span className="mt-0.5 block text-sm leading-snug text-zinc-700 dark:text-zinc-300">{prompt}</span>
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

function Composer({
  value,
  onChange,
  onSubmit,
  disabled,
}: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  disabled: boolean;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 200)}px`;
  }, [value]);

  useEffect(() => {
    if (!disabled) ref.current?.focus();
  }, [disabled]);

  const tooLong = value.length > MAX_MESSAGE;
  return (
    <div className="shrink-0 px-4 pb-4 pt-2 md:px-6">
      <form
        className="mx-auto w-full max-w-3xl"
        onSubmit={(event) => {
          event.preventDefault();
          if (!tooLong) onSubmit();
        }}
      >
        <div className="flex items-end gap-2 rounded-2xl border border-zinc-300 bg-white p-2 pl-4 shadow-sm transition focus-within:border-zinc-400 focus-within:shadow-md dark:border-zinc-700 dark:bg-zinc-900 dark:focus-within:border-zinc-600">
          <textarea
            ref={ref}
            rows={1}
            value={value}
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                if (!disabled && !tooLong) onSubmit();
              }
            }}
            placeholder={disabled ? "Working on your answer…" : "Ask about literacy, population, sex ratio, districts…"}
            className="max-h-[200px] min-h-[36px] flex-1 resize-none bg-transparent py-1.5 text-[15px] outline-none placeholder:text-zinc-400"
          />
          <button
            type="submit"
            disabled={disabled || !value.trim() || tooLong}
            aria-label="Send"
            className="rounded-xl bg-teal-700 p-2 text-white transition hover:bg-teal-800 disabled:bg-zinc-200 disabled:text-zinc-400 dark:disabled:bg-zinc-800 dark:disabled:text-zinc-600"
          >
            <ArrowUp className="size-4" />
          </button>
        </div>
        <p className="mt-2 flex justify-between px-1 text-[11px] text-zinc-400">
          <span>Answers come only from the indexed Census reports. Shift + Enter for a new line.</span>
          {value.length > MAX_MESSAGE - 500 && (
            <span className={tooLong ? "text-red-500" : ""}>
              {value.length}/{MAX_MESSAGE}
            </span>
          )}
        </p>
      </form>
    </div>
  );
}
