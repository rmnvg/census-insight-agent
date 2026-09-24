"use client";

import {
  Check,
  Library,
  Loader2,
  Moon,
  MoreHorizontal,
  Pencil,
  Search,
  ShieldCheck,
  SquarePen,
  Sun,
  Trash2,
  X,
} from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";

import { api } from "@/lib/api";
import { groupSessions } from "@/lib/dates";
import type { HealthStatus, SessionSummary } from "@/lib/types";

import { useChat } from "./chat-provider";
import { LogoMark } from "./logo";

type Props = {
  activeId: string | null;
  onNewChat: () => void;
  onOpenLibrary: () => void;
};

export function Sidebar({ activeId, onNewChat, onOpenLibrary }: Props) {
  const { sessions, sessionsLoaded, threads, documents } = useChat();
  const [query, setQuery] = useState("");

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const list = needle
      ? sessions.filter((session) => (session.title ?? "").toLowerCase().includes(needle))
      : sessions;
    return groupSessions(list);
  }, [sessions, query]);

  return (
    <div className="flex h-full flex-col border-r border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900">
      <div className="flex items-center gap-2.5 px-4 pb-3 pt-4">
        <LogoMark />
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold tracking-tight">Census Insight</p>
          <p className="text-[11px] text-zinc-500 dark:text-zinc-400">Citation-grounded agent</p>
        </div>
      </div>

      <div className="space-y-2 px-3">
        <button
          type="button"
          onClick={onNewChat}
          className="flex w-full items-center gap-2 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm font-medium shadow-sm transition hover:border-zinc-300 hover:bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-800 dark:hover:bg-zinc-700"
        >
          <SquarePen className="size-4" />
          New chat
          <kbd className="ml-auto hidden font-sans text-[11px] text-zinc-400 md:inline">⇧⌘O</kbd>
        </button>
        <label className="flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm text-zinc-500 focus-within:bg-white focus-within:ring-1 focus-within:ring-zinc-300 dark:focus-within:bg-zinc-800 dark:focus-within:ring-zinc-700">
          <Search className="size-4 shrink-0" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search chats"
            className="w-full bg-transparent text-zinc-900 outline-none placeholder:text-zinc-400 dark:text-zinc-100"
          />
        </label>
      </div>

      <nav className="mt-3 flex-1 overflow-y-auto px-2 pb-4" aria-label="Chat history">
        {!sessionsLoaded ? (
          <div className="space-y-2 px-2 pt-2">
            {[0, 1, 2, 3].map((index) => (
              <div key={index} className="h-7 animate-pulse rounded-md bg-zinc-200/70 dark:bg-zinc-800" />
            ))}
          </div>
        ) : visible.length === 0 ? (
          <p className="px-3 pt-2 text-xs text-zinc-500">
            {query ? "No chats match your search." : "Your conversations will appear here."}
          </p>
        ) : (
          visible.map(({ group, sessions: items }) => (
            <section key={group} className="mb-3">
              <h3 className="px-3 pb-1 pt-2 text-[11px] font-medium uppercase tracking-wider text-zinc-400">
                {group}
              </h3>
              <ul className="space-y-0.5">
                {items.map((session) => (
                  <SessionRow
                    key={session.session_id}
                    session={session}
                    active={session.session_id === activeId}
                    busy={Boolean(threads[session.session_id]?.pending)}
                    onDeleted={() => {
                      if (session.session_id === activeId) onNewChat();
                    }}
                  />
                ))}
              </ul>
            </section>
          ))
        )}
      </nav>

      <div className="space-y-1 border-t border-zinc-200 p-3 dark:border-zinc-800">
        <button
          type="button"
          onClick={onOpenLibrary}
          className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm hover:bg-zinc-200/60 dark:hover:bg-zinc-800"
        >
          <Library className="size-4" />
          Document library
          <span className="ml-auto rounded-full bg-zinc-200 px-2 py-0.5 text-[11px] font-medium text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
            {documents.length}
          </span>
        </button>
        <Link
          href="/trust"
          className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm hover:bg-zinc-200/60 dark:hover:bg-zinc-800"
        >
          <ShieldCheck className="size-4" />
          Trust scorecard
        </Link>
        <div className="flex items-center justify-between px-3 pt-1">
          <ServiceHealth />
          <ThemeToggle />
        </div>
      </div>
    </div>
  );
}

function SessionRow({
  session,
  active,
  busy,
  onDeleted,
}: {
  session: SessionSummary;
  active: boolean;
  busy: boolean;
  onDeleted: () => void;
}) {
  const { rename, remove } = useChat();
  const [menuOpen, setMenuOpen] = useState(false);
  const [editing, setEditing] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [draft, setDraft] = useState(session.title ?? "");
  const menuRef = useRef<HTMLDivElement>(null);
  const title = session.title || "Untitled chat";

  useEffect(() => {
    if (!menuOpen) return;
    const close = (event: MouseEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) setMenuOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [menuOpen]);

  const base = `group relative flex items-center rounded-lg text-sm transition ${
    active
      ? "bg-zinc-200/80 text-zinc-900 dark:bg-zinc-800 dark:text-white"
      : "text-zinc-700 hover:bg-zinc-200/50 dark:text-zinc-300 dark:hover:bg-zinc-800/60"
  }`;

  if (editing) {
    const commit = async () => {
      const value = draft.trim();
      setEditing(false);
      if (value && value !== session.title) await rename(session.session_id, value).catch(() => undefined);
    };
    return (
      <li className={base}>
        <input
          autoFocus
          value={draft}
          maxLength={120}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={commit}
          onKeyDown={(event) => {
            if (event.key === "Enter") void commit();
            if (event.key === "Escape") setEditing(false);
          }}
          className="w-full rounded-lg bg-white px-3 py-1.5 outline-none ring-1 ring-teal-600 dark:bg-zinc-900"
        />
      </li>
    );
  }

  if (confirming) {
    return (
      <li className={`${base} gap-1 bg-red-50 px-3 py-1.5 dark:bg-red-950/40`}>
        <span className="min-w-0 flex-1 truncate text-red-700 dark:text-red-300">Delete this chat?</span>
        <button
          type="button"
          aria-label="Confirm delete"
          className="rounded p-1 text-red-700 hover:bg-red-100 dark:text-red-300 dark:hover:bg-red-900/50"
          onClick={async () => {
            setConfirming(false);
            await remove(session.session_id).catch(() => undefined);
            onDeleted();
          }}
        >
          <Check className="size-4" />
        </button>
        <button
          type="button"
          aria-label="Cancel delete"
          className="rounded p-1 hover:bg-zinc-200 dark:hover:bg-zinc-800"
          onClick={() => setConfirming(false)}
        >
          <X className="size-4" />
        </button>
      </li>
    );
  }

  return (
    <li className={base}>
      <Link
        href={`/c/${session.session_id}`}
        className="flex min-w-0 flex-1 items-center gap-2 px-3 py-1.5"
        title={title}
      >
        {busy && <Loader2 className="size-3.5 shrink-0 animate-spin text-teal-600" />}
        <span className="truncate">{title}</span>
      </Link>
      <div ref={menuRef} className="relative pr-1">
        <button
          type="button"
          aria-label="Chat options"
          onClick={() => setMenuOpen((open) => !open)}
          className={`rounded-md p-1 text-zinc-500 hover:bg-zinc-300/60 dark:hover:bg-zinc-700 ${
            menuOpen ? "opacity-100" : "opacity-0 group-hover:opacity-100 focus:opacity-100"
          }`}
        >
          <MoreHorizontal className="size-4" />
        </button>
        {menuOpen && (
          <div className="absolute right-0 top-8 z-50 w-36 overflow-hidden rounded-lg border border-zinc-200 bg-white py-1 text-sm shadow-lg dark:border-zinc-700 dark:bg-zinc-800">
            <button
              type="button"
              className="flex w-full items-center gap-2 px-3 py-1.5 hover:bg-zinc-100 dark:hover:bg-zinc-700"
              onClick={() => {
                setMenuOpen(false);
                setDraft(session.title ?? "");
                setEditing(true);
              }}
            >
              <Pencil className="size-3.5" /> Rename
            </button>
            <button
              type="button"
              className="flex w-full items-center gap-2 px-3 py-1.5 text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-950/40"
              onClick={() => {
                setMenuOpen(false);
                setConfirming(true);
              }}
            >
              <Trash2 className="size-3.5" /> Delete
            </button>
          </div>
        )}
      </div>
    </li>
  );
}

const SERVICES = [
  { key: "health", label: "API" },
  { key: "health/qdrant", label: "Qdrant" },
  { key: "health/executor", label: "Executor" },
] as const;

function ServiceHealth() {
  const [status, setStatus] = useState<Record<string, HealthStatus>>({});

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      const results = await Promise.all(SERVICES.map((service) => api.health(service.key)));
      if (!cancelled) {
        setStatus(Object.fromEntries(SERVICES.map((service, index) => [service.key, results[index]])));
      }
    };
    void check();
    const timer = window.setInterval(check, 30_000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  return (
    <div className="flex items-center gap-3">
      {SERVICES.map((service) => {
        const value = status[service.key] ?? "unknown";
        const color =
          value === "ok" ? "bg-emerald-500" : value === "error" ? "bg-red-500" : "bg-zinc-400";
        return (
          <span
            key={service.key}
            className="flex items-center gap-1 text-[11px] text-zinc-500"
            title={`${service.label}: ${value === "ok" ? "healthy" : value === "error" ? "unavailable" : "checking"}`}
          >
            <span className={`size-1.5 rounded-full ${color}`} />
            {service.label}
          </span>
        );
      })}
    </div>
  );
}

function ThemeToggle() {
  const toggle = () => {
    const dark = !document.documentElement.classList.contains("dark");
    document.documentElement.classList.toggle("dark", dark);
    try {
      localStorage.setItem("theme", dark ? "dark" : "light");
    } catch {
      // Theme still applies for this page view.
    }
  };
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label="Toggle theme"
      className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-200/60 dark:hover:bg-zinc-800"
    >
      <Sun className="hidden size-4 dark:block" />
      <Moon className="size-4 dark:hidden" />
    </button>
  );
}
