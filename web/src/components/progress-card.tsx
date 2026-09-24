"use client";

import { Check, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";

import { formatDuration } from "@/lib/dates";

import type { Pending } from "./chat-provider";
import { LogoMark } from "./logo";

export function ProgressCard({ pending }: { pending: Pending | null }) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(timer);
  }, []);

  const steps = pending?.steps ?? [];
  // Consecutive repeats (e.g. retrieval rounds) collapse into one visible step.
  const visible = steps.filter((step, index) => index === 0 || steps[index - 1].node !== step.node);

  return (
    <div className="flex gap-3">
      <LogoMark className="mt-0.5 size-7 shrink-0" />
      <div className="min-w-0 flex-1 rounded-2xl border border-zinc-200 bg-zinc-50/60 p-4 dark:border-zinc-800 dark:bg-zinc-900/60">
        <div className="mb-3 flex items-center justify-between text-xs text-zinc-500">
          <span className="font-medium text-zinc-700 dark:text-zinc-300">
            {pending ? "Researching the Census reports" : "Still working on this answer…"}
          </span>
          {pending && <span className="tabular-nums">{formatDuration(now - pending.startedAt)}</span>}
        </div>
        <ol className="space-y-1.5">
          {visible.length === 0 && (
            <li className="flex items-center gap-2 text-sm text-zinc-500">
              <Loader2 className="size-3.5 animate-spin text-teal-600" />
              {pending ? "Starting…" : "The agent is still running; this updates automatically."}
            </li>
          )}
          {visible.map((step, index) => {
            const current = index === visible.length - 1;
            return (
              <li key={`${step.node}-${step.at}`} className="animate-step-in flex items-center gap-2 text-sm">
                {current ? (
                  <Loader2 className="size-3.5 shrink-0 animate-spin text-teal-600" />
                ) : (
                  <Check className="size-3.5 shrink-0 text-emerald-600" />
                )}
                <span className={current ? "text-zinc-900 dark:text-zinc-100" : "text-zinc-500"}>{step.label}</span>
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}
