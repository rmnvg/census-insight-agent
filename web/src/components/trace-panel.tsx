"use client";

import { Activity, Loader2, Wrench } from "lucide-react";
import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import { formatDuration } from "@/lib/dates";
import { REFUSAL_LABELS, sanitizeTrace, traceDurationMs } from "@/lib/trace";
import type { RunTrace } from "@/lib/types";

import { Dialog } from "./dialog";

export function TracePanel({ runId }: { runId: string }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="flex items-center gap-1 rounded-md px-2 py-1 text-xs hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
      >
        <Activity className="size-3.5" /> Trace
      </button>
      {open && <TraceDialog runId={runId} onClose={() => setOpen(false)} />}
    </>
  );
}

function TraceDialog({ runId, onClose }: { runId: string; onClose: () => void }) {
  const [trace, setTrace] = useState<RunTrace | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    api.trace(runId).then(setTrace, () => setFailed(true));
  }, [runId]);

  const duration = trace ? traceDurationMs(trace) : null;
  const events = trace ? sanitizeTrace(trace) : [];
  return (
    <Dialog
      open
      onClose={onClose}
      title="Execution trace"
      subtitle={<span className="font-mono">run {runId}</span>}
    >
      <div className="space-y-5 p-5 text-sm">
        {failed && <p className="text-zinc-500">The trace is unavailable.</p>}
        {!trace && !failed && <Loader2 className="mx-auto size-5 animate-spin text-zinc-400" />}
        {trace && (
          <>
            <div className="flex flex-wrap gap-2 text-xs">
              <Pill tone={trace.run_status === "completed" ? "good" : "bad"}>{trace.run_status}</Pill>
              <Pill tone={trace.answer_status === "answered" ? "good" : "warn"}>{trace.answer_status}</Pill>
              {trace.refusal_reason && (
                <Pill tone="warn">{REFUSAL_LABELS[trace.refusal_reason] ?? trace.refusal_reason}</Pill>
              )}
              {trace.error_code && <Pill tone="bad">{trace.error_code}</Pill>}
              {duration !== null && <Pill>{formatDuration(duration)} total</Pill>}
            </div>

            {trace.tool_calls.length > 0 && (
              <section>
                <h3 className="mb-2 flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-zinc-400">
                  <Wrench className="size-3.5" /> Tool calls
                </h3>
                <ul className="divide-y divide-zinc-100 rounded-lg ring-1 ring-zinc-200 dark:divide-zinc-800 dark:ring-zinc-800">
                  {trace.tool_calls.map((call, index) => (
                    <li key={index} className="flex items-center gap-3 px-3 py-2 text-xs">
                      <span className="font-mono font-medium">{call.tool_name}</span>
                      <span className={call.status === "ok" ? "text-emerald-600" : "text-red-600"}>{call.status}</span>
                      <span className="text-zinc-500">{call.result_count} results</span>
                      <span className="ml-auto tabular-nums text-zinc-400">{formatDuration(call.latency_ms)}</span>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            <section>
              <h3 className="mb-2 text-xs font-medium uppercase tracking-wider text-zinc-400">Decisions</h3>
              <ol className="relative space-y-3 border-l border-zinc-200 pl-4 dark:border-zinc-800">
                {events.map((event, index) => (
                  <li key={index} className="relative">
                    <span className="absolute -left-[21px] top-1.5 size-2 rounded-full bg-teal-600 ring-4 ring-white dark:ring-zinc-900" />
                    <div className="flex items-baseline gap-2">
                      <span className="font-medium">{event.event.replaceAll("_", " ")}</span>
                      {event.node && <span className="font-mono text-[11px] text-zinc-400">{event.node}</span>}
                      {event.latencyMs !== null && (
                        <span className="ml-auto tabular-nums text-[11px] text-zinc-400">
                          {formatDuration(event.latencyMs)}
                        </span>
                      )}
                    </div>
                    {event.details.length > 0 && (
                      <dl className="mt-1 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
                        {event.details.map(([key, value]) => (
                          <div key={key} className="contents">
                            <dt className="text-zinc-400">{key.replaceAll("_", " ")}</dt>
                            <dd className="min-w-0 break-words font-mono text-zinc-600 dark:text-zinc-300">{value}</dd>
                          </div>
                        ))}
                      </dl>
                    )}
                  </li>
                ))}
              </ol>
            </section>
          </>
        )}
      </div>
    </Dialog>
  );
}

function Pill({ children, tone }: { children: React.ReactNode; tone?: "good" | "warn" | "bad" }) {
  const color =
    tone === "good"
      ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300"
      : tone === "warn"
        ? "bg-amber-50 text-amber-800 dark:bg-amber-950/50 dark:text-amber-300"
        : tone === "bad"
          ? "bg-red-50 text-red-700 dark:bg-red-950/50 dark:text-red-300"
          : "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300";
  return <span className={`rounded-full px-2.5 py-1 font-medium ${color}`}>{children}</span>;
}
