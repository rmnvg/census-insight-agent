"use client";

import { Check, CircleSlash, Gauge, Loader2, Play, ShieldCheck, Target, X } from "lucide-react";
import { useEffect, useState } from "react";

import { api, ApiRequestError } from "@/lib/api";
import type { Scorecard, TrustCaseResult } from "@/lib/types";

import { useChat } from "./chat-provider";

const CATEGORY_LABEL: Record<TrustCaseResult["category"], string> = {
  lookup: "Lookup",
  comparison: "Comparison",
  ranking: "District ranking",
  chart: "Chart",
  refusal: "Must refuse",
};

export function TrustView() {
  const { send } = useChat();
  const [card, setCard] = useState<Scorecard | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.scorecard().then(setCard, (caught) =>
      setError(
        caught instanceof ApiRequestError && caught.status === 404
          ? "No benchmark has been run yet."
          : "The scorecard is unavailable.",
      ),
    );
  }, []);

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-5xl px-4 py-8 pl-14 md:px-8">
        <p className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-teal-700 dark:text-teal-400">
          <ShieldCheck className="size-3.5" /> Trust scorecard
        </p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight md:text-3xl">Measured, not claimed</h1>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-zinc-500 dark:text-zinc-400">
          A fixed benchmark of questions whose answers were verified by hand against the source tables, plus questions the
          agent must refuse. Every live answer is audited independently of the agent’s own validation.
        </p>

        {error && <p className="mt-8 text-sm text-zinc-500">{error}</p>}
        {!card && !error && <Loader2 className="mt-10 size-6 animate-spin text-zinc-400" />}
        {card && (
          <>
            <p className="mt-2 text-xs text-zinc-400">
              Run {new Date(card.generated_at).toLocaleString()} · {card.model} · {card.cases_total} live questions
            </p>
            <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Tile
                icon={Target}
                label="Correct answers"
                value={`${card.answerable_passed}/${card.answerable_total}`}
                hint="Expected values, labels and artifacts all present"
              />
              <Tile
                icon={CircleSlash}
                label="Correct refusals"
                value={`${card.refusal_passed}/${card.refusal_total}`}
                hint="Out-of-scope or unanswerable questions declined"
              />
              <Tile
                icon={ShieldCheck}
                label="Numbers not in their own citation"
                value={String(card.ungrounded_claims)}
                hint={`Across ${card.claims_checked} claims checked`}
                good={card.ungrounded_claims === 0}
              />
              <Tile
                icon={Gauge}
                label="Wrong answers shown"
                value={String(card.wrong_answers)}
                hint={`Median ${card.median_latency_seconds} s per question`}
                good={card.wrong_answers === 0}
              />
            </div>

            <section className="mt-6 grid gap-3 rounded-xl border border-zinc-200 p-4 text-sm text-zinc-600 md:grid-cols-2 dark:border-zinc-800 dark:text-zinc-300">
              <Method label="Grounded numbers">Every numeric claim must appear verbatim in one of its own cited source quotes.</Method>
              <Method label="Honest arithmetic">Every derived value (for example a sex-ratio gap) is recomputed from its operands.</Method>
              <Method label="Correct answers">The hand-verified value, district and artifact type must all be present.</Method>
              <Method label="Refusals">Questions outside the corpus must be declined with no numbers at all.</Method>
            </section>

            <div className="mt-6 overflow-x-auto rounded-xl ring-1 ring-zinc-200 dark:ring-zinc-800">
              <table className="w-full min-w-[720px] text-sm">
                <thead className="bg-zinc-50 text-left text-xs text-zinc-500 dark:bg-zinc-900">
                  <tr>
                    <th className="px-3 py-2 font-medium">Result</th>
                    <th className="px-3 py-2 font-medium">Question</th>
                    <th className="px-3 py-2 font-medium">Verified expectation</th>
                    <th className="px-3 py-2 text-right font-medium">Time</th>
                    <th className="px-3 py-2" />
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800">
                  {card.results.map((result) => (
                    <tr key={result.case_id} className="align-top">
                      <td className="px-3 py-2.5">
                        <span
                          className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${
                            result.passed
                              ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300"
                              : "bg-red-50 text-red-700 dark:bg-red-950/50 dark:text-red-300"
                          }`}
                        >
                          {result.passed ? <Check className="size-3" /> : <X className="size-3" />}
                          {result.passed ? "Pass" : "Fail"}
                        </span>
                        <p className="mt-1 text-[11px] text-zinc-400">{CATEGORY_LABEL[result.category]}</p>
                      </td>
                      <td className="px-3 py-2.5">
                        <p className="text-zinc-800 dark:text-zinc-200">{result.question}</p>
                        <p className="mt-0.5 line-clamp-2 text-xs text-zinc-500">
                          {result.outcome === "error"
                            ? `Error: ${result.error_code}`
                            : result.outcome === "refused"
                              ? "Refused"
                              : result.answer_excerpt}
                        </p>
                        {!result.passed && result.missing.length > 0 && (
                          <p className="mt-0.5 text-xs text-red-600 dark:text-red-400">Missing: {result.missing.join(", ")}</p>
                        )}
                      </td>
                      <td className="px-3 py-2.5">
                        <p className="font-mono text-xs text-zinc-700 dark:text-zinc-300">{result.expected}</p>
                        <p className="mt-0.5 text-[11px] leading-4 text-zinc-400">{result.source_note}</p>
                      </td>
                      <td className="px-3 py-2.5 text-right text-xs tabular-nums text-zinc-500">
                        {result.latency_seconds.toFixed(0)} s{result.attempts > 1 ? ` · ${result.attempts} tries` : ""}
                      </td>
                      <td className="px-3 py-2.5 text-right">
                        <button
                          type="button"
                          onClick={() => void send(null, result.question)}
                          title="Ask this question live in a new chat"
                          className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-teal-700 ring-1 ring-teal-600/30 hover:bg-teal-50 dark:text-teal-300 dark:hover:bg-teal-950/40"
                        >
                          <Play className="size-3" /> Ask
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-4 text-xs text-zinc-500">
              Reproduce with{" "}
              <code className="rounded bg-zinc-100 px-1.5 py-0.5 dark:bg-zinc-800">
                uv run python scripts/trust_benchmark.py --allow-paid-calls
              </code>{" "}
              (each case is a billable agent turn).
            </p>
          </>
        )}
      </div>
    </div>
  );
}

function Tile({
  icon: Icon,
  label,
  value,
  hint,
  good,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  hint: string;
  good?: boolean;
}) {
  return (
    <div className="rounded-xl border border-zinc-200 p-4 dark:border-zinc-800">
      <p className="flex items-center gap-1.5 text-xs text-zinc-500">
        <Icon className="size-3.5" /> {label}
      </p>
      <p className={`mt-1.5 text-3xl font-semibold tabular-nums ${good ? "text-emerald-700 dark:text-emerald-400" : ""}`}>
        {value}
      </p>
      <p className="mt-1 text-[11px] text-zinc-400">{hint}</p>
    </div>
  );
}

function Method({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <p>
      <span className="font-medium text-zinc-800 dark:text-zinc-100">{label}.</span> {children}
    </p>
  );
}
