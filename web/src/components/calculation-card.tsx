"use client";

import { Calculator, FileText } from "lucide-react";

import { calculationCards } from "@/lib/calculations";
import { describeDerivation, formatNumber } from "@/lib/trace";
import type { ChatResponse, Citation } from "@/lib/types";

export function CalculationCards({ response, onOpen }: { response: ChatResponse; onOpen: (citation: Citation) => void }) {
  const cards = calculationCards(response);
  if (!cards.length) return null;
  return (
    <section aria-label="Calculations" className="space-y-3">
      {cards.map(({ claim, inputs }) => (
        <details key={claim.claim_id} open className="group rounded-xl border border-teal-700/20 bg-teal-50/50 p-4 dark:border-teal-500/25 dark:bg-teal-950/20">
          <summary className="flex cursor-pointer flex-wrap items-center gap-x-2 gap-y-1 text-sm font-semibold text-teal-900 dark:text-teal-200">
            <Calculator className="size-4" /> Show the calculation
            <span className="ml-auto text-[11px] font-normal text-teal-700 dark:text-teal-400">Computed in code</span>
          </summary>
          <div className="mt-4 space-y-3">
            <div className="grid gap-2 sm:grid-cols-2">
              {inputs.map(({ claim: input, citations }, index) => (
                <div key={`${input.claim_id}-${index}`} className="rounded-lg border border-teal-700/10 bg-white/80 p-3 dark:bg-zinc-900/70">
                  <p className="text-sm font-medium">{input.region || "Source value"}</p>
                  <p className="mt-1 text-2xl font-semibold tabular-nums tracking-tight">
                    {formatNumber(input.value!)}
                    {input.unit && <span className="ml-1 text-xs font-normal text-zinc-500">{input.unit.replaceAll("_", " ")}</span>}
                  </p>
                  <p className="mt-1 text-xs text-zinc-500">{[input.metric?.replaceAll("_", " "), input.year, input.residence_scope, input.population_scope].filter(Boolean).join(" · ")}</p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {citations.map((citation) => (
                      <button key={citation.citation_id} type="button" onClick={() => onOpen(citation)} title={citation.document_title}
                        className="inline-flex items-center gap-1 text-xs font-medium text-teal-700 hover:underline focus-visible:outline-2 focus-visible:outline-offset-2 dark:text-teal-300">
                        <FileText className="size-3" /> View source · p. {citation.page_number}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
            <p className="overflow-x-auto rounded-lg bg-white/80 px-3 py-2 font-mono text-lg font-semibold tabular-nums dark:bg-zinc-900/70">{describeDerivation(claim)}</p>
            <p className="text-sm text-teal-950 dark:text-teal-100">{claim.text}</p>
          </div>
        </details>
      ))}
    </section>
  );
}
