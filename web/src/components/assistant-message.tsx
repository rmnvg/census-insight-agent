"use client";

import {
  AlertTriangle,
  Calculator,
  Check,
  Copy,
  FileText,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
} from "lucide-react";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { answerIsClaims, snippetPreview, stripDownloadLinks, stripSourcesSection } from "@/lib/answer";
import { citationNumbers, describeDerivation, orderedCitations } from "@/lib/trace";
import type { ApiError, ChatResponse, Citation, Claim } from "@/lib/types";

import { ArtifactCard } from "./artifact-card";
import { LogoMark } from "./logo";
import { PageViewer } from "./page-viewer";
import { TracePanel } from "./trace-panel";

export function AssistantMessage({ response }: { response: ChatResponse }) {
  const [viewing, setViewing] = useState<Citation | null>(null);
  const [copied, setCopied] = useState(false);
  const citations = orderedCitations(response.citations);
  const numbers = citationNumbers(response.citations);
  const byId = new Map(citations.map((citation) => [citation.citation_id, citation]));
  const claims = response.claims.filter((claim) => claim.text.trim());
  const claimsAreAnswer = answerIsClaims(response);
  const body = stripDownloadLinks(stripSourcesSection(response.answer));
  const [showAllClaims, setShowAllClaims] = useState(false);
  const [showAllSources, setShowAllSources] = useState(false);
  const SOURCE_PREVIEW = 6;
  const CLAIM_PREVIEW = 4;
  const visibleClaims = showAllClaims ? claims : claims.slice(0, CLAIM_PREVIEW);

  const copy = async () => {
    await navigator.clipboard.writeText(response.answer).catch(() => undefined);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="flex gap-3">
      <LogoMark className="mt-0.5 size-7 shrink-0" />
      <div className="min-w-0 flex-1 space-y-4">
        <StatusBadge response={response} />

        {claimsAreAnswer ? (
          <div className="answer-prose">
            {claims.map((claim) => (
              <ClaimLine key={claim.claim_id} claim={claim} numbers={numbers} byId={byId} onOpen={setViewing} />
            ))}
          </div>
        ) : (
          <>
            {body && (
              <div className="answer-prose">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{body}</ReactMarkdown>
              </div>
            )}
            {claims.length > 0 && citations.length > 0 && (
              <section>
                <h4 className="mb-2 text-xs font-medium uppercase tracking-wider text-zinc-400">
                  Verified claims · {claims.length}
                </h4>
                <div className="space-y-2 text-sm">
                  {visibleClaims.map((claim) => (
                    <ClaimLine key={claim.claim_id} claim={claim} numbers={numbers} byId={byId} onOpen={setViewing} />
                  ))}
                </div>
                {claims.length > CLAIM_PREVIEW && (
                  <button
                    type="button"
                    onClick={() => setShowAllClaims((value) => !value)}
                    className="mt-2 text-xs font-medium text-teal-700 hover:underline dark:text-teal-400"
                  >
                    {showAllClaims ? "Show fewer" : `Show all ${claims.length} verified claims`}
                  </button>
                )}
              </section>
            )}
          </>
        )}

        {response.limitations.length > 0 && (
          <section className="rounded-xl border border-amber-200 bg-amber-50/70 p-3 text-sm dark:border-amber-900/60 dark:bg-amber-950/30">
            <p className="mb-1 flex items-center gap-1.5 font-medium text-amber-800 dark:text-amber-300">
              <AlertTriangle className="size-4" /> Limitations
            </p>
            <ul className="list-disc space-y-1 pl-5 text-amber-900/90 dark:text-amber-200/90">
              {response.limitations.map((limitation) => (
                <li key={limitation}>{limitation}</li>
              ))}
            </ul>
          </section>
        )}

        {response.artifacts.map((artifact) => (
          <ArtifactCard key={artifact.artifact_id} artifact={artifact} />
        ))}

        {citations.length > 0 && (
          <section>
            <h4 className="mb-2 text-xs font-medium uppercase tracking-wider text-zinc-400">
              Sources · {citations.length}
            </h4>
            <div className="grid gap-2 sm:grid-cols-2">
              {(showAllSources ? citations : citations.slice(0, SOURCE_PREVIEW)).map((citation) => (
                <button
                  key={citation.citation_id}
                  type="button"
                  onClick={() => setViewing(citation)}
                  className="group flex min-w-0 flex-col gap-1.5 rounded-xl border border-zinc-200 p-3 text-left transition hover:border-teal-600/40 hover:shadow-sm dark:border-zinc-800 dark:hover:border-teal-500/40"
                >
                  <span className="flex items-center gap-2 text-xs">
                    <span className="rounded bg-teal-50 px-1.5 font-semibold text-teal-700 dark:bg-teal-950/60 dark:text-teal-300">
                      {numbers.get(citation.citation_id)}
                    </span>
                    <FileText className="size-3.5 text-zinc-400" />
                    <span className="min-w-0 truncate font-medium text-zinc-700 dark:text-zinc-300">
                      {citation.document_title}
                    </span>
                  </span>
                  <span className="line-clamp-2 font-mono text-[11.5px] leading-snug text-zinc-500">
                    {snippetPreview(citation.snippet)}
                  </span>
                  <span className="text-[11px] font-medium text-teal-700 opacity-80 group-hover:opacity-100 dark:text-teal-400">
                    Page {citation.page_number} · View in PDF →
                  </span>
                </button>
              ))}
            </div>
            {citations.length > SOURCE_PREVIEW && (
              <button
                type="button"
                onClick={() => setShowAllSources((value) => !value)}
                className="mt-2 text-xs font-medium text-teal-700 hover:underline dark:text-teal-400"
              >
                {showAllSources ? "Show fewer sources" : `Show all ${citations.length} sources`}
              </button>
            )}
          </section>
        )}

        <div className="flex items-center gap-1 text-zinc-400">
          <button
            type="button"
            onClick={copy}
            className="flex items-center gap-1 rounded-md px-2 py-1 text-xs hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
          >
            {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
            {copied ? "Copied" : "Copy"}
          </button>
          <TracePanel runId={response.trace_id} />
        </div>
      </div>
      {viewing && <PageViewer key={viewing.citation_id} citation={viewing} onClose={() => setViewing(null)} />}
    </div>
  );
}

function ClaimLine({
  claim,
  numbers,
  byId,
  onOpen,
}: {
  claim: Claim;
  numbers: Map<string, number>;
  byId: Map<string, Citation>;
  onOpen: (citation: Citation) => void;
}) {
  const derivation = describeDerivation(claim);
  return (
    <p>
      {claim.text}
      {claim.citation_ids.map((id) => {
        const citation = byId.get(id);
        return citation ? (
          <button
            key={id}
            type="button"
            onClick={() => onOpen(citation)}
            title={`${citation.document_title}, page ${citation.page_number}`}
            className="ml-1 inline-flex -translate-y-0.5 items-center rounded bg-teal-50 px-1.5 align-middle text-[11px] font-semibold leading-5 text-teal-700 ring-1 ring-teal-600/20 hover:bg-teal-100 dark:bg-teal-950/60 dark:text-teal-300 dark:ring-teal-400/20"
          >
            {numbers.get(id)}
          </button>
        ) : null;
      })}
      {derivation && (
        <span className="mt-1.5 flex w-fit items-center gap-1.5 rounded-md bg-zinc-100 px-2 py-0.5 font-mono text-[12px] text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
          <Calculator className="size-3" />
          {derivation}
          <span className="font-sans text-[10px] uppercase tracking-wide text-zinc-400">computed in code</span>
        </span>
      )}
    </p>
  );
}

function StatusBadge({ response }: { response: ChatResponse }) {
  if (response.refusal) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-50 px-2.5 py-1 text-xs font-medium text-amber-800 ring-1 ring-amber-600/20 dark:bg-amber-950/40 dark:text-amber-300">
        <ShieldAlert className="size-3.5" />
        Declined rather than guessed
      </span>
    );
  }
  const sources = orderedCitations(response.citations).length;
  if (!sources) return null;
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-medium text-emerald-700 ring-1 ring-emerald-600/20 dark:bg-emerald-950/40 dark:text-emerald-300">
      <ShieldCheck className="size-3.5" />
      Verified against {sources} source{sources === 1 ? "" : "s"}
    </span>
  );
}

export function ErrorMessage({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  return (
    <div className="flex gap-3">
      <LogoMark className="mt-0.5 size-7 shrink-0 opacity-60" />
      <div className="min-w-0 flex-1 rounded-xl border border-red-200 bg-red-50/60 p-3.5 text-sm dark:border-red-900/60 dark:bg-red-950/20">
        <p className="text-red-800 dark:text-red-300">{error.message}</p>
        <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
          <code className="rounded bg-red-100 px-1.5 py-0.5 text-red-700 dark:bg-red-900/40 dark:text-red-300">
            {error.error_code}
          </code>
          {error.trace_id && <TracePanel runId={error.trace_id} />}
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="ml-auto flex items-center gap-1 rounded-md bg-white px-2 py-1 font-medium text-zinc-700 ring-1 ring-zinc-200 hover:bg-zinc-50 dark:bg-zinc-900 dark:text-zinc-200 dark:ring-zinc-700"
            >
              <RotateCcw className="size-3.5" /> Retry
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
