"use client";

import {
  AlertTriangle,
  Check,
  CircleSlash,
  Clock,
  FileDown,
  FileText,
  Loader2,
  Printer,
  ShieldCheck,
  Telescope,
} from "lucide-react";
import { useEffect, useState } from "react";

import { formatDuration } from "@/lib/dates";
import { reportReferences, reportToMarkdown, reportToPrintableHtml } from "@/lib/report";
import type { ResearchReport, ResearchSection } from "@/lib/types";

import { AssistantMessage } from "./assistant-message";
import type { Pending } from "./chat-provider";
import { LogoMark } from "./logo";
import { TracePanel } from "./trace-panel";

function Header({ title, subtitle, children }: { title: string; subtitle: string; children?: React.ReactNode }) {
  return (
    <div className="border-b border-zinc-200 bg-gradient-to-br from-teal-50/80 to-white px-5 py-4 dark:border-zinc-800 dark:from-teal-950/30 dark:to-zinc-900">
      <p className="mb-1 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-teal-700 dark:text-teal-400">
        <Telescope className="size-3.5" /> Deep research
      </p>
      <h3 className="text-lg font-semibold leading-snug tracking-tight">{title}</h3>
      <p className="mt-0.5 text-sm text-zinc-500">{subtitle}</p>
      {children}
    </div>
  );
}

/** Live view while a brief runs: the plan, with each section's current agent step. */
export function ResearchProgress({ pending }: { pending: Pending }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(timer);
  }, []);
  const research = pending.research;
  const done = research?.sections.filter((section) => section.state === "done").length ?? 0;
  return (
    <div className="flex gap-3">
      <LogoMark className="mt-0.5 size-7 shrink-0" />
      <div className="min-w-0 flex-1 overflow-hidden rounded-2xl border border-zinc-200 dark:border-zinc-800">
        <Header
          title={research?.title ?? "Planning the research brief…"}
          subtitle={
            research
              ? `${done} of ${research.sections.length} sections verified · ${formatDuration(now - pending.startedAt)}`
              : `Breaking “${pending.text}” into answerable questions · ${formatDuration(now - pending.startedAt)}`
          }
        />
        <ol className="divide-y divide-zinc-100 dark:divide-zinc-800">
          {!research && (
            <li className="flex items-center gap-2 px-5 py-4 text-sm text-zinc-500">
              <Loader2 className="size-4 animate-spin text-teal-600" /> Planning questions from the document library
            </li>
          )}
          {research?.sections.map((section, index) => {
            const last = section.steps.at(-1);
            const status = section.result?.status;
            return (
              <li key={index} className="animate-step-in flex items-start gap-3 px-5 py-3">
                <span className="mt-0.5">
                  {section.state === "queued" ? (
                    <Clock className="size-4 text-zinc-400" />
                  ) : section.state === "running" ? (
                    <Loader2 className="size-4 animate-spin text-teal-600" />
                  ) : status === "answered" ? (
                    <Check className="size-4 text-emerald-600" />
                  ) : (
                    <CircleSlash className="size-4 text-amber-600" />
                  )}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium">{section.heading}</p>
                  <p className="truncate text-xs text-zinc-500">{section.question}</p>
                  <p className="mt-0.5 text-[11px] text-zinc-400">
                    {section.state === "queued"
                      ? "Queued"
                      : section.state === "running"
                        ? (last?.label ?? "Starting…")
                        : status === "answered"
                          ? `Verified · ${section.result?.response?.claims.length ?? 0} claims`
                          : status === "declined"
                            ? "Declined: evidence not verifiable"
                            : "Not completed"}
                  </p>
                </div>
              </li>
            );
          })}
          {research && !research.inScope && (
            <li className="px-5 py-4 text-sm text-amber-800 dark:text-amber-300">{research.reason}</li>
          )}
        </ol>
      </div>
    </div>
  );
}

function download(filename: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function slug(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60) || "research-brief";
}

export function ResearchReportCard({ report }: { report: ResearchReport }) {
  const answered = report.sections.filter((section) => section.status === "answered").length;
  const sources = reportReferences(report).size;
  const printable = () => {
    const view = window.open("", "_blank");
    if (!view) return;
    view.document.write(reportToPrintableHtml(report));
    view.document.close();
    view.focus();
    view.print();
  };
  return (
    <div className="flex gap-3">
      <LogoMark className="mt-0.5 size-7 shrink-0" />
      <article className="min-w-0 flex-1 overflow-hidden rounded-2xl border border-zinc-200 dark:border-zinc-800">
        <Header title={report.title} subtitle={report.topic}>
          <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
            <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2.5 py-1 font-medium text-emerald-700 ring-1 ring-emerald-600/20 dark:bg-emerald-950/40 dark:text-emerald-300">
              <ShieldCheck className="size-3.5" /> {answered} of {report.sections.length} sections verified
            </span>
            <span className="rounded-full bg-white px-2.5 py-1 font-medium text-zinc-600 ring-1 ring-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:ring-zinc-700">
              {report.verified_claim_count} verified claims
            </span>
            <span className="rounded-full bg-white px-2.5 py-1 font-medium text-zinc-600 ring-1 ring-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:ring-zinc-700">
              {sources} source pages
            </span>
            <span className="rounded-full bg-white px-2.5 py-1 font-medium text-zinc-600 ring-1 ring-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:ring-zinc-700">
              {formatDuration(report.duration_seconds * 1000)}
            </span>
            <span className="ml-auto flex gap-1.5">
              <button
                type="button"
                onClick={() => download(`${slug(report.title)}.md`, reportToMarkdown(report), "text/markdown")}
                className="flex items-center gap-1 rounded-md bg-white px-2 py-1 font-medium text-zinc-700 ring-1 ring-zinc-200 hover:bg-zinc-50 dark:bg-zinc-800 dark:text-zinc-200 dark:ring-zinc-700"
              >
                <FileDown className="size-3.5" /> Markdown
              </button>
              <button
                type="button"
                onClick={printable}
                className="flex items-center gap-1 rounded-md bg-teal-700 px-2 py-1 font-medium text-white hover:bg-teal-800"
              >
                <Printer className="size-3.5" /> Export PDF
              </button>
            </span>
          </div>
          <p className="mt-3 text-[11px] leading-4 text-zinc-500">
            Every factual sentence below is a separately validated claim with its own citations. Headings and questions
            are planning labels only.
          </p>
        </Header>
        {!report.in_scope && (
          <p className="flex items-start gap-2 px-5 py-4 text-sm text-amber-800 dark:text-amber-300">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" />
            {report.reason ?? "This topic cannot be answered from the Census reports."}
          </p>
        )}
        <div className="divide-y divide-zinc-100 dark:divide-zinc-800">
          {report.sections.map((section, index) => (
            <Section key={index} index={index} section={section} />
          ))}
        </div>
      </article>
    </div>
  );
}

function Section({ index, section }: { index: number; section: ResearchSection }) {
  return (
    <section className="px-5 py-5">
      <h4 className="flex items-baseline gap-2 text-base font-semibold">
        <span className="text-sm tabular-nums text-teal-700 dark:text-teal-400">{index + 1}.</span>
        {section.heading}
      </h4>
      <p className="mb-3 mt-0.5 flex items-center gap-1.5 text-xs text-zinc-500">
        <FileText className="size-3" /> {section.question}
      </p>
      {section.status === "answered" && section.response ? (
        <AssistantMessage response={section.response} embedded />
      ) : section.status === "declined" ? (
        <div className="rounded-xl border border-amber-200 bg-amber-50/70 p-3 text-sm text-amber-900 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-200">
          <p className="mb-1 flex items-center gap-1.5 font-medium">
            <CircleSlash className="size-4" /> Declined rather than guessed
          </p>
          The agent could not verify this against the source pages, so it left the section out.
          {section.response && (
            <div className="mt-2">
              <TracePanel runId={section.response.trace_id} />
            </div>
          )}
        </div>
      ) : (
        <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-3 text-sm text-zinc-600 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300">
          {section.error?.message ?? "This section could not be completed."}
          {section.error?.error_code && (
            <code className="ml-2 rounded bg-zinc-200 px-1.5 py-0.5 text-xs dark:bg-zinc-800">{section.error.error_code}</code>
          )}
          {section.error?.trace_id && (
            <div className="mt-2">
              <TracePanel runId={section.error.trace_id} />
            </div>
          )}
        </div>
      )}
    </section>
  );
}
