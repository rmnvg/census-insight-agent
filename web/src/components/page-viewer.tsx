"use client";

import { ChevronLeft, ChevronRight, Highlighter, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";

import { cleanSectionPath, parseSnippetTable } from "@/lib/answer";
import { pageImageUrl } from "@/lib/api";
import type { Citation } from "@/lib/types";

import { Dialog } from "./dialog";

/** Shows the authoritative PDF page behind a citation, with the quoted text highlighted. */
export function PageViewer({ citation, onClose }: { citation: Citation; onClose: () => void }) {
  const [page, setPage] = useState(citation.page_number);
  const [pageCount, setPageCount] = useState<number | null>(null);
  const [loaded, setLoaded] = useState<{
    page: number;
    src: string | null;
    failed: boolean;
    highlight: string | null;
  } | null>(null);
  const status = loaded?.page !== page ? "loading" : loaded.failed ? "error" : "ready";
  const src = loaded?.src ?? null;

  useEffect(() => {
    let objectUrl: string | null = null;
    const controller = new AbortController();
    const highlight = page === citation.page_number ? citation.snippet : undefined;
    fetch(pageImageUrl(citation.document_id, page, highlight), { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(String(response.status));
        const count = Number(response.headers.get("x-page-count"));
        if (count > 0) setPageCount(count);
        const highlightStatus = response.headers.get("x-highlight");
        objectUrl = URL.createObjectURL(await response.blob());
        setLoaded({ page, src: objectUrl, failed: false, highlight: highlightStatus });
      })
      .catch((error) => {
        if (error?.name !== "AbortError") setLoaded({ page, src: null, failed: true, highlight: null });
      });
    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [citation, page]);

  const cited = page === citation.page_number;
  return (
    <Dialog
      open
      wide
      onClose={onClose}
      title={citation.document_title}
      subtitle={
        <span className="flex flex-wrap items-center gap-x-2">
          <span>Physical PDF page {citation.page_number}</span>
          {citation.section_path.length > 0 && <span>· {cleanSectionPath(citation.section_path).join(" › ")}</span>}
        </span>
      }
    >
      <div className="grid gap-0 md:grid-cols-[minmax(0,1fr)_280px]">
        <div className="relative flex min-h-[60vh] items-start justify-center bg-zinc-100 p-4 dark:bg-zinc-950">
          {status === "loading" && (
            <Loader2 className="absolute left-1/2 top-1/3 size-6 -translate-x-1/2 animate-spin text-zinc-400" />
          )}
          {status === "error" ? (
            <p className="self-center text-sm text-zinc-500">The page image is unavailable.</p>
          ) : (
            src && (
              // eslint-disable-next-line @next/next/no-img-element -- blob URL from the API proxy
              <img
                src={src}
                alt={`${citation.document_title}, page ${page}`}
                className={`max-h-[75vh] w-auto rounded-md bg-white shadow-lg transition-opacity ${
                  status === "loading" ? "opacity-40" : "opacity-100"
                }`}
              />
            )
          )}
        </div>
        <aside className="space-y-4 border-t border-zinc-200 p-4 text-sm md:border-l md:border-t-0 dark:border-zinc-800">
          <div className="flex items-center justify-between">
            <button
              type="button"
              disabled={page <= 1}
              onClick={() => setPage((value) => value - 1)}
              className="rounded-lg border border-zinc-200 p-1.5 disabled:opacity-40 dark:border-zinc-700"
              aria-label="Previous page"
            >
              <ChevronLeft className="size-4" />
            </button>
            <span className="tabular-nums text-zinc-600 dark:text-zinc-300">
              Page {page}
              {pageCount ? ` of ${pageCount}` : ""}
            </span>
            <button
              type="button"
              disabled={pageCount !== null && page >= pageCount}
              onClick={() => setPage((value) => value + 1)}
              className="rounded-lg border border-zinc-200 p-1.5 disabled:opacity-40 dark:border-zinc-700"
              aria-label="Next page"
            >
              <ChevronRight className="size-4" />
            </button>
          </div>
          {!cited && (
            <button
              type="button"
              onClick={() => setPage(citation.page_number)}
              className="w-full rounded-lg bg-teal-700 px-3 py-1.5 text-xs font-medium text-white hover:bg-teal-800"
            >
              Back to cited page {citation.page_number}
            </button>
          )}
          <div>
            <p className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-zinc-500">
              <Highlighter className="size-3.5" /> Quoted evidence
            </p>
            <QuotedEvidence snippet={citation.snippet} />
            {cited && status === "ready" && <HighlightNote status={loaded?.highlight ?? null} />}
          </div>
          <dl className="space-y-1 text-[11px] text-zinc-500">
            <div className="flex gap-1">
              <dt className="shrink-0">Document:</dt>
              <dd className="truncate font-mono">{citation.document_id}</dd>
            </div>
            <div className="flex gap-1">
              <dt className="shrink-0">Chunk:</dt>
              <dd className="truncate font-mono">{citation.chunk_id}</dd>
            </div>
          </dl>
        </aside>
      </div>
    </Dialog>
  );
}

const HIGHLIGHT_NOTES: Record<string, string> = {
  matched: "The quoted text is highlighted on the page.",
  unmatched:
    "The quote couldn’t be located token-for-token on this page (tables often extract differently), so nothing is highlighted. Compare it with the page directly.",
  no_text_layer:
    "This PDF draws its text as vector graphics with no text layer, so the quote can’t be located automatically. Compare the quoted row with the page image.",
};

function HighlightNote({ status }: { status: string | null }) {
  const note = status ? HIGHLIGHT_NOTES[status] : undefined;
  if (!note) return null;
  return (
    <p className={`mt-2 text-[11px] leading-4 ${status === "matched" ? "text-emerald-700 dark:text-emerald-400" : "text-zinc-500"}`}>
      {note} The citation itself was verified in code against the indexed page text and its checksum.
    </p>
  );
}

function QuotedEvidence({ snippet }: { snippet: string }) {
  const table = parseSnippetTable(snippet);
  if (!table) {
    return (
      <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded-lg bg-amber-50 p-3 font-mono text-xs leading-relaxed text-zinc-800 dark:bg-amber-950/30 dark:text-zinc-200">
        {snippet}
      </pre>
    );
  }
  return (
    <div className="max-h-72 overflow-auto rounded-lg bg-amber-50 p-1 dark:bg-amber-950/30">
      <table className="w-full border-collapse text-[11px] leading-snug text-zinc-800 dark:text-zinc-200">
        <tbody>
          {table.map((row, rowIndex) => (
            <tr key={rowIndex} className="border-b border-amber-200/60 last:border-0 dark:border-amber-900/40">
              {row.map((cell, cellIndex) => (
                <td
                  key={cellIndex}
                  className={`px-1.5 py-1 align-top ${cell.strong ? "font-semibold text-zinc-950 dark:text-white" : ""} ${
                    /^[\d,.-]+$/.test(cell.text) ? "text-right font-mono tabular-nums" : ""
                  }`}
                >
                  {cell.text}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
