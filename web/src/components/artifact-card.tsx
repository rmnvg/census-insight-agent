"use client";

import { BarChart3, Download, FileJson, Loader2, Table2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { artifactFileUrl } from "@/lib/api";
import { isNumeric, parseCsv } from "@/lib/csv";
import { buildChartModel, recordToCitation, type SourceManifest, type SourceRecord } from "@/lib/provenance";
import type { Artifact, Citation } from "@/lib/types";

import { PageViewer } from "./page-viewer";
import { ProvenanceChart } from "./provenance-chart";

type View = "interactive" | "image" | "table";
type Loaded = { rows: string[][]; manifest: SourceManifest | null } | "error" | null;

export function ArtifactCard({ artifact }: { artifact: Artifact }) {
  const chart = artifact.artifact_type === "chart";
  const url = (filename: string) => artifactFileUrl(artifact.session_id, artifact.artifact_id, filename);
  const dataFile = chart ? "plotted-data.csv" : "table.csv";
  const [loaded, setLoaded] = useState<Loaded>(null);
  const [chosen, setChosen] = useState<View | null>(null);
  const [viewing, setViewing] = useState<Citation | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    const get = (filename: string) =>
      fetch(url(filename), { signal: controller.signal }).then((response) => {
        if (!response.ok) throw new Error(String(response.status));
        return response.text();
      });
    Promise.all([get(dataFile), get("source-manifest.json").catch(() => null)])
      .then(([csv, manifest]) =>
        setLoaded({
          rows: parseCsv(csv.replace(/^﻿/, "")),
          manifest: manifest ? (JSON.parse(manifest) as SourceManifest) : null,
        }),
      )
      .catch((error) => {
        if (error?.name !== "AbortError") setLoaded("error");
      });
    return () => controller.abort();
    // The artifact identity fully determines these URLs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [artifact.artifact_id, dataFile]);

  const model = useMemo(
    () => (loaded && loaded !== "error" ? buildChartModel(loaded.rows, loaded.manifest) : null),
    [loaded],
  );
  const views: View[] = chart ? ["interactive", "image", "table"] : model ? ["interactive", "table"] : ["table"];
  const defaultView: View = model ? "interactive" : chart ? "image" : "table";
  const view = chosen ?? defaultView;
  const open = (record: SourceRecord) => setViewing(recordToCitation(record));
  const downloads = chart
    ? [
        ["chart.png", "PNG"],
        ["plotted-data.csv", "CSV"],
      ]
    : [
        ["table.csv", "CSV"],
        ["table.md", "Markdown"],
      ];
  const labels: Record<View, string> = { interactive: "Interactive", image: "Image", table: chart ? "Data" : "Table" };

  return (
    <section className="overflow-hidden rounded-xl border border-zinc-200 dark:border-zinc-800">
      <header className="flex flex-wrap items-center gap-2 border-b border-zinc-200 bg-zinc-50/80 px-3.5 py-2.5 dark:border-zinc-800 dark:bg-zinc-900/60">
        {chart ? (
          <BarChart3 className="size-4 text-teal-700 dark:text-teal-400" />
        ) : (
          <Table2 className="size-4 text-teal-700 dark:text-teal-400" />
        )}
        <h4 className="min-w-0 flex-1 truncate text-sm font-medium">{artifact.title}</h4>
        {views.length > 1 && (
          <div className="flex rounded-lg bg-zinc-200/70 p-0.5 text-xs dark:bg-zinc-800">
            {views.map((value) => (
              <button
                key={value}
                type="button"
                onClick={() => setChosen(value)}
                className={`rounded-md px-2.5 py-1 font-medium transition ${
                  view === value ? "bg-white shadow-sm dark:bg-zinc-700" : "text-zinc-500"
                }`}
              >
                {labels[value]}
              </button>
            ))}
          </div>
        )}
      </header>
      <div className="p-3.5">
        {view === "image" ? (
          // eslint-disable-next-line @next/next/no-img-element -- validated PNG served by the API
          <img src={url("chart.png")} alt={artifact.title} className="mx-auto max-h-[480px] w-auto rounded-md bg-white" />
        ) : loaded === "error" ? (
          <p className="text-sm text-zinc-500">The data file is unavailable.</p>
        ) : !loaded ? (
          <Loader2 className="mx-auto size-5 animate-spin text-zinc-400" />
        ) : view === "interactive" && model ? (
          <>
            <ProvenanceChart model={model} title={artifact.title} onOpenSource={open} />
            <p className="mt-3 text-[11px] text-zinc-500">
              Every bar is a verified source cell. Hover for its page, click to open the original PDF.
            </p>
          </>
        ) : (
          <DataTable rows={loaded.rows} manifest={loaded.manifest} onOpenSource={open} />
        )}
      </div>
      <footer className="flex flex-wrap items-center gap-1.5 border-t border-zinc-200 px-3.5 py-2 text-xs dark:border-zinc-800">
        <span className="mr-1 text-zinc-500">Generated in the network-isolated executor from verified cells</span>
        <span className="ml-auto" />
        {downloads.map(([filename, label]) => (
          <a
            key={filename}
            href={url(filename)}
            download={filename}
            className="flex items-center gap-1 rounded-md px-2 py-1 font-medium text-zinc-600 ring-1 ring-zinc-200 hover:bg-zinc-50 dark:text-zinc-300 dark:ring-zinc-700 dark:hover:bg-zinc-800"
          >
            <Download className="size-3" /> {label}
          </a>
        ))}
        <a
          href={url("source-manifest.json")}
          download="source-manifest.json"
          title="Every plotted value mapped back to its PDF page, chunk, and checksum"
          className="flex items-center gap-1 rounded-md px-2 py-1 font-medium text-teal-700 ring-1 ring-teal-600/30 hover:bg-teal-50 dark:text-teal-300 dark:hover:bg-teal-950/40"
        >
          <FileJson className="size-3" /> Provenance
        </a>
      </footer>
      {viewing && <PageViewer key={viewing.citation_id} citation={viewing} onClose={() => setViewing(null)} />}
    </section>
  );
}

const HIDDEN_COLUMNS = new Set(["row_id"]);

function DataTable({
  rows,
  manifest,
  onOpenSource,
}: {
  rows: string[][];
  manifest: SourceManifest | null;
  onOpenSource: (record: SourceRecord) => void;
}) {
  const [header, ...body] = rows;
  if (!header) return <p className="text-sm text-zinc-500">The data file is empty.</p>;
  const rowIdIndex = header.indexOf("row_id");
  const sources = new Map(
    (manifest?.source_records ?? []).filter((record) => record.field === "value").map((record) => [record.row_id, record]),
  );
  const visible = header.map((name, index) => ({ name, index })).filter(({ name }) => !HIDDEN_COLUMNS.has(name));
  const showSource = rowIdIndex !== -1 && sources.size > 0;
  return (
    <div className="max-h-96 overflow-auto rounded-lg ring-1 ring-zinc-200 dark:ring-zinc-800">
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-zinc-50 dark:bg-zinc-900">
          <tr>
            {visible.map(({ name }) => (
              <th key={name} className="whitespace-nowrap px-3 py-2 text-left text-xs font-semibold text-zinc-600 dark:text-zinc-300">
                {name}
              </th>
            ))}
            {showSource && <th className="px-3 py-2 text-right text-xs font-semibold text-zinc-600 dark:text-zinc-300">Source</th>}
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800">
          {body.map((row, rowIndex) => {
            const source = showSource ? sources.get(row[rowIdIndex]) : undefined;
            return (
              <tr key={rowIndex} className="hover:bg-zinc-50/70 dark:hover:bg-zinc-900/50">
                {visible.map(({ index }) => (
                  <td
                    key={index}
                    className={`px-3 py-1.5 ${isNumeric(row[index] ?? "") ? "text-right font-mono text-[13px] tabular-nums" : ""}`}
                  >
                    {row[index]}
                  </td>
                ))}
                {showSource && (
                  <td className="px-3 py-1.5 text-right">
                    {source ? (
                      <button
                        type="button"
                        onClick={() => onOpenSource(source)}
                        className="text-xs font-medium text-teal-700 hover:underline dark:text-teal-400"
                      >
                        p. {source.page_number}
                      </button>
                    ) : (
                      <span className="text-xs text-zinc-400">—</span>
                    )}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
