"use client";

import { BarChart3, Download, FileJson, Loader2, Table2 } from "lucide-react";
import { useEffect, useState } from "react";

import { artifactFileUrl } from "@/lib/api";
import { isNumeric, parseCsv } from "@/lib/csv";
import type { Artifact } from "@/lib/types";

export function ArtifactCard({ artifact }: { artifact: Artifact }) {
  const chart = artifact.artifact_type === "chart";
  const [tab, setTab] = useState<"visual" | "data">("visual");
  const url = (filename: string) => artifactFileUrl(artifact.session_id, artifact.artifact_id, filename);
  const dataFile = chart ? "plotted-data.csv" : "table.csv";
  const downloads = chart
    ? [
        ["chart.png", "PNG"],
        ["plotted-data.csv", "CSV"],
      ]
    : [
        ["table.csv", "CSV"],
        ["table.md", "Markdown"],
      ];

  return (
    <section className="overflow-hidden rounded-xl border border-zinc-200 dark:border-zinc-800">
      <header className="flex flex-wrap items-center gap-2 border-b border-zinc-200 bg-zinc-50/80 px-3.5 py-2.5 dark:border-zinc-800 dark:bg-zinc-900/60">
        {chart ? <BarChart3 className="size-4 text-teal-700 dark:text-teal-400" /> : <Table2 className="size-4 text-teal-700 dark:text-teal-400" />}
        <h4 className="min-w-0 flex-1 truncate text-sm font-medium">{artifact.title}</h4>
        {chart && (
          <div className="flex rounded-lg bg-zinc-200/70 p-0.5 text-xs dark:bg-zinc-800">
            {(["visual", "data"] as const).map((value) => (
              <button
                key={value}
                type="button"
                onClick={() => setTab(value)}
                className={`rounded-md px-2.5 py-1 font-medium transition ${
                  tab === value ? "bg-white shadow-sm dark:bg-zinc-700" : "text-zinc-500"
                }`}
              >
                {value === "visual" ? "Chart" : "Data"}
              </button>
            ))}
          </div>
        )}
      </header>
      <div className="p-3.5">
        {chart && tab === "visual" ? (
          // eslint-disable-next-line @next/next/no-img-element -- validated PNG served by the API
          <img src={url("chart.png")} alt={artifact.title} className="mx-auto max-h-[480px] w-auto rounded-md bg-white" />
        ) : (
          <CsvTable src={url(dataFile)} />
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
    </section>
  );
}

function CsvTable({ src }: { src: string }) {
  const [rows, setRows] = useState<string[][] | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    fetch(src, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(String(response.status));
        return response.text();
      })
      .then((text) => setRows(parseCsv(text.replace(/^﻿/, ""))))
      .catch((error) => {
        if (error?.name !== "AbortError") setFailed(true);
      });
    return () => controller.abort();
  }, [src]);

  if (failed) return <p className="text-sm text-zinc-500">The data file is unavailable.</p>;
  if (!rows) return <Loader2 className="mx-auto size-5 animate-spin text-zinc-400" />;
  const [header, ...body] = rows;
  return (
    <div className="max-h-96 overflow-auto rounded-lg ring-1 ring-zinc-200 dark:ring-zinc-800">
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-zinc-50 dark:bg-zinc-900">
          <tr>
            {header.map((cell, index) => (
              <th key={index} className="whitespace-nowrap px-3 py-2 text-left text-xs font-semibold text-zinc-600 dark:text-zinc-300">
                {cell}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800">
          {body.map((row, rowIndex) => (
            <tr key={rowIndex} className="hover:bg-zinc-50/70 dark:hover:bg-zinc-900/50">
              {row.map((cell, index) => (
                <td
                  key={index}
                  className={`px-3 py-1.5 ${isNumeric(cell) ? "text-right font-mono tabular-nums text-[13px]" : ""}`}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
