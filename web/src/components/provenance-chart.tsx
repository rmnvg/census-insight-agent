"use client";

import { ExternalLink } from "lucide-react";
import { useRef, useState } from "react";

import { niceTicks, type ChartModel, type ChartRow, type SourceRecord } from "@/lib/provenance";

const SERIES_COLORS = ["var(--viz-series-1)", "var(--viz-series-2)", "var(--viz-series-3)"];
const format = (value: number) => new Intl.NumberFormat("en-IN", { maximumFractionDigits: 2 }).format(value);

function unitSuffix(unit: string | null): string {
  if (!unit) return "";
  return unit === "%" || unit.toLowerCase() === "percent" ? "%" : ` ${unit}`;
}

type Hover = { row: ChartRow; x: number; y: number };

/**
 * Horizontal bars drawn from the executor's validated CSV, where every bar is a link to the
 * physical PDF page its value was verified against (via the artifact's source manifest).
 */
export function ProvenanceChart({
  model,
  title,
  onOpenSource,
}: {
  model: ChartModel;
  title: string;
  onOpenSource: (record: SourceRecord) => void;
}) {
  const [hover, setHover] = useState<Hover | null>(null);
  const frame = useRef<HTMLDivElement>(null);
  const ticks = niceTicks(model.max);
  const axisMax = ticks[ticks.length - 1] || 1;
  const multiSeries = model.series.length > 1;
  const colorFor = (series: string) =>
    multiSeries ? SERIES_COLORS[model.series.indexOf(series) % SERIES_COLORS.length] : "var(--viz-accent)";
  const direction = model.ranking && model.rows[0].value < model.rows[model.rows.length - 1].value ? "Lowest" : "Highest";
  const suffix = unitSuffix(model.unit);

  const show = (row: ChartRow, target: HTMLElement) => {
    const box = frame.current?.getBoundingClientRect();
    const bar = target.getBoundingClientRect();
    if (!box) return;
    const x = Math.max(0, Math.min(bar.right - box.left + 10, box.width - 232));
    // Keep the tooltip (about 90px tall, centred on the bar) inside the chart frame.
    const y = Math.max(46, Math.min(bar.top - box.top + bar.height / 2, box.height - 46));
    setHover({ row, x, y });
  };

  return (
    <div ref={frame} className="viz-root relative" role="group" aria-label={title} onMouseLeave={() => setHover(null)}>
      {multiSeries && (
        <div className="mb-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-zinc-600 dark:text-zinc-300">
          {model.series.map((series) => (
            <span key={series} className="flex items-center gap-1.5">
              <span className="size-2.5 rounded-[2px]" style={{ background: colorFor(series) }} />
              {series}
            </span>
          ))}
        </div>
      )}
      <div className={model.labels.length > 14 ? "max-h-[440px] overflow-y-auto pr-1" : ""}>
        <div className="grid grid-cols-[minmax(0,max-content)_minmax(0,1fr)] items-center gap-x-3 gap-y-2">
          {model.labels.map((label, labelIndex) => {
            const rows = model.rows.filter((row) => row.label === label);
            const winner = model.ranking && labelIndex === 0;
            return (
              <div key={label} className="contents">
                <div className="flex max-w-[180px] items-center justify-end gap-1.5 text-right text-[13px] text-zinc-700 dark:text-zinc-300">
                  {winner && (
                    <span className="shrink-0 rounded bg-teal-50 px-1 text-[10px] font-semibold uppercase tracking-wide text-teal-700 dark:bg-teal-950/60 dark:text-teal-300">
                      {direction}
                    </span>
                  )}
                  <span className="truncate" title={label}>
                    {label}
                  </span>
                </div>
                <div className="relative flex flex-col gap-[2px] py-0.5">
                  {ticks.map((tick) => (
                    <span
                      key={tick}
                      aria-hidden
                      className="absolute inset-y-0 w-px"
                      style={{ left: `${(tick / axisMax) * 100}%`, background: "var(--viz-grid)" }}
                    />
                  ))}
                  {rows.map((row) => {
                    const width = Math.max(0.5, (row.value / axisMax) * 100);
                    const muted = model.ranking && !winner;
                    const inside = width > 82;
                    const text = `${format(row.value)}${suffix === "%" ? "%" : ""}`;
                    return (
                      <div key={row.rowId} className="relative flex items-center">
                        <button
                          type="button"
                          aria-label={`${label}${row.series ? `, ${row.series}` : ""}: ${format(row.value)}${suffix}${
                            row.source ? `. Source page ${row.source.page_number}` : ""
                          }`}
                          onMouseEnter={(event) => show(row, event.currentTarget)}
                          onFocus={(event) => show(row, event.currentTarget)}
                          onBlur={() => setHover(null)}
                          onClick={() => row.source && onOpenSource(row.source)}
                          className={`relative h-[18px] rounded-r-[4px] transition-[filter,box-shadow] hover:brightness-110 focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-teal-600 ${
                            row.source ? "cursor-pointer" : "cursor-default"
                          } ${hover?.row.rowId === row.rowId ? "ring-2 ring-zinc-900/15 dark:ring-white/25" : ""}`}
                          style={{ width: `${width}%`, background: colorFor(row.series), opacity: muted ? 0.38 : 1 }}
                        >
                          {inside && (
                            <span className="absolute inset-y-0 right-1.5 flex items-center text-[11px] font-medium tabular-nums text-white">
                              {text}
                            </span>
                          )}
                        </button>
                        {!inside && (
                          <span className="ml-1.5 text-[11px] tabular-nums text-zinc-600 dark:text-zinc-400">{text}</span>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <div className="mt-1 grid grid-cols-[minmax(0,max-content)_minmax(0,1fr)] gap-x-3">
        <span className="invisible max-w-[180px] text-[13px]">{model.labels[0]}</span>
        <div className="relative h-4 text-[10px] tabular-nums text-zinc-400">
          {ticks.map((tick) => (
            <span key={tick} className="absolute -translate-x-1/2" style={{ left: `${(tick / axisMax) * 100}%` }}>
              {format(tick)}
            </span>
          ))}
        </div>
      </div>
      {model.unit && <p className="mt-1 text-right text-[11px] text-zinc-400">{model.unit}</p>}
      {hover && (
        <div
          role="tooltip"
          className="pointer-events-none absolute z-10 w-56 -translate-y-1/2 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-xs shadow-lg dark:border-zinc-700 dark:bg-zinc-900"
          style={{ left: hover.x, top: hover.y }}
        >
          <p className="text-sm font-semibold tabular-nums text-zinc-900 dark:text-white">
            {format(hover.row.value)}
            <span className="font-normal text-zinc-500">{suffix}</span>
          </p>
          <p className="text-zinc-600 dark:text-zinc-300">
            {hover.row.label}
            {hover.row.series ? ` · ${hover.row.series}` : ""}
          </p>
          {hover.row.source ? (
            <p className="mt-1.5 flex items-center gap-1 border-t border-zinc-100 pt-1.5 text-teal-700 dark:border-zinc-800 dark:text-teal-400">
              <ExternalLink className="size-3" />
              Verified on page {hover.row.source.page_number} · click to open
            </p>
          ) : (
            <p className="mt-1.5 border-t border-zinc-100 pt-1.5 text-zinc-500 dark:border-zinc-800">Source row unavailable</p>
          )}
        </div>
      )}
    </div>
  );
}
