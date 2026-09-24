import type { Citation } from "./types";

// Mirrors backend `SourceRecord` in backend/app/execution/contracts.py.
export type SourceRecord = {
  source_record_id: string;
  row_id: string;
  field: string;
  raw_value: string;
  normalized_numeric_value: number | null;
  unit: string | null;
  metric: string | null;
  region: string | null;
  year: number | null;
  document_title: string | null;
  document_id: string;
  page_number: number;
  chunk_id: string;
  exact_supporting_quote: string;
  source_checksum: string;
};

export type SourceManifest = { dataset_title: string; source_records: SourceRecord[] };

export type ChartRow = {
  rowId: string;
  label: string;
  series: string;
  value: number;
  source: SourceRecord | null;
};

export type ChartModel = {
  rows: ChartRow[];
  series: string[];
  labels: string[];
  unit: string | null;
  max: number;
  /** Rows arrive sorted best-first for a ranking; the first row is the answer. */
  ranking: boolean;
};

const REQUIRED_COLUMNS = ["row_id", "label", "series", "value"];

/** Join the executor's validated CSV rows to their manifest provenance by `row_id`. */
export function buildChartModel(csv: string[][], manifest: SourceManifest | null): ChartModel | null {
  const [header, ...body] = csv;
  if (!header || !REQUIRED_COLUMNS.every((column) => header.includes(column))) return null;
  const index = Object.fromEntries(header.map((column, position) => [column, position]));
  const sources = new Map(
    (manifest?.source_records ?? [])
      .filter((record) => record.field === "value")
      .map((record) => [record.row_id, record]),
  );
  const rows: ChartRow[] = [];
  for (const cells of body) {
    const value = Number(cells[index.value]?.replace(/,/g, ""));
    if (!Number.isFinite(value)) return null;
    const rowId = cells[index.row_id];
    rows.push({
      rowId,
      label: titleCase(cells[index.label] ?? ""),
      series: cells[index.series]?.trim() || "",
      value,
      source: sources.get(rowId) ?? null,
    });
  }
  if (!rows.length) return null;
  const series = [...new Set(rows.map((row) => row.series))];
  const labels = [...new Set(rows.map((row) => row.label))];
  const values = rows.map((row) => row.value);
  const descending = values.every((value, i) => i === 0 || values[i - 1] >= value);
  const ascending = values.every((value, i) => i === 0 || values[i - 1] <= value);
  return {
    rows,
    series,
    labels,
    unit: rows.find((row) => row.source?.unit)?.source?.unit ?? null,
    max: Math.max(0, ...values),
    ranking: series.length === 1 && rows.length >= 5 && (descending || ascending),
  };
}

/** Census labels arrive upper-cased for state rows ("MADHYA PRADESH"). */
export function titleCase(label: string): string {
  const trimmed = label.trim();
  if (trimmed !== trimmed.toUpperCase() || !/[A-Z]/.test(trimmed)) return trimmed;
  return trimmed.toLowerCase().replace(/(^|[\s\-(/])([a-z])/g, (_, gap, letter) => gap + letter.toUpperCase());
}

/** Round-number axis ticks from zero, 3-6 of them. */
export function niceTicks(max: number): number[] {
  if (max <= 0) return [0];
  const rough = max / 5;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 2.5, 5, 10].map((factor) => factor * magnitude).find((value) => value >= rough)!;
  const ticks: number[] = [];
  for (let value = 0; value <= max + step * 0.001; value += step) ticks.push(Number(value.toFixed(10)));
  if (ticks[ticks.length - 1] < max) ticks.push(Number((ticks[ticks.length - 1] + step).toFixed(10)));
  return ticks;
}

/** A source record is a citation in all but name; the page viewer takes this shape. */
export function recordToCitation(record: SourceRecord): Citation {
  return {
    citation_id: `source-${record.row_id}`,
    document_id: record.document_id,
    document_title: record.document_title ?? record.document_id,
    page_number: record.page_number,
    snippet: record.exact_supporting_quote,
    chunk_id: record.chunk_id,
    section_path: [],
    evidence_span: { evidence_id: record.chunk_id, start_offset: 0, end_offset: record.exact_supporting_quote.length },
  };
}
