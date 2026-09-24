import { describe, expect, it } from "vitest";

import { parseCsv } from "./csv";
import { buildChartModel, niceTicks, recordToCitation, titleCase, type SourceManifest } from "./provenance";

const record = (row_id: string, page: number) => ({
  source_record_id: `s-${row_id}`,
  row_id,
  field: "value",
  raw_value: "1",
  normalized_numeric_value: 1,
  unit: "females per 1000 males",
  metric: "sex ratio",
  region: "x",
  year: 2011,
  document_title: "Primary Census Abstract Data Highlights: Madhya Pradesh",
  document_id: "census-2011-madhya-pradesh-pca-highlights",
  page_number: page,
  chunk_id: `chunk-${row_id}`,
  exact_supporting_quote: "| 40 | Balaghat | 1021 |",
  source_checksum: "0".repeat(64),
});

describe("buildChartModel", () => {
  it("joins live ranking rows to their source pages and detects the ranking", () => {
    // Shape of a live ranking artifact's table.csv (Madhya Pradesh districts, sex ratio).
    const csv = parseCsv(
      "row_id,label,series,value\nrow-40,Balaghat,,1021.0\nrow-48,Alirajpur,,1011.0\nrow-37,Mandla,,1008.0\nrow-36,Dindori,,1002.0\nrow-1,Bhind,,837.0\n",
    );
    const manifest: SourceManifest = { dataset_title: "t", source_records: [record("row-40", 33)] };
    const model = buildChartModel(csv, manifest)!;
    expect(model.ranking).toBe(true);
    expect(model.rows[0]).toMatchObject({ label: "Balaghat", value: 1021, series: "" });
    expect(model.rows[0].source?.page_number).toBe(33);
    expect(model.rows[1].source).toBeNull();
    expect(model.unit).toBe("females per 1000 males");
  });

  it("keeps multi-series order and title-cases upper-case state labels", () => {
    const csv = parseCsv("row_id,label,series,value\nrow-1,ODISHA,Total,72.9\nrow-2,MADHYA PRADESH,Total,69.3\n");
    const model = buildChartModel(csv, null)!;
    expect(model.labels).toEqual(["Odisha", "Madhya Pradesh"]);
    expect(model.series).toEqual(["Total"]);
    expect(model.ranking).toBe(false);
  });

  it("refuses CSVs without the validated artifact columns", () => {
    expect(buildChartModel(parseCsv("Region,Value\nA,1\n"), null)).toBeNull();
    expect(buildChartModel(parseCsv("row_id,label,series,value\nr,A,,n/a\n"), null)).toBeNull();
  });
});

describe("helpers", () => {
  it("builds round ticks from zero that cover the maximum", () => {
    expect(niceTicks(75.36)).toEqual([0, 20, 40, 60, 80]);
    expect(niceTicks(1021)).toEqual([0, 250, 500, 750, 1000, 1250]);
    expect(niceTicks(0)).toEqual([0]);
  });

  it("title-cases only all-caps labels", () => {
    expect(titleCase("MADHYA PRADESH")).toBe("Madhya Pradesh");
    expect(titleCase("Dakshina Kannada")).toBe("Dakshina Kannada");
  });

  it("turns a source record into a page-viewer citation", () => {
    const citation = recordToCitation(record("row-40", 33));
    expect(citation).toMatchObject({ page_number: 33, chunk_id: "chunk-row-40", snippet: "| 40 | Balaghat | 1021 |" });
  });
});
