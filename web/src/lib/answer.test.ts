import { describe, expect, it } from "vitest";

import { answerIsClaims, stripSourcesSection } from "./answer";
import type { ChatResponse, Claim } from "./types";

const claim = (text: string) => ({ claim_id: text, text, citation_ids: ["c1"] }) as Claim;

describe("answer rendering", () => {
  const answer =
    "The literacy rate of Karnataka in 2011 was 75.36%.\n\nThe literacy rate of Karnataka in 2011 was 75.4%.\n\nSources:\n- [Karnataka, p. 50; “| a | b |”]";

  it("drops the plain-text sources list", () => {
    expect(stripSourcesSection(answer)).toBe(
      "The literacy rate of Karnataka in 2011 was 75.36%.\n\nThe literacy rate of Karnataka in 2011 was 75.4%.",
    );
    expect(stripSourcesSection("No sources here.")).toBe("No sources here.");
  });

  it("detects answers rendered from validated claims", () => {
    const response = {
      answer,
      refusal: false,
      claims: [
        claim("The literacy rate of Karnataka in 2011 was 75.36%."),
        claim("The literacy rate of Karnataka in 2011 was 75.4%."),
      ],
    } as ChatResponse;
    expect(answerIsClaims(response)).toBe(true);
    expect(answerIsClaims({ ...response, refusal: true })).toBe(false);
    expect(answerIsClaims({ ...response, answer: "A summary paragraph.\n\nSources:\n- x" })).toBe(false);
  });
});

import { cleanSectionPath, parseSnippetTable, snippetPreview, stripDownloadLinks } from "./answer";

// Verbatim from a live Karnataka citation (physical page 50).
const REAL_TABLE_SNIPPET = `| State / District Code | State / District | Literates 2011 |
|-----------------------|------------------|--------------------|
|                       |                  | Total              |
| 1                     | 2                | 3                  |
| -                     | <b>KARNATAKA</b> | <b>4,06,47,322</b> |`;

describe("snippet and artifact text", () => {
  it("previews the cited data row, not the header", () => {
    expect(snippetPreview(REAL_TABLE_SNIPPET)).toBe("- · KARNATAKA · 4,06,47,322");
    expect(snippetPreview("In continuation of the **trend**,\n literacy rose.")).toBe(
      "In continuation of the trend, literacy rose.",
    );
  });

  it("parses Markdown table quotes and keeps bold cells", () => {
    const rows = parseSnippetTable(REAL_TABLE_SNIPPET)!;
    expect(rows).toHaveLength(4);
    expect(rows.at(-1)!.map((cell) => [cell.text, cell.strong])).toEqual([
      ["-", false],
      ["KARNATAKA", true],
      ["4,06,47,322", true],
    ]);
    expect(parseSnippetTable("Plain prose evidence.")).toBeNull();
  });

  it("removes raw download paths and Markdown from section paths", () => {
    expect(
      stripDownloadLinks("Created Literacy chart. Download: /sessions/abc/artifacts/x/files/chart.png"),
    ).toBe("Created Literacy chart.");
    expect(cleanSectionPath(["**Census of India 2011**", "Statement 19"])).toEqual([
      "Census of India 2011",
      "Statement 19",
    ]);
  });
});

describe("breadcrumb-prefixed table quotes", () => {
  it("previews the data row when the quote starts with its breadcrumb", () => {
    // Shape of a live Karnataka Statement 19 citation.
    const snippet = `**Census of India 2011** > **Chapter-3** > Statement 19 > Literates and Literacy Rate by residence : 2011 (Persons)

| State / District Code | State / District | Literates 2011 |
|-----------------------|------------------|----------------|
| -                     | <b>KARNATAKA</b> | <b>4,06,47,322</b> |`;
    expect(snippetPreview(snippet)).toBe("- · KARNATAKA · 4,06,47,322");
  });
});
