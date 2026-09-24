import { describe, expect, it } from "vitest";

import { reportReferences, reportToMarkdown, reportToPrintableHtml } from "./report";
import type { ChatResponse, Citation, ResearchReport } from "./types";

const citation = (id: string, page: number, chunk: string): Citation =>
  ({ citation_id: id, document_id: "doc", document_title: "Census 2011 Karnataka", page_number: page, chunk_id: chunk }) as Citation;

const answered = (text: string, cite: Citation): ChatResponse =>
  ({
    answer: `${text}\n\nSources:\n- raw`,
    claims: [{ claim_id: "c", text, citation_ids: [cite.citation_id] }],
    citations: [cite],
    artifacts: [],
    limitations: [],
    refusal: false,
    trace_id: "t",
  }) as unknown as ChatResponse;

const report: ResearchReport = {
  topic: "Gender balance",
  title: "Gender balance in <Karnataka>",
  in_scope: true,
  reason: null,
  verified_claim_count: 2,
  citation_count: 1,
  duration_seconds: 30,
  sections: [
    { heading: "Sex ratio", question: "What was the sex ratio of Karnataka?", status: "answered", session_id: "s1", error: null, response: answered("Karnataka's sex ratio was 973.", citation("citation-1", 30, "a")) },
    { heading: "Again", question: "Same source?", status: "answered", session_id: "s2", error: null, response: answered("Rural was 979.", citation("citation-1", 30, "a")) },
    { heading: "GDP", question: "GDP?", status: "declined", session_id: "s3", error: null, response: { ...answered("x", citation("c", 1, "z")), answer: "I can't verify that.", claims: [], citations: [], refusal: true } },
  ],
};

describe("research report export", () => {
  it("numbers one shared source once across sections", () => {
    expect([...reportReferences(report).values()].map((item) => item.number)).toEqual([1]);
  });

  it("renders claims with references and keeps declined sections honest", () => {
    const markdown = reportToMarkdown(report);
    expect(markdown).toContain("- Karnataka's sex ratio was 973. [1]");
    expect(markdown).toContain("- Rural was 979. [1]");
    expect(markdown).toContain("**Declined:** I can't verify that.");
    expect(markdown).toContain("1. Census 2011 Karnataka, physical PDF page 30.");
    expect(markdown).not.toContain("Sources:");
  });

  it("escapes HTML in the printable version", () => {
    const html = reportToPrintableHtml(report);
    expect(html).toContain("Gender balance in &lt;Karnataka&gt;");
    expect(html).not.toContain("<Karnataka>");
  });
});
