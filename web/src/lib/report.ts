import { stripDownloadLinks, stripSourcesSection } from "./answer";
import type { Citation, ResearchReport } from "./types";

type Reference = { number: number; citation: Citation };

function sectionBody(text: string): string {
  return stripDownloadLinks(stripSourcesSection(text)).trim();
}

/** Number citations across the whole brief, first appearance first, one number per chunk. */
export function reportReferences(report: ResearchReport): Map<string, Reference> {
  const references = new Map<string, Reference>();
  for (const section of report.sections) {
    for (const citation of section.response?.citations ?? []) {
      const key = `${citation.document_id}:${citation.page_number}:${citation.chunk_id}`;
      if (!references.has(key)) references.set(key, { number: references.size + 1, citation });
    }
  }
  return references;
}

function referenceKey(citation: Citation): string {
  return `${citation.document_id}:${citation.page_number}:${citation.chunk_id}`;
}

/** A portable Markdown brief: every sentence is a validated claim with its reference numbers. */
export function reportToMarkdown(report: ResearchReport): string {
  const references = reportReferences(report);
  const lines = [
    `# ${report.title}`,
    "",
    `*Research topic:* ${report.topic}`,
    "",
    `${report.sections.filter((s) => s.status === "answered").length} of ${report.sections.length} sections answered · ${report.verified_claim_count} verified claims · ${references.size} sources`,
    "",
    "> Every factual sentence below passed citation validation against the Census 2011 source PDFs. Headings and questions are planning labels.",
    "",
  ];
  if (!report.in_scope) lines.push(report.reason ?? "This topic cannot be answered from the Census reports.", "");
  for (const section of report.sections) {
    lines.push(`## ${section.heading}`, "", `*${section.question}*`, "");
    const response = section.response;
    if (section.status === "answered" && response) {
      const byId = new Map(response.citations.map((citation) => [citation.citation_id, citation]));
      const claims = response.claims.filter((claim) => claim.text.trim());
      if (claims.length) {
        for (const claim of claims.slice(0, 12)) {
          const refs = claim.citation_ids
            .map((id) => byId.get(id))
            .filter((citation): citation is Citation => Boolean(citation))
            .map((citation) => references.get(referenceKey(citation))?.number)
            .filter((value): value is number => value !== undefined);
          lines.push(`- ${claim.text}${refs.length ? ` [${[...new Set(refs)].join(", ")}]` : ""}`);
        }
        if (claims.length > 12) lines.push(`- …and ${claims.length - 12} more verified rows in the table artifact.`);
      } else {
        lines.push(sectionBody(response.answer));
      }
      for (const artifact of response.artifacts) lines.push("", `*Artifact:* ${artifact.title} (${artifact.artifact_type}, generated in the isolated executor)`);
    } else {
      lines.push(
        section.status === "declined"
          ? `**Declined:** ${sectionBody(response?.answer ?? "")} The agent refuses rather than guesses when evidence cannot be verified.`
          : `**Not completed:** ${section.error?.message ?? "This section failed."}`,
      );
    }
    lines.push("");
  }
  if (references.size) {
    lines.push("## References", "");
    for (const { number, citation } of references.values()) {
      lines.push(`${number}. ${citation.document_title}, physical PDF page ${citation.page_number}.`);
    }
    lines.push("");
  }
  return lines.join("\n");
}

const escapeHtml = (value: string) =>
  value.replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]!);

/** A self-contained, print-ready HTML document of the brief (for "Save as PDF"). */
export function reportToPrintableHtml(report: ResearchReport): string {
  const markdown = reportToMarkdown(report);
  const body = markdown
    .split("\n")
    .map((line) => {
      const text = escapeHtml(line).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\*(.+?)\*/g, "<em>$1</em>");
      if (line.startsWith("# ")) return `<h1>${text.slice(2)}</h1>`;
      if (line.startsWith("## ")) return `<h2>${text.slice(3)}</h2>`;
      if (line.startsWith("> ")) return `<p class="note">${text.slice(5)}</p>`;
      if (line.startsWith("- ")) return `<li>${text.slice(2)}</li>`;
      if (/^\d+\. /.test(line)) return `<p class="ref">${text}</p>`;
      return line.trim() ? `<p>${text}</p>` : "";
    })
    .join("\n");
  return `<!doctype html><html><head><meta charset="utf-8"><title>${escapeHtml(report.title)}</title><style>
body{font:15px/1.6 ui-sans-serif,system-ui,sans-serif;color:#18181b;max-width:760px;margin:40px auto;padding:0 24px}
h1{font-size:26px;margin:0 0 8px}h2{font-size:18px;margin:28px 0 6px;border-top:1px solid #e4e4e7;padding-top:18px}
li{margin:4px 0 4px 18px}.note{color:#0f766e;background:#f0fdfa;padding:10px 12px;border-radius:8px}.ref{font-size:13px;color:#52525b;margin:2px 0}
@media print{body{margin:0 auto}}</style></head><body>${body}
<p class="ref" style="margin-top:32px">Generated by Census Insight Agent · citation-grounded research over Census 2011 India reports.</p></body></html>`;
}
