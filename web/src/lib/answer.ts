import type { ChatResponse } from "./types";

const SOURCES_MARKER = "\n\nSources:\n";

/** The backend appends a plain-text "Sources:" list for text clients; the UI renders
 * structured citations instead. */
export function stripSourcesSection(answer: string): string {
  const index = answer.lastIndexOf(SOURCES_MARKER);
  return (index === -1 ? answer : answer.slice(0, index)).trim();
}

const normalize = (value: string) => value.replace(/\s+/g, " ").trim();

/** True when the answer body is exactly the validated claims (the normal cited-answer case),
 * so the claims can be rendered as the answer itself with inline citations. */
export function answerIsClaims(response: ChatResponse): boolean {
  if (response.refusal || !response.claims.length) return false;
  const body = normalize(stripSourcesSection(response.answer));
  return body === normalize(response.claims.map((claim) => claim.text).join("\n\n"));
}

/** Artifact answers embed raw API download paths for text clients; the UI has buttons instead. */
export function stripDownloadLinks(body: string): string {
  return body.replace(/\s*Download:\s*\/sessions\/\S+/g, "").trim();
}

export function cleanSectionPath(parts: string[]): string[] {
  return parts.map((part) => part.replace(/\*\*|__|`/g, "").trim()).filter(Boolean);
}

export type SnippetCell = { text: string; strong: boolean };

const SEPARATOR_ROW = /^\|?[\s\-:|]+\|?$/;

/** Parse a Markdown pipe-table quote into rows; null when the snippet is prose. */
export function parseSnippetTable(snippet: string): SnippetCell[][] | null {
  const lines = snippet
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  if (!lines.length || !lines.every((line) => line.startsWith("|"))) return null;
  const rows = lines
    .filter((line) => !SEPARATOR_ROW.test(line))
    .map((line) =>
      line
        .replace(/^\|/, "")
        .replace(/\|$/, "")
        .split("|")
        .map((raw) => ({
          strong: /<b>|\*\*/i.test(raw),
          text: raw
            .replace(/<br\s*\/?>/gi, " ")
            .replace(/<\/?[A-Za-z][^>]*>/g, "")
            .replace(/\*\*/g, "")
            .replace(/\s+/g, " ")
            .trim(),
        })),
    )
    .filter((row) => row.some((cell) => cell.text));
  return rows.length ? rows : null;
}

/** One-line preview: for tables, the last data-bearing row (where cited values sit). */
export function snippetPreview(snippet: string): string {
  const table = parseSnippetTable(snippet);
  if (table) {
    const dataRow = [...table].reverse().find((row) => row.some((cell) => /\d/.test(cell.text)));
    const row = dataRow ?? table[table.length - 1];
    return row
      .map((cell) => cell.text)
      .filter(Boolean)
      .join(" · ");
  }
  return snippet.replace(/<\/?[A-Za-z][^>]*>/g, "").replace(/\*\*/g, "").replace(/\s+/g, " ").trim();
}
