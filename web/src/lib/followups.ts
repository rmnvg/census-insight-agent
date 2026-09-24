import { titleCase } from "./provenance";
import type { ChatResponse } from "./types";

export type FollowUp = { label: string; prompt: string };

function metricName(raw: string): string {
  return raw.replace(/_/g, " ").replace(/\s+/g, " ").trim().toLowerCase();
}

/**
 * Next questions derived from the answer's validated claims and the document library.
 * Deterministic: no model call, and every suggestion is a task the agent supports.
 */
export function suggestFollowUps(response: ChatResponse, libraryRegions: string[], limit = 4): FollowUp[] {
  if (response.refusal) {
    const region = libraryRegions[0];
    return region
      ? [
          { label: "Try a lookup", prompt: `What was the literacy rate of ${region} in 2011?` },
          { label: "See what's covered", prompt: `Summarize the key population findings for ${region}.` },
        ]
      : [];
  }
  const sourceClaims = response.claims.filter((claim) => !claim.derivation && claim.metric);
  const metric = sourceClaims[0]?.metric ? metricName(sourceClaims[0].metric) : null;
  if (!metric) return [];
  const answered = [...new Set(sourceClaims.map((claim) => titleCase(claim.region ?? "")).filter(Boolean))];
  const answeredKeys = new Set(answered.map((region) => region.toLowerCase()));
  const others = [...new Set(libraryRegions)].filter((region) => !answeredKeys.has(region.toLowerCase()));
  const primary = answered[0];
  const hasArtifact = response.artifacts.length > 0;
  const suggestions: FollowUp[] = [];

  if (primary && others[0] && answered.length === 1) {
    // A pronoun follow-up exercises the agent's validated conversation memory.
    suggestions.push({ label: `Compare with ${others[0]}`, prompt: `How does that compare with ${others[0]}?` });
  }
  if (!hasArtifact) {
    const chartRegions = answered.length >= 2 ? answered : primary && others[0] ? [primary, others[0]] : [];
    if (chartRegions.length >= 2) {
      suggestions.push({
        label: "Show as a chart",
        prompt: `Create a bar chart comparing the ${metric} of ${chartRegions.join(" and ")}.`,
      });
    }
  }
  if (primary && !sourceClaims.some((claim) => claim.residence_scope && claim.residence_scope !== "total")) {
    suggestions.push({
      label: "Rural vs urban",
      prompt: `Make a table of the ${metric} of ${primary} by rural and urban areas.`,
    });
  }
  if (primary && answered.length === 1) {
    suggestions.push({
      label: "Top district",
      prompt: `Which district of ${primary} had the highest ${metric}?`,
    });
  }
  if (!hasArtifact && response.citations.length) {
    suggestions.push({ label: "Show the evidence", prompt: "Which source pages support those values?" });
  }
  const seen = new Set<string>();
  return suggestions.filter((item) => !seen.has(item.prompt) && seen.add(item.prompt)).slice(0, limit);
}
