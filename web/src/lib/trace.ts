import type { Citation, Claim, RunTrace } from "./types";

// Same allowlists as the Streamlit UI (frontend/view_models.py): traces show observable
// decisions, never prompts, evidence text, or model reasoning.
const TRACE_EVENTS = new Set([
  "classification",
  "task_classification",
  "artifact_proposal_validation",
  "query_resolution",
  "skill_selected",
  "retrieval_completed",
  "evidence_assessment",
  "artifact_data_requirement",
  "artifact_dataset_validation",
  "code_policy_validation",
  "executor_submission",
  "executor_result",
  "artifact_execution",
  "artifact_repair_decision",
  "artifact_validation",
  "citation_validation",
  "run_completed",
  "run_failed",
]);

const DETAIL_FIELDS = new Set([
  "task_type",
  "resolved_query",
  "selected_skill",
  "candidate_count",
  "assessment_chunk_count",
  "assessment_characters",
  "selected_evidence_count",
  "sufficient",
  "valid",
  "status",
  "exit_code",
  "timed_out",
  "error_code",
  "attempt",
  "repair",
  "retry_count",
  "generated_filenames",
  "required_targets",
  "represented_targets",
]);

export type TraceView = {
  event: string;
  node: string | null;
  details: [string, string][];
  latencyMs: number | null;
  timestamp: string;
};

function safeValue(value: unknown): string | null {
  if (value === null || ["string", "number", "boolean"].includes(typeof value)) return String(value);
  if (Array.isArray(value) && value.every((item) => ["string", "number", "boolean"].includes(typeof item))) {
    return value.join(", ");
  }
  return null;
}

export function sanitizeTrace(trace: RunTrace): TraceView[] {
  return trace.events
    .filter((event) => TRACE_EVENTS.has(event.event))
    .map((event) => ({
      event: event.event,
      node: event.node,
      timestamp: event.timestamp,
      latencyMs: event.latency_ms,
      details: Object.entries(event.details).flatMap(([key, value]) => {
        const safe = DETAIL_FIELDS.has(key) ? safeValue(value) : null;
        return safe === null ? [] : [[key, safe] as [string, string]];
      }),
    }));
}

export function traceDurationMs(trace: RunTrace): number | null {
  if (trace.events.length < 2) return null;
  const first = Date.parse(trace.events[0].timestamp);
  const last = Date.parse(trace.events[trace.events.length - 1].timestamp);
  return Number.isFinite(first) && Number.isFinite(last) ? Math.max(0, last - first) : null;
}

export function orderedCitations(citations: Citation[]): Citation[] {
  const unique = new Map<string, Citation>();
  for (const citation of citations) if (!unique.has(citation.citation_id)) unique.set(citation.citation_id, citation);
  return [...unique.values()];
}

export function citationNumbers(citations: Citation[]): Map<string, number> {
  return new Map(orderedCitations(citations).map((citation, index) => [citation.citation_id, index + 1]));
}

const OPERATION_SYMBOL = { sum: "+", difference: "−", percentage_difference: "Δ%" } as const;

export function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-IN", { maximumFractionDigits: 4 }).format(value);
}

export function describeDerivation(claim: Claim): string | null {
  const derivation = claim.derivation;
  if (!derivation) return null;
  const symbol = OPERATION_SYMBOL[derivation.operation];
  const expression =
    derivation.operation === "percentage_difference"
      ? `(${formatNumber(derivation.operands[0])} → ${formatNumber(derivation.operands[1])})`
      : derivation.operands.map(formatNumber).join(` ${symbol} `);
  const unit = derivation.unit === "percentage_points" ? " pp" : derivation.unit === "percent" ? "%" : "";
  return `${expression} = ${formatNumber(derivation.result)}${unit}`;
}

export const REFUSAL_LABELS: Record<string, string> = {
  insufficient_evidence: "Declined: evidence not found in the reports",
  citation_validation_failed: "Declined: a claim failed citation validation",
  artifact_generation_failed: "Declined: artifact could not be verified",
  operational_error: "Stopped: operational error",
};
