import { describe, expect, it } from "vitest";

import { citationNumbers, describeDerivation, sanitizeTrace } from "./trace";
import type { Citation, Claim, RunTrace } from "./types";

describe("trace sanitizing", () => {
  it("keeps only allowlisted events and scalar detail fields", () => {
    const trace = {
      session_id: "s",
      run_id: "r",
      events: [
        { timestamp: "t", event: "classification", node: "classify_task", details: { task_type: "lookup", prompt: "secret" }, latency_ms: 5 },
        { timestamp: "t", event: "structured_draft", node: "synthesize", details: { claims: [] }, latency_ms: 1 },
        { timestamp: "t", event: "retrieval_completed", node: "call_tools", details: { candidate_count: 12, nested: { a: 1 } }, latency_ms: null },
      ],
    } as unknown as RunTrace;
    expect(sanitizeTrace(trace).map((event) => [event.event, event.details])).toEqual([
      ["classification", [["task_type", "lookup"]]],
      ["retrieval_completed", [["candidate_count", "12"]]],
    ]);
  });
});

describe("citations and derivations", () => {
  it("numbers citations by first appearance, ignoring duplicates", () => {
    const citation = (id: string) => ({ citation_id: id }) as Citation;
    expect([...citationNumbers([citation("b"), citation("a"), citation("b")])]).toEqual([
      ["b", 1],
      ["a", 2],
    ]);
  });

  it("renders app-computed arithmetic", () => {
    const claim = {
      derivation: { operation: "difference", operands: [75.36, 72.87], result: 2.49, unit: "percentage_points", input_claim_ids: ["a", "b"] },
    } as unknown as Claim;
    expect(describeDerivation(claim)).toBe("75.36 − 72.87 = 2.49 pp");
    expect(describeDerivation({ derivation: null } as Claim)).toBeNull();
  });
});
