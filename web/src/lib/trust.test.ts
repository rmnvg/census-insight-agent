import { describe, expect, it } from "vitest";

import { failureReasons, verdictOf } from "./trust";
import type { TrustCaseResult } from "./types";

const base: TrustCaseResult = {
  case_id: "ka-literacy",
  category: "lookup",
  question: "q",
  expected: "e",
  passed: false,
  outcome: "answered",
  answer_excerpt: "",
  observed_values: [],
  missing: [],
  claims_checked: 0,
  ungrounded_claims: [],
  arithmetic_errors: [],
  uncited_claims: 0,
  citation_count: 0,
  artifact_types: [],
  latency_seconds: 1,
  attempts: 1,
  error_code: null,
  trace_id: null,
  source_note: "",
};

describe("trust verdicts", () => {
  it("uses the scorer's verdict when present", () => {
    expect(verdictOf({ ...base, verdict: "wrong_answer" })).toBe("wrong_answer");
  });

  it.each([
    [{ outcome: "error" as const }, "error"],
    [{ passed: true }, "passed"],
    [{ outcome: "refused" as const }, "false_refusal"],
    [{ ungrounded_claims: ["x"] }, "unsupported"],
    [{ category: "refusal" as const }, "wrong_answer"],
    [{ missing: ["975"] }, "incomplete"],
  ])("infers %o for older scorecards", (overrides, verdict) => {
    expect(verdictOf({ ...base, ...overrides })).toBe(verdict);
  });

  it("prefers recorded failure reasons", () => {
    expect(failureReasons({ ...base, missing: ["975"] })).toEqual(["missing 975"]);
    expect(failureReasons({ ...base, failures: ["misattributed: x"] })).toEqual(["misattributed: x"]);
  });
});
