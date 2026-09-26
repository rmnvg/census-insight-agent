import type { TrustCaseResult, Verdict } from "./types";

/** The scorer's verdict, inferred for scorecards written before verdicts existed. */
export function verdictOf(result: TrustCaseResult): Verdict {
  if (result.verdict) return result.verdict;
  if (result.outcome === "error") return "error";
  if (result.passed) return "passed";
  if (result.outcome === "refused" && result.category !== "refusal") return "false_refusal";
  if (result.ungrounded_claims.length > 0) return "unsupported";
  if (result.category === "refusal" || result.arithmetic_errors.length > 0) return "wrong_answer";
  return "incomplete";
}

export const VERDICT_LABEL: Record<Verdict, string> = {
  passed: "Pass",
  wrong_answer: "Wrong",
  unsupported: "Unsupported",
  incomplete: "Incomplete",
  false_refusal: "Declined",
  error: "Error",
};

/** Why a case failed, most serious first; empty for a pass. */
export function failureReasons(result: TrustCaseResult): string[] {
  if (result.failures) return result.failures;
  return result.missing.map((item) => `missing ${item}`);
}
