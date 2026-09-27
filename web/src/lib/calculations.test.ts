import { describe, expect, it } from "vitest";

import { calculationCards } from "./calculations";
import { describeDerivation } from "./trace";
import type { ChatResponse, Citation, Claim } from "./types";

function comparison(): ChatResponse {
  const source = (id: string, region: string, value: number): Claim => ({
    claim_id: id, region, value, citation_ids: [`citation-${id}`], text: `${region}: ${value}`,
    document_derived: false, metric: "sex_ratio", year: 2011, population_scope: "persons",
    residence_scope: "total", unit: "females per 1000 males", derivation: null,
  });
  const left = source("od", "Odisha", 979);
  const right = source("ka", "Karnataka", 973);
  return {
    claims: [left, right, {
      ...left, claim_id: "gap", document_derived: true, region: null, value: 6, unit: "count",
      citation_ids: [...left.citation_ids, ...right.citation_ids],
      derivation: { operation: "difference", operands: [979, 973], result: 6, unit: "count", input_claim_ids: ["od", "ka"] },
    }],
    citations: ["od", "ka"].map((id) => ({ citation_id: `citation-${id}`, chunk_id: id }) as Citation),
    answer: "", artifacts: [], limitations: [], refusal: false, trace_id: "run",
  };
}

describe("calculation card lineage", () => {
  it("preserves backend operand order and each operand's own citations", () => {
    const response = comparison();
    response.claims.reverse();
    const [card] = calculationCards(response);
    expect(card.inputs.map(({ claim, citations }) => [claim.region, citations[0].chunk_id])).toEqual([
      ["Odisha", "od"], ["Karnataka", "ka"],
    ]);
    expect(describeDerivation(card.claim)).toBe("979 − 973 = 6");
  });

  it("does not guess a source by value when an input claim is missing", () => {
    const response = comparison();
    response.claims[0].claim_id = "unrelated-same-number";
    expect(calculationCards(response)).toEqual([]);
  });

  it("omits cards with mismatched operands or incomplete citation links", () => {
    const mismatch = comparison();
    mismatch.claims[0].value = 990;
    expect(calculationCards(mismatch)).toEqual([]);
    const missingCitation = comparison();
    missingCitation.citations.pop();
    expect(calculationCards(missingCitation)).toEqual([]);
    const missingOperand = comparison();
    missingOperand.claims[2].derivation!.input_claim_ids.pop();
    expect(calculationCards(missingOperand)).toEqual([]);
  });

  it("does not present calculations for refusals or source-only answers", () => {
    const response = comparison();
    response.refusal = true;
    expect(calculationCards(response)).toEqual([]);
    response.refusal = false;
    response.claims.pop();
    expect(calculationCards(response)).toEqual([]);
  });

  it("shows the actual relative-percent formula and distinguishes percentage points", () => {
    const claim = comparison().claims[2];
    claim.derivation = { operation: "percentage_difference", operands: [75, 60], result: 25, unit: "percent", input_claim_ids: ["a", "b"] };
    expect(describeDerivation(claim)).toBe("(75 − 60) / |60| × 100 = 25%");
    claim.derivation = { ...claim.derivation, operation: "difference", result: 15, unit: "percentage_points" };
    expect(describeDerivation(claim)).toBe("75 − 60 = 15 pp");
  });
});
