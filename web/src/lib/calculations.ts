import type { ChatResponse, Citation, Claim } from "./types";

export type CalculationCardModel = {
  claim: Claim;
  inputs: { claim: Claim; citations: Citation[] }[];
};

/** Follow explicit backend lineage; never infer operand identities from matching numbers. */
export function calculationCards(response: ChatResponse): CalculationCardModel[] {
  if (response.refusal) return [];
  const claims = new Map(response.claims.map((claim) => [claim.claim_id, claim]));
  const citations = new Map(response.citations.map((citation) => [citation.citation_id, citation]));
  return response.claims.flatMap((claim) => {
    const derivation = claim.derivation;
    if (!claim.document_derived || !derivation || !Number.isFinite(derivation.result)) return [];
    if (!derivation.operands.length || derivation.input_claim_ids.length !== derivation.operands.length) return [];
    const inputs: CalculationCardModel["inputs"] = [];
    for (const [index, id] of derivation.input_claim_ids.entries()) {
      const input = claims.get(id);
      if (!input || input.document_derived || input.value !== derivation.operands[index] || !Number.isFinite(input.value)) return [];
      const sources = input.citation_ids.map((id) => citations.get(id));
      if (!sources.length || sources.some((source) => !source)) return [];
      inputs.push({ claim: input, citations: sources as Citation[] });
    }
    return [{ claim, inputs }];
  });
}
