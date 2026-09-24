import { describe, expect, it } from "vitest";

import { suggestFollowUps } from "./followups";
import type { ChatResponse, Claim } from "./types";

const claim = (fields: Partial<Claim>) => ({ claim_id: "c", text: "t", citation_ids: ["x"], ...fields }) as Claim;
const LIBRARY = ["Karnataka", "Madhya Pradesh", "Odisha"];

describe("suggestFollowUps", () => {
  it("offers a memory follow-up, chart, breakdown and ranking after a single lookup", () => {
    const response = {
      refusal: false,
      artifacts: [],
      citations: [{ citation_id: "x" }],
      claims: [claim({ metric: "Literacy Rate", region: "KARNATAKA", residence_scope: "total", value: 75.36 })],
    } as unknown as ChatResponse;
    expect(suggestFollowUps(response, LIBRARY).map((item) => item.prompt)).toEqual([
      "How does that compare with Madhya Pradesh?",
      "Create a bar chart comparing the literacy rate of Karnataka and Madhya Pradesh.",
      "Make a table of the literacy rate of Karnataka by rural and urban areas.",
      "Which district of Karnataka had the highest literacy rate?",
    ]);
  });

  it("charts a completed comparison and never re-suggests an answered region", () => {
    const response = {
      refusal: false,
      artifacts: [],
      citations: [{ citation_id: "x" }],
      claims: [
        claim({ metric: "sex_ratio", region: "Odisha" }),
        claim({ metric: "sex_ratio", region: "Madhya Pradesh" }),
        claim({ metric: "sex_ratio", derivation: { operation: "difference" } as Claim["derivation"] }),
      ],
    } as unknown as ChatResponse;
    const prompts = suggestFollowUps(response, LIBRARY).map((item) => item.prompt);
    expect(prompts[0]).toBe("Create a bar chart comparing the sex ratio of Odisha and Madhya Pradesh.");
    expect(prompts.some((prompt) => prompt.includes("compare with Odisha"))).toBe(false);
  });

  it("suggests answerable questions after a refusal", () => {
    const response = { refusal: true, claims: [], citations: [], artifacts: [] } as unknown as ChatResponse;
    expect(suggestFollowUps(response, LIBRARY)[0].prompt).toBe("What was the literacy rate of Karnataka in 2011?");
  });
});
