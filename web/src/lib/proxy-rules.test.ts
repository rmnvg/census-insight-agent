import { describe, expect, it } from "vitest";

import { isAllowed } from "./proxy-rules";

const SESSION = "6e594f6e52754ef5932df652638e3f0d";
const RUN = "bd52f034-88f2-4c4d-8a5f-938ebc0d0ed8";

describe("proxy allowlist", () => {
  it.each([
    ["GET", "health"],
    ["GET", "health/qdrant"],
    ["GET", "sessions"],
    ["POST", "sessions"],
    ["PATCH", `sessions/${SESSION}`],
    ["DELETE", `sessions/${SESSION}`],
    ["GET", `sessions/${SESSION}/messages`],
    ["GET", `sessions/${SESSION}/artifacts/${RUN}/files/chart.png`],
    ["POST", "chat/stream"],
    ["GET", `runs/${RUN}/trace`],
    ["GET", "documents/census-2011-karnataka-pca-highlights/pages/50"],
    ["POST", "documents/upload"],
    ["GET", "documents/uploads/0123456789abcdef0123456789abcdef"],
    ["DELETE", "documents/upload-kerala-0123456789"],
  ])("forwards %s /%s", (method, path) => {
    expect(isAllowed(method, path)).toBe(true);
  });

  it.each([
    ["POST", "admin/ingest"],
    ["POST", "retrieval/search"],
    ["GET", "openapi.json"],
    ["GET", "docs"],
    ["DELETE", "sessions"],
    ["GET", "sessions/not-a-session/messages"],
    ["GET", `sessions/${SESSION}/artifacts/${RUN}/files/execution-result.json`],
    ["GET", `sessions/${SESSION}/artifacts/${RUN}/files/../../../etc/passwd`],
    ["GET", "documents/../manifests/pages/1"],
    ["POST", "documents/x/pages/1"],
    ["GET", "health//executor"],
  ])("blocks %s /%s", (method, path) => {
    expect(isAllowed(method, path)).toBe(false);
  });
});
