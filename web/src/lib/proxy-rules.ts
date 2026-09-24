// The browser only reaches the backend through this allowlist, mirroring the Streamlit UI's
// trust boundary: admin ingestion, raw retrieval search, and OpenAPI docs are never exposed.

const SESSION = "[0-9a-f]{32}";
const UUID = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}";
const DOCUMENT = "[A-Za-z0-9._-]{1,128}";
const ARTIFACT_FILE = "(chart\\.png|plotted-data\\.csv|table\\.csv|table\\.md|source-manifest\\.json)";

const RULES: [methods: string[], pattern: RegExp][] = [
  [["GET"], /^health(\/executor|\/qdrant)?$/],
  [["GET", "POST"], /^sessions$/],
  [["GET", "PATCH", "DELETE"], new RegExp(`^sessions/${SESSION}$`)],
  [["GET"], new RegExp(`^sessions/${SESSION}/(messages|context|artifacts)$`)],
  [["GET"], new RegExp(`^sessions/${SESSION}/artifacts/${UUID}(/files/${ARTIFACT_FILE})?$`)],
  [["POST"], /^chat(\/stream)?$/],
  [["GET"], new RegExp(`^runs/${UUID}/trace$`)],
  [["GET"], /^documents$/],
  [["POST"], /^documents\/upload$/],
  [["GET"], /^documents\/uploads(\/[0-9a-f]{32})?$/],
  [["GET"], new RegExp(`^documents/${DOCUMENT}/coverage$`)],
  [["GET"], new RegExp(`^documents/${DOCUMENT}/pages/[0-9]{1,4}$`)],
  [["DELETE"], new RegExp(`^documents/${DOCUMENT}$`)],
];

export function isAllowed(method: string, path: string): boolean {
  if (path.includes("..") || path.includes("//")) return false;
  return RULES.some(([methods, pattern]) => methods.includes(method) && pattern.test(path));
}
