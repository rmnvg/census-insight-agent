// Mirrors the FastAPI response models (backend/app/agent/models.py and friends).

export type SessionRecord = {
  session_id: string;
  created_at: string;
  updated_at: string;
  title: string | null;
};

export type SessionSummary = SessionRecord & { message_count: number };

export type EvidenceSpan = { evidence_id: string; start_offset: number; end_offset: number };

export type Citation = {
  citation_id: string;
  document_id: string;
  document_title: string;
  page_number: number;
  snippet: string;
  chunk_id: string;
  section_path: string[];
  evidence_span: EvidenceSpan;
};

export type ClaimDerivation = {
  operation: "sum" | "difference" | "percentage_difference";
  operands: number[];
  result: number;
  unit: string;
  input_claim_ids: string[];
};

export type Claim = {
  claim_id: string;
  text: string;
  citation_ids: string[];
  document_derived: boolean;
  metric: string | null;
  region: string | null;
  year: number | null;
  population_scope: string | null;
  residence_scope: string | null;
  value: number | null;
  unit: string | null;
  derivation: ClaimDerivation | null;
};

export type Artifact = {
  artifact_id: string;
  artifact_type: "chart" | "table";
  title: string;
  filename: string;
  media_type: string;
  byte_size: number;
  sha256: string;
  session_id: string;
  run_id: string;
  source_manifest_path: string;
  download_url: string;
};

export type ChatResponse = {
  answer: string;
  claims: Claim[];
  citations: Citation[];
  artifacts: Artifact[];
  limitations: string[];
  refusal: boolean;
  trace_id: string;
};

export type ApiError = {
  error_code: string;
  message: string;
  session_id?: string | null;
  trace_id?: string | null;
  retryable?: boolean;
};

export type ResearchSection = {
  heading: string;
  question: string;
  status: "answered" | "declined" | "failed";
  session_id: string;
  response: ChatResponse | null;
  error: ApiError | null;
};

export type ResearchReport = {
  topic: string;
  title: string;
  in_scope: boolean;
  reason: string | null;
  sections: ResearchSection[];
  verified_claim_count: number;
  citation_count: number;
  duration_seconds: number;
};

export type TranscriptEntry = {
  message_id: string;
  role: "user" | "assistant";
  created_at: string;
  content: string | null;
  mode?: "chat" | "research";
  response: ChatResponse | null;
  error: ApiError | null;
  report?: ResearchReport | null;
};

export type ResearchPlanEvent = {
  title: string;
  in_scope: boolean;
  reason: string | null;
  sections: { heading: string; question: string }[];
};

export type SessionTranscript = { session: SessionRecord; messages: TranscriptEntry[] };

export type TraceEvent = {
  timestamp: string;
  event: string;
  node: string | null;
  details: Record<string, unknown>;
  latency_ms: number | null;
};

export type ToolCall = {
  tool_name: string;
  arguments: Record<string, unknown>;
  status: string;
  result_count: number;
  error_type: string | null;
  latency_ms: number;
};

export type RunTrace = {
  session_id: string;
  run_id: string;
  events: TraceEvent[];
  tool_calls: ToolCall[];
  errors: string[];
  status: string;
  error_code: string | null;
  run_status: string;
  answer_status: string;
  refusal_reason: string | null;
};

export type DocumentSummary = {
  document_id: string;
  title: string;
  region: string;
  source_checksum: string;
};

export type CoverageSummary = {
  document: DocumentSummary;
  coverage: {
    document_id: string;
    pdf_page_count: number;
    indexed_pages: number;
    blank_decorative_pages: number;
    excluded_visual_pages: number;
    failed_mappings: number;
    percentage_pages_indexed: number;
  };
  limitations: { message: string; excluded_pages: number[] }[];
};

export type UploadJob = {
  job_id: string;
  document_id: string;
  title: string;
  region: string;
  original_filename: string;
  byte_size: number;
  page_count: number;
  source_checksum: string;
  status: "queued" | "processing" | "succeeded" | "failed";
  detail: string | null;
  indexed_pages: number;
  failed_pages: number[];
  chunks: number;
  created_at: string;
  updated_at: string;
};

export type HealthStatus = "ok" | "error" | "unknown";

export type ProgressStep = { node: string; label: string; at: number };

export type TrustCaseResult = {
  case_id: string;
  category: "lookup" | "comparison" | "ranking" | "chart" | "refusal";
  question: string;
  expected: string;
  passed: boolean;
  outcome: "answered" | "refused" | "error";
  answer_excerpt: string;
  observed_values: number[];
  missing: string[];
  claims_checked: number;
  ungrounded_claims: string[];
  arithmetic_errors: string[];
  uncited_claims: number;
  citation_count: number;
  artifact_types: string[];
  latency_seconds: number;
  attempts: number;
  error_code: string | null;
  trace_id: string | null;
  source_note: string;
  // Scorer version 2 fields; absent from older scorecards.
  verdict?: Verdict;
  misattributed_claims?: string[];
  failures?: string[];
  repeat?: number;
};

export type Verdict = "passed" | "wrong_answer" | "unsupported" | "incomplete" | "false_refusal" | "error";

export type Scorecard = {
  generated_at: string;
  model: string;
  cases_total: number;
  cases_passed: number;
  answerable_total: number;
  answerable_passed: number;
  refusal_total: number;
  refusal_passed: number;
  claims_checked: number;
  ungrounded_claims: number;
  uncited_claims: number;
  arithmetic_errors: number;
  wrong_answers: number;
  median_latency_seconds: number;
  results: TrustCaseResult[];
  scorer_version?: number;
  false_refusals?: number;
  unsupported_answers?: number;
  incomplete_answers?: number;
  misattributed_claims?: number;
  repeats?: number;
  unstable_cases?: string[];
};
