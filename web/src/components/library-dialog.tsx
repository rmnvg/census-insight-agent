"use client";

import {
  CheckCircle2,
  FileUp,
  Loader2,
  ShieldCheck,
  Trash2,
  Upload,
  XCircle,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { api, ApiRequestError } from "@/lib/api";
import type { CoverageSummary, DocumentSummary, UploadJob } from "@/lib/types";

import { useChat } from "./chat-provider";
import { Dialog } from "./dialog";

const MAX_BYTES = 50 * 1024 * 1024;

export function LibraryDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { documents, refreshDocuments } = useChat();
  const [jobs, setJobs] = useState<UploadJob[]>([]);

  const refreshJobs = useCallback(async () => {
    try {
      setJobs(await api.uploads());
    } catch {
      // Upload history is informational; keep the last known list.
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    // Refresh the library each time the dialog opens.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void Promise.all([refreshDocuments(), refreshJobs()]);
  }, [open, refreshDocuments, refreshJobs]);

  const active = jobs.some((job) => job.status === "queued" || job.status === "processing");
  useEffect(() => {
    if (!open || !active) return;
    const timer = window.setInterval(async () => {
      await refreshJobs();
      await refreshDocuments();
    }, 2500);
    return () => window.clearInterval(timer);
  }, [open, active, refreshJobs, refreshDocuments]);

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="Document library"
      subtitle="The agent answers only from these documents. Every citation points to a physical page of the original PDF."
    >
      <div className="space-y-6 p-5">
        <UploadForm onUploaded={refreshJobs} />
        {jobs.length > 0 && (
          <section>
            <h3 className="mb-2 text-xs font-medium uppercase tracking-wider text-zinc-400">Recent uploads</h3>
            <ul className="space-y-2">
              {jobs.slice(0, 5).map((job) => (
                <JobRow key={job.job_id} job={job} />
              ))}
            </ul>
          </section>
        )}
        <section>
          <h3 className="mb-2 text-xs font-medium uppercase tracking-wider text-zinc-400">
            Indexed documents · {documents.length}
          </h3>
          <ul className="divide-y divide-zinc-100 rounded-xl ring-1 ring-zinc-200 dark:divide-zinc-800 dark:ring-zinc-800">
            {documents.map((document) => (
              <DocumentRow
                key={document.document_id}
                document={document}
                onDeleted={async () => {
                  await refreshDocuments();
                  await refreshJobs();
                }}
              />
            ))}
            {documents.length === 0 && <li className="p-4 text-sm text-zinc-500">No documents are indexed yet.</li>}
          </ul>
        </section>
      </div>
    </Dialog>
  );
}

function titleFromFilename(name: string): string {
  return name
    .replace(/\.pdf$/i, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function UploadForm({ onUploaded }: { onUploaded: () => Promise<void> }) {
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [region, setRegion] = useState("");
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const choose = (candidate: File | undefined) => {
    setError(null);
    if (!candidate) return;
    if (candidate.type && candidate.type !== "application/pdf" && !candidate.name.toLowerCase().endsWith(".pdf")) {
      setError("Please choose a PDF file.");
      return;
    }
    if (candidate.size > MAX_BYTES) {
      setError("PDFs must be 50 MB or smaller.");
      return;
    }
    setFile(candidate);
    if (!title) setTitle(titleFromFilename(candidate.name));
  };

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!file || !title.trim() || !region.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await api.uploadDocument(file, title.trim(), region.trim());
      setFile(null);
      setTitle("");
      setRegion("");
      await onUploaded();
    } catch (caught) {
      setError(caught instanceof ApiRequestError ? caught.error.message : "Upload failed.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={submit} className="space-y-3">
      <div
        role="button"
        tabIndex={0}
        onClick={() => input.current?.click()}
        onKeyDown={(event) => (event.key === "Enter" || event.key === " ") && input.current?.click()}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          choose(event.dataTransfer.files[0]);
        }}
        className={`flex cursor-pointer flex-col items-center gap-2 rounded-xl border-2 border-dashed px-4 py-6 text-center transition ${
          dragging
            ? "border-teal-600 bg-teal-50/60 dark:bg-teal-950/30"
            : "border-zinc-300 hover:border-zinc-400 dark:border-zinc-700 dark:hover:border-zinc-600"
        }`}
      >
        <FileUp className="size-6 text-zinc-400" />
        {file ? (
          <p className="text-sm font-medium">
            {file.name} <span className="font-normal text-zinc-500">· {(file.size / 1024 / 1024).toFixed(1)} MB</span>
          </p>
        ) : (
          <>
            <p className="text-sm font-medium">Drop a PDF here or click to browse</p>
            <p className="text-xs text-zinc-500">Text-layer PDFs up to 50 MB and 400 pages</p>
          </>
        )}
        <input
          ref={input}
          type="file"
          accept="application/pdf,.pdf"
          className="hidden"
          onChange={(event) => choose(event.target.files?.[0])}
        />
      </div>
      {file && (
        <div className="grid gap-3 sm:grid-cols-[1fr_180px_auto]">
          <input
            value={title}
            onChange={(event) => setTitle(event.target.value)}
            placeholder="Document title"
            maxLength={200}
            required
            className="rounded-lg border border-zinc-300 bg-transparent px-3 py-2 text-sm outline-none focus:border-teal-600 dark:border-zinc-700"
          />
          <input
            value={region}
            onChange={(event) => setRegion(event.target.value)}
            placeholder="Region (e.g. Kerala)"
            maxLength={80}
            required
            className="rounded-lg border border-zinc-300 bg-transparent px-3 py-2 text-sm outline-none focus:border-teal-600 dark:border-zinc-700"
          />
          <button
            type="submit"
            disabled={busy || !title.trim() || !region.trim()}
            className="flex items-center justify-center gap-1.5 rounded-lg bg-teal-700 px-4 py-2 text-sm font-medium text-white hover:bg-teal-800 disabled:opacity-50"
          >
            {busy ? <Loader2 className="size-4 animate-spin" /> : <Upload className="size-4" />}
            Index
          </button>
        </div>
      )}
      {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}
      <p className="flex items-start gap-1.5 text-[11px] leading-4 text-zinc-500">
        <ShieldCheck className="mt-px size-3.5 shrink-0" />
        Uploads go through the same citation-safe pipeline as the built-in reports: page-level extraction,
        checksum-bound chunks, and hybrid indexing. Pages without a text layer are excluded rather than guessed.
      </p>
    </form>
  );
}

function JobRow({ job }: { job: UploadJob }) {
  const running = job.status === "queued" || job.status === "processing";
  return (
    <li className="flex items-start gap-3 rounded-xl bg-zinc-50 p-3 text-sm dark:bg-zinc-800/50">
      {running ? (
        <Loader2 className="mt-0.5 size-4 shrink-0 animate-spin text-teal-600" />
      ) : job.status === "succeeded" ? (
        <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-emerald-600" />
      ) : (
        <XCircle className="mt-0.5 size-4 shrink-0 text-red-600" />
      )}
      <div className="min-w-0 flex-1">
        <p className="truncate font-medium">{job.title}</p>
        <p className="text-xs text-zinc-500">
          {running
            ? job.status === "queued"
              ? "Queued…"
              : `Extracting, embedding and indexing ${job.page_count} pages…`
            : job.detail}
        </p>
      </div>
      <span className="shrink-0 rounded-full bg-zinc-200 px-2 py-0.5 text-[11px] dark:bg-zinc-700">{job.region}</span>
    </li>
  );
}

function DocumentRow({ document, onDeleted }: { document: DocumentSummary; onDeleted: () => Promise<void> }) {
  const [coverage, setCoverage] = useState<CoverageSummary | null>(null);
  const [confirm, setConfirm] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const uploaded = document.document_id.startsWith("upload-");

  useEffect(() => {
    api.coverage(document.document_id).then(setCoverage, () => setCoverage(null));
  }, [document.document_id]);

  const percent = coverage?.coverage.percentage_pages_indexed;
  return (
    <li className="flex items-center gap-3 p-3">
      <div className="min-w-0 flex-1">
        <p className="flex items-center gap-2 truncate text-sm font-medium">
          <span className="truncate">{document.title}</span>
          {uploaded && (
            <span className="rounded bg-teal-50 px-1.5 text-[10px] font-semibold uppercase tracking-wide text-teal-700 dark:bg-teal-950/60 dark:text-teal-300">
              Uploaded
            </span>
          )}
        </p>
        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-zinc-500">
          <span>{document.region}</span>
          {coverage && (
            <span className="flex items-center gap-1.5" title="Pages indexed for automated answering">
              <span className="h-1.5 w-16 overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-700">
                <span className="block h-full rounded-full bg-teal-600" style={{ width: `${percent}%` }} />
              </span>
              {coverage.coverage.indexed_pages}/{coverage.coverage.pdf_page_count} pages
            </span>
          )}
          <span className="font-mono" title={`SHA-256 ${document.source_checksum}`}>
            sha256 {document.source_checksum.slice(0, 10)}…
          </span>
        </div>
      </div>
      {uploaded &&
        (confirm ? (
          <div className="flex items-center gap-1 text-xs">
            <button
              type="button"
              disabled={deleting}
              onClick={async () => {
                setDeleting(true);
                try {
                  await api.deleteDocument(document.document_id);
                  await onDeleted();
                } finally {
                  setDeleting(false);
                  setConfirm(false);
                }
              }}
              className="rounded-md bg-red-600 px-2 py-1 font-medium text-white hover:bg-red-700"
            >
              {deleting ? "Deleting…" : "Delete"}
            </button>
            <button type="button" onClick={() => setConfirm(false)} className="rounded-md px-2 py-1 hover:bg-zinc-100 dark:hover:bg-zinc-800">
              Cancel
            </button>
          </div>
        ) : (
          <button
            type="button"
            aria-label={`Delete ${document.title}`}
            onClick={() => setConfirm(true)}
            className="rounded-md p-1.5 text-zinc-400 hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-950/40"
          >
            <Trash2 className="size-4" />
          </button>
        ))}
    </li>
  );
}
