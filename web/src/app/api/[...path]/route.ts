import type { NextRequest } from "next/server";

import { isAllowed } from "@/lib/proxy-rules";

// Same-origin gateway: the browser never learns the backend address, and only allowlisted
// public endpoints are forwarded (see lib/proxy-rules.ts).
const BACKEND = (process.env.CENSUS_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const FORWARDED_RESPONSE_HEADERS = [
  "content-type",
  "content-disposition",
  "cache-control",
  "x-page-count",
  "x-highlight",
];

// Request bodies are checked against these caps before being read, then buffered so the
// backend always receives an exact Content-Length (it refuses length-less uploads).
const UPLOAD_MAX_BYTES = 51 * 1024 * 1024;
const JSON_MAX_BYTES = 64 * 1024;

export const dynamic = "force-dynamic";

function tooLarge(status: number, message: string) {
  return Response.json({ error_code: "REQUEST_REJECTED", message }, { status });
}

async function forward(request: NextRequest, ctx: RouteContext<"/api/[...path]">) {
  const { path } = await ctx.params;
  const joined = path.join("/");
  if (!isAllowed(request.method, joined)) {
    return Response.json({ error_code: "NOT_FOUND", message: "Not found" }, { status: 404 });
  }
  const target = `${BACKEND}/${joined}${request.nextUrl.search}`;
  const headers = new Headers();
  for (const name of ["content-type", "accept"]) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  let body: ArrayBuffer | undefined;
  if (!["GET", "HEAD", "DELETE"].includes(request.method)) {
    const limit = joined === "documents/upload" ? UPLOAD_MAX_BYTES : JSON_MAX_BYTES;
    const declared = Number(request.headers.get("content-length"));
    if (!Number.isFinite(declared) || declared <= 0) {
      if (joined === "documents/upload") return tooLarge(411, "Content-Length is required for uploads.");
    } else if (declared > limit) {
      return tooLarge(413, "The request is too large.");
    }
    body = await request.arrayBuffer();
    if (body.byteLength > limit) return tooLarge(413, "The request is too large.");
  }
  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: request.method,
      headers,
      body,
      signal: request.signal,
      cache: "no-store",
      redirect: "manual",
    });
  } catch {
    return Response.json(
      {
        error_code: "BACKEND_UNAVAILABLE",
        message: "The Census service is temporarily unavailable.",
        retryable: true,
      },
      { status: 502 },
    );
  }
  const responseHeaders = new Headers();
  for (const name of FORWARDED_RESPONSE_HEADERS) {
    const value = upstream.headers.get(name);
    if (value) responseHeaders.set(name, value);
  }
  if (upstream.headers.get("content-type")?.startsWith("text/event-stream")) {
    // `no-transform` stops response compression from buffering the event stream.
    responseHeaders.set("cache-control", "no-cache, no-transform");
    responseHeaders.set("x-accel-buffering", "no");
  }
  return new Response(upstream.body, { status: upstream.status, headers: responseHeaders });
}

export {
  forward as GET,
  forward as POST,
  forward as PATCH,
  forward as DELETE,
};
