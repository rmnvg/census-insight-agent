import { describe, expect, it } from "vitest";

import { SseParser } from "./sse";

describe("SseParser", () => {
  it("parses events split across arbitrary chunk boundaries", () => {
    const stream =
      'event: started\ndata: {"a":1}\n\n: keepalive\n\nevent: progress\ndata: {"node":"plan"}\n\nevent: result\ndata: {"answer":"x"}\n\n';
    const parser = new SseParser();
    const events = [];
    for (let index = 0; index < stream.length; index += 7) events.push(...parser.push(stream.slice(index, index + 7)));
    expect(events).toEqual([
      { event: "started", data: '{"a":1}' },
      { event: "progress", data: '{"node":"plan"}' },
      { event: "result", data: '{"answer":"x"}' },
    ]);
  });

  it("handles CRLF line endings and multi-line data", () => {
    const parser = new SseParser();
    expect(parser.push("event: error\r\ndata: a\r\ndata: b\r\n\r\n")).toEqual([{ event: "error", data: "a\nb" }]);
  });
});
