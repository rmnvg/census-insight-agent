import { describe, expect, it } from "vitest";

import { formatDuration, groupSessions } from "./dates";

const session = (id: string, updated_at: string) => ({
  session_id: id,
  created_at: updated_at,
  updated_at,
  title: id,
  message_count: 2,
});

describe("groupSessions", () => {
  it("buckets by calendar day in display order", () => {
    const now = new Date("2026-09-24T12:00:00");
    const groups = groupSessions(
      [
        session("today", "2026-09-24T08:00:00"),
        session("yesterday", "2026-09-23T23:00:00"),
        session("week", "2026-09-19T10:00:00"),
        session("old", "2026-01-01T10:00:00"),
      ],
      now,
    );
    expect(groups.map((group) => [group.group, group.sessions.map((item) => item.session_id)])).toEqual([
      ["Today", ["today"]],
      ["Yesterday", ["yesterday"]],
      ["Previous 7 days", ["week"]],
      ["Older", ["old"]],
    ]);
  });

  it("formats durations", () => {
    expect([formatDuration(250), formatDuration(4200), formatDuration(75_000)]).toEqual(["250 ms", "4.2 s", "1 m 15 s"]);
  });
});
