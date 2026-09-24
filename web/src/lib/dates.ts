import type { SessionSummary } from "./types";

const GROUPS = ["Today", "Yesterday", "Previous 7 days", "Previous 30 days", "Older"] as const;
export type SessionGroup = (typeof GROUPS)[number];

function startOfDay(date: Date): number {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
}

export function groupFor(updatedAt: string, now: Date = new Date()): SessionGroup {
  const days = Math.round((startOfDay(now) - startOfDay(new Date(updatedAt))) / 86_400_000);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days <= 7) return "Previous 7 days";
  if (days <= 30) return "Previous 30 days";
  return "Older";
}

export function groupSessions(
  sessions: SessionSummary[],
  now: Date = new Date(),
): { group: SessionGroup; sessions: SessionSummary[] }[] {
  const buckets = new Map<SessionGroup, SessionSummary[]>();
  for (const session of sessions) {
    const group = groupFor(session.updated_at, now);
    buckets.set(group, [...(buckets.get(group) ?? []), session]);
  }
  return GROUPS.filter((group) => buckets.has(group)).map((group) => ({
    group,
    sessions: buckets.get(group)!,
  }));
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  return `${Math.floor(seconds / 60)} m ${Math.round(seconds % 60)} s`;
}
