import test from "node:test";
import assert from "node:assert/strict";
import { buildRecentActivity, filterScansByRange } from "./dashboardMetrics.js";

const now = Date.parse("2026-11-04T12:00:00.000Z");
const scans = [
  { id: "today", createdAt: "2026-11-04T10:00:00.000Z" },
  { id: "week", createdAt: "2026-10-30T12:00:00.000Z" },
  { id: "month", createdAt: "2026-10-10T12:00:00.000Z" },
  { id: "old", createdAt: "2026-09-01T12:00:00.000Z" },
  { id: "future", createdAt: "2026-11-05T12:00:00.000Z" },
  { id: "invalid", createdAt: "not-a-date" },
];

test("dashboard ranges include scans within the selected period only", () => {
  assert.deepEqual(filterScansByRange(scans, "7", now).map((scan) => scan.id), ["today", "week"]);
  assert.deepEqual(filterScansByRange(scans, "30", now).map((scan) => scan.id), ["today", "week", "month"]);
});

test("all-time range preserves the full scan list", () => {
  assert.equal(filterScansByRange(scans, "all", now), scans);
});

test("recent activity counts saved scans by local calendar day", () => {
  const activity = buildRecentActivity([
    { createdAt: "2026-11-04T10:00:00.000Z" },
    { createdAt: "2026-11-04T11:00:00.000Z" },
    { createdAt: "2026-11-04T13:00:00.000Z" },
    { createdAt: "2026-11-03T11:00:00.000Z" },
    { createdAt: "not-a-date" },
  ], now, 2);

  assert.deepEqual(activity.map(({ count }) => count), [1, 2]);
});
