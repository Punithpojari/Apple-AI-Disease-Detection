export function filterScansByRange(scans, range, now = Date.now()) {
  if (range === "all") return scans;

  const days = Number(range);
  if (!Number.isFinite(days) || days <= 0) return scans;

  const cutoff = now - days * 24 * 60 * 60 * 1000;
  return scans.filter((scan) => {
    const createdAt = new Date(scan.createdAt).getTime();
    return Number.isFinite(createdAt) && createdAt >= cutoff && createdAt <= now;
  });
}

export function buildRecentActivity(scans, now = Date.now(), days = 7) {
  const today = new Date(now);
  today.setHours(0, 0, 0, 0);

  return Array.from({ length: days }, (_, index) => {
    const date = new Date(today);
    date.setDate(today.getDate() - (days - index - 1));
    const key = `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`;
    const count = scans.filter((scan) => {
      const createdAt = new Date(scan.createdAt);
      return Number.isFinite(createdAt.getTime()) &&
        createdAt.getTime() <= now &&
        createdAt.getFullYear() === date.getFullYear() &&
        createdAt.getMonth() === date.getMonth() &&
        createdAt.getDate() === date.getDate();
    }).length;

    return {
      date: key,
      label: new Intl.DateTimeFormat(undefined, { weekday: "short" }).format(date),
      count,
    };
  });
}
