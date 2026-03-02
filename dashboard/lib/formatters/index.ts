/**
 * Formatting utilities for dashboard data display
 */

/**
 * Format a number with specified decimal places
 */
export function fmt(n: number | undefined | null, decimals = 2): string {
  if (n == null || isNaN(n)) return "—";
  return n.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

/**
 * Format a number as USD currency
 */
export function fmtUsd(n: number | undefined | null): string {
  if (n == null || isNaN(n)) return "—";
  return `$${fmt(n)}`;
}

/**
 * Format a number as percentage
 */
export function fmtPct(n: number | undefined | null): string {
  if (n == null || isNaN(n)) return "—";
  return `${fmt(n * 100)}%`;
}

/**
 * Format a date string to human-readable format
 */
export function fmtDate(s: string | undefined | null): string {
  if (!s) return "—";
  try {
    return new Date(s).toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return s;
  }
}

/**
 * Format timezone-specific time
 */
export function fmtTz(date: Date, tz: string, label: string): string {
  return `${label} ${date.toLocaleTimeString("en-US", {
    timeZone: tz,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  })}`;
}

/**
 * Format P&L value with sign
 */
export function fmtPnl(v: number): string {
  return `${v >= 0 ? "+" : ""}$${v.toFixed(0)}`;
}

/**
 * Format stat value for OHLC display
 */
export function fmtStat(v: number): string {
  return `${v >= 0 ? "+" : ""}$${Math.abs(v).toFixed(0)}`;
}

/**
 * Format date label (e.g., "Mar 15")
 */
export function fmtDateLabel(d: Date): string {
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

/**
 * Convert date to YYYY-MM-DD format
 */
export function toYMD(d: Date): string {
  return d.toISOString().slice(0, 10);
}
