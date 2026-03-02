import { type ChartConfig } from "@/components/ui/chart";
import { CHART_COLORS } from "@/lib/constants";

/**
 * Chart configuration objects for Recharts
 */

export const dualEquityChartConfig = {
  botA: { label: "Bot A", color: CHART_COLORS.botA },
  botB: { label: "Bot B", color: CHART_COLORS.botB },
} satisfies ChartConfig;

export const liveChartConfig = {
  a: { label: "Bot A", color: CHART_COLORS.botA },
  b: { label: "Bot B", color: CHART_COLORS.botB },
  ma_a: { label: "MA A", color: CHART_COLORS.botA },
  ma_b: { label: "MA B", color: CHART_COLORS.botB },
  ma_spread: { label: "MA B−A", color: CHART_COLORS.spread },
} satisfies ChartConfig;

/**
 * Return all Wednesday dates (YYYY-MM-DD) present in the data set.
 * Uses the actual date keys so ReferenceLine x values always match.
 */
export function extractWednesdays(dates: string[]): string[] {
  return dates.filter((d) => new Date(d + "T12:00:00Z").getUTCDay() === 3);
}

/**
 * Calculate rolling mean — requires exactly n non-null values in the window.
 */
export function rollingMean(vals: (number | null)[], n: number): (number | null)[] {
  return vals.map((_, i) => {
    if (i < n - 1) return null;
    const slice = vals.slice(i - n + 1, i + 1);
    if (slice.some((v) => v == null)) return null;
    return (slice as number[]).reduce((a, b) => a + b, 0) / n;
  });
}

/**
 * Summarize values into OHLC format
 */
export function summarizeOHLC(vals: number[]): { open: number; high: number; low: number; now: number } | null {
  if (vals.length === 0) return null;
  return {
    open: vals[0],
    high: Math.max(...vals),
    low: Math.min(...vals),
    now: vals[vals.length - 1],
  };
}
