"use client";

import * as React from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { Badge } from "@/components/ui/badge";
import { PauseIcon } from "@/components/ui/pause";
import { ChartLineIcon } from "@/components/ui/chart-line";
import {
  Area,
  ComposedChart,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts";
import { EquityPoint, EquityDisplayMode, ChartDataPoint } from "@/types/dashboard";
import { STARTING_EQUITY, CHART_COLORS } from "@/lib/constants";
import { dualEquityChartConfig, extractWednesdays } from "@/lib/chart-configs";

interface DualEquityChartProps {
  dataA: EquityPoint[];
  dataB: EquityPoint[];
  isPaused?: boolean;
}

// Custom dot component for chart
interface DotProps {
  cx: number;
  cy: number;
  index: number;
  value?: number;
}

function LastValueDot({ cx, cy, index, value, color, label }: DotProps & { color: string; label: (v: number) => string }) {
  if (value == null) return null;
  return (
    <g>
      <circle cx={cx} cy={cy} r={2.5} fill={color} />
      <text
        x={cx + 5}
        y={cy}
        fontSize={9}
        fill={color}
        dominantBaseline="middle"
      >
        {label(value)}
      </text>
    </g>
  );
}

export function DualEquityChart({
  dataA,
  dataB,
  isPaused = false,
}: DualEquityChartProps) {
  const [equityMode, setEquityMode] = React.useState<EquityDisplayMode>("abs");

  // Merge both series by date (last value per day wins)
  const chartData: ChartDataPoint[] = React.useMemo(() => {
    const dateMap = new Map<string, ChartDataPoint>();

    for (const d of dataA) {
      const key = d.timestamp.slice(0, 10);
      const existing = dateMap.get(key) || { date: key };
      existing.botA = d.equity;
      dateMap.set(key, existing);
    }

    for (const d of dataB) {
      const key = d.timestamp.slice(0, 10);
      const existing = dateMap.get(key) || { date: key };
      existing.botB = d.equity;
      dateMap.set(key, existing);
    }

    return Array.from(dateMap.values()).sort((a, b) =>
      a.date.localeCompare(b.date)
    );
  }, [dataA, dataB]);

  const wednesdays = React.useMemo(
    () => extractWednesdays(chartData.map((d) => d.date)),
    [chartData]
  );

  const displayData = React.useMemo(() => {
    const toDisplay = (equity: number) =>
      equityMode === "abs"
        ? equity
        : +(((equity - STARTING_EQUITY) / STARTING_EQUITY) * 100).toFixed(3);

    return chartData.map((pt) => ({
      date: pt.date,
      botA: pt.botA != null ? toDisplay(pt.botA) : undefined,
      botB: pt.botB != null ? toDisplay(pt.botB) : undefined,
    }));
  }, [chartData, equityMode]);

  const fmtY = React.useCallback((v: number) =>
    equityMode === "abs"
      ? `$${(v / 1000).toFixed(1)}k`
      : `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`,
    [equityMode]
  );

  const yDomain: [number, number] = React.useMemo(() => {
    const vals = displayData
      .flatMap((pt) => [pt.botA, pt.botB])
      .filter((v): v is number => v != null);
    if (vals.length === 0)
      return equityMode === "abs" ? [99_000, 101_000] : [-1, 1];
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const spread = hi - lo;
    const minSpread = equityMode === "abs" ? 200 : 0.2;
    const pad = Math.max(spread * 0.15, minSpread / 2);
    return [lo - pad, hi + pad];
  }, [displayData, equityMode]);

  // Insufficient history — need ≥3 unique trading days
  const hasEnoughData = chartData.length >= 3;

  return (
    <Card className="col-span-3">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-1.5">
              <ChartLineIcon size={16} />
              Equity Curve
            </CardTitle>
            <CardDescription>Bot A vs Bot B — portfolio value over time</CardDescription>
          </div>
          {isPaused && (
            <Badge
              variant="secondary"
              className="flex items-center gap-1 text-[10px]"
            >
              <PauseIcon size={10} />
              PAUSED
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {!hasEnoughData ? (
          <div className="flex h-72 flex-col items-center justify-center gap-2">
            <p className="text-xs text-muted-foreground">
              Building history… {chartData.length}/3 days
            </p>
            <div className="flex gap-1">
              {[...Array(3)].map((_, i) => (
                <span
                  key={i}
                  className="inline-block h-1 w-4 rounded-full bg-muted animate-pulse"
                  style={{ animationDelay: `${i * 200}ms` }}
                />
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {/* Control bar */}
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1 rounded-md border border-border p-0.5">
                {(["abs", "pct"] as EquityDisplayMode[]).map((m) => (
                  <button
                    key={m}
                    onClick={() => setEquityMode(m)}
                    className={`rounded px-2.5 py-0.5 text-[11px] font-medium transition-colors ${
                      equityMode === m
                        ? "bg-foreground text-background"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {m === "abs" ? "Equity $" : "Return %"}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
                <span className="flex items-center gap-1.5">
                  <span
                    className="inline-block h-0.5 w-4 rounded-full"
                    style={{ backgroundColor: CHART_COLORS.botA }}
                  />
                  Bot A
                </span>
                <span className="flex items-center gap-1.5">
                  <span
                    className="inline-block h-0.5 w-4 rounded-full"
                    style={{
                      backgroundColor: CHART_COLORS.botB,
                      backgroundImage:
                        "repeating-linear-gradient(90deg, hsl(213 80% 58%) 0 4px, transparent 4px 7px)",
                    }}
                  />
                  Bot B
                </span>
                {wednesdays.length > 0 && (
                  <span className="flex items-center gap-1.5">
                    <span
                      className="inline-block h-3 border-l"
                      style={{
                        borderColor: CHART_COLORS.rebalance,
                        borderStyle: "dashed",
                      }}
                    />
                    Rebalance
                  </span>
                )}
              </div>
            </div>

            <ChartContainer config={dualEquityChartConfig} className="h-72 w-full">
              <ComposedChart
                data={displayData}
                margin={{ top: 4, right: 48, left: 0, bottom: 0 }}
              >
                <defs>
                  <linearGradient id="gradientA" x1="0" y1="0" x2="0" y2="1">
                    <stop
                      offset="0%"
                      stopColor={CHART_COLORS.botA}
                      stopOpacity={0.15}
                    />
                    <stop
                      offset="100%"
                      stopColor={CHART_COLORS.botA}
                      stopOpacity={0}
                    />
                  </linearGradient>
                  <linearGradient id="gradientB" x1="0" y1="0" x2="0" y2="1">
                    <stop
                      offset="0%"
                      stopColor={CHART_COLORS.botB}
                      stopOpacity={0.1}
                    />
                    <stop
                      offset="100%"
                      stopColor={CHART_COLORS.botB}
                      stopOpacity={0}
                    />
                  </linearGradient>
                </defs>
                <XAxis dataKey="date" hide />
                <YAxis
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={fmtY}
                  fontSize={10}
                  width={equityMode === "abs" ? 54 : 62}
                  domain={yDomain}
                  tick={{ fill: "var(--muted-foreground)" }}
                />
                {wednesdays.map((d) => (
                  <ReferenceLine
                    key={d}
                    x={d}
                    stroke={CHART_COLORS.rebalance}
                    strokeOpacity={0.4}
                    strokeDasharray="2 4"
                    strokeWidth={1}
                  />
                ))}
                <ChartTooltip
                  content={
                    <ChartTooltipContent
                      formatter={(value: unknown, name: unknown) => (
                        <span className="tabular-nums">
                          {String(name) === "botA" ? "A" : "B"}: {fmtY(Number(value))}
                        </span>
                      )}
                    />
                  }
                />
                <Area
                  type="monotone"
                  dataKey="botA"
                  stroke={CHART_COLORS.botA}
                  strokeWidth={1.5}
                  fill="url(#gradientA)"
                  connectNulls
                  dot={(props: DotProps) => {
                    if (props.index !== displayData.length - 1 || props.value == null) {
                      return <g key={props.index} />;
                    }
                    return (
                      <LastValueDot
                        key={props.index}
                        {...props}
                        color={CHART_COLORS.botA}
                        label={fmtY}
                      />
                    );
                  }}
                  isAnimationActive={false}
                />
                <Area
                  type="monotone"
                  dataKey="botB"
                  stroke={CHART_COLORS.botB}
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                  fill="url(#gradientB)"
                  connectNulls
                  dot={(props: DotProps) => {
                    if (props.index !== displayData.length - 1 || props.value == null) {
                      return <g key={props.index} />;
                    }
                    return (
                      <LastValueDot
                        key={props.index}
                        {...props}
                        color={CHART_COLORS.botB}
                        label={fmtY}
                      />
                    );
                  }}
                  isAnimationActive={false}
                />
              </ComposedChart>
            </ChartContainer>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
