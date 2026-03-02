"use client";

import * as React from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { Badge } from "@/components/ui/badge";
import { ActivityIcon } from "@/components/ui/activity";
import {
  Area,
  Line,
  ComposedChart,
  ReferenceLine,
  XAxis,
  YAxis,
  Brush,
  ReferenceDot,
} from "recharts";
import type { DateRange } from "react-day-picker";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";
import { SessionMode, TimeWindowKey, SessionPoint, SessionStats } from "@/types/dashboard";
import { STARTING_EQUITY, WINDOW_LIMITS, MA_WINDOW, CHART_COLORS } from "@/lib/constants";
import { liveChartConfig, rollingMean, summarizeOHLC } from "@/lib/chart-configs";
import { fmtDateLabel, toYMD, fmtPnl, fmtStat } from "@/lib/formatters";
import { fetchSessionHistory } from "@/lib/api";

interface LiveSessionChartProps {
  data: SessionPoint[];
  onClear: () => void;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function ChartCrosshair({ points }: { points?: any[]; height?: number }) {
  if (!points?.[0]) return null;
  return (
    <line
      x1={points[0].x}
      y1={0}
      x2={points[0].x}
      y2={224}
      stroke="rgba(255,255,255,0.10)"
      strokeWidth={1}
      strokeDasharray="2 3"
    />
  );
}

export function LiveSessionChart({ data, onClear }: LiveSessionChartProps) {
  const [mode, setMode] = React.useState<SessionMode>("pnl");
  const [timeWindow, setTimeWindow] = React.useState<TimeWindowKey>("all");
  const [showMA, setShowMA] = React.useState(false);
  const [paused, setPaused] = React.useState(false);
  const [frozenData, setFrozenData] = React.useState<SessionPoint[]>([]);

  // History mode
  const [historyMode, setHistoryMode] = React.useState(false);
  const [dateRange, setDateRange] = React.useState<DateRange | undefined>();
  const [historyData, setHistoryData] = React.useState<SessionPoint[]>([]);
  const [historyLoading, setHistoryLoading] = React.useState(false);
  const [calendarOpen, setCalendarOpen] = React.useState(false);

  // Brush visible range — yDomain, stats, sessionHL derive from this slice only
  const [brushRange, setBrushRange] = React.useState<{ start: number; end: number } | null>(null);

  const loadHistory = React.useCallback(async (range: DateRange) => {
    if (!range.from) return;
    const from = toYMD(range.from);
    const to = toYMD(range.to ?? range.from);
    setHistoryLoading(true);
    const res = await fetchSessionHistory(from, to);
    setHistoryData(
      (res?.history ?? []).map((p) => ({
        time: p.ts.slice(11, 19),
        a: p.equity_a,
        b: p.equity_b,
        spy_dd: p.spy_dd ?? null,
        pos_a: p.pos_a ?? null,
        pos_b: p.pos_b ?? null,
      }))
    );
    setHistoryLoading(false);
  }, []);

  const handleTogglePause = () => {
    if (!paused) setFrozenData(data);
    setPaused((p) => !p);
  };

  React.useEffect(() => {
    if (!paused) setFrozenData(data);
  }, [data, paused]);

  const displayData = historyMode
    ? historyData
    : paused
      ? frozenData
      : data;

  const windowedData = React.useMemo(() => {
    if (historyMode) return displayData;
    const limit = WINDOW_LIMITS[timeWindow];
    return isFinite(limit) ? displayData.slice(-limit) : displayData;
  }, [displayData, timeWindow, historyMode]);

  // Derive chart data
  const chartData = React.useMemo(() => {
    if (windowedData.length === 0) return [];
    const pts = windowedData.map((pt) => {
      const pnlA = pt.a != null ? +(pt.a - STARTING_EQUITY).toFixed(2) : null;
      const pnlB = pt.b != null ? +(pt.b - STARTING_EQUITY).toFixed(2) : null;
      const sp =
        pt.a != null && pt.b != null ? +(pt.b - pt.a).toFixed(2) : null;
      return {
        time: pt.time,
        a: mode === "pnl" ? pnlA : null,
        b: mode === "pnl" ? pnlB : null,
        spread: mode === "spread" ? sp : null,
      };
    });
    const maA = rollingMean(pts.map((p) => p.a), MA_WINDOW);
    const maB = rollingMean(pts.map((p) => p.b), MA_WINDOW);
    const maS = rollingMean(pts.map((p) => p.spread), MA_WINDOW);
    return pts.map((p, i) => ({
      ...p,
      ma_a: showMA ? maA[i] : null,
      ma_b: showMA ? maB[i] : null,
      ma_spread: showMA ? maS[i] : null,
    }));
  }, [windowedData, mode, showMA]);

  // Keep brush viewport pinned to latest data as new points stream in
  React.useEffect(() => {
    if (chartData.length > 0) {
      const start = Math.max(0, chartData.length - 240);
      setBrushRange({ start, end: chartData.length - 1 });
    }
  }, [chartData.length]); // eslint-disable-line react-hooks/exhaustive-deps

  // The slice of chartData actually visible inside the Brush viewport
  const visibleData = React.useMemo(() => {
    if (!brushRange) return chartData;
    return chartData.slice(brushRange.start, brushRange.end + 1);
  }, [chartData, brushRange]);

  // Profitability-aware fill colours
  const lastPt = visibleData.length > 0 ? visibleData[visibleData.length - 1] : null;
  const aFill = (lastPt?.a ?? 0) >= 0 ? CHART_COLORS.positive : CHART_COLORS.negative;
  const bFill = (lastPt?.b ?? 0) >= 0 ? CHART_COLORS.neutral : CHART_COLORS.negative;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const sFill = ((lastPt as any)?.spread ?? 0) >= 0 ? CHART_COLORS.spread : CHART_COLORS.negative;

  // Stats — from visible data only
  const stats: SessionStats = React.useMemo(() => {
    const aVals = visibleData.map((d) => d.a).filter((v): v is number => v != null);
    const bVals = visibleData.map((d) => d.b).filter((v): v is number => v != null);
    const sVals = visibleData
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      .map((d) => (d as any).spread)
      .filter((v): v is number => v != null);
    return {
      a: summarizeOHLC(aVals),
      b: summarizeOHLC(bVals),
      spread: summarizeOHLC(sVals),
    };
  }, [visibleData]);

  // Session H/L for reference lines — from visible data only
  const sessionHigh = React.useMemo(() => {
    if (mode === "spread") return stats.spread?.high ?? null;
    const h = Math.max(stats.a?.high ?? -Infinity, stats.b?.high ?? -Infinity);
    return isFinite(h) ? h : null;
  }, [mode, stats]);

  const sessionLow = React.useMemo(() => {
    if (mode === "spread") return stats.spread?.low ?? null;
    const l = Math.min(stats.a?.low ?? Infinity, stats.b?.low ?? Infinity);
    return isFinite(l) ? l : null;
  }, [mode, stats]);

  // Y domain — from visible data only so Brush doesn't compress the viewport
  const yDomain: [number, number] = React.useMemo(() => {
    let vals: number[];
    if (mode === "spread") {
      vals = visibleData
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .map((pt) => (pt as any).spread)
        .filter((v): v is number => v != null);
    } else {
      vals = visibleData
        .flatMap((pt) => [pt.a, pt.b])
        .filter((v): v is number => v != null);
    }
    if (vals.length === 0) return mode === "spread" ? [-50, 50] : [-100, 100];
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const range = hi - lo;
    const minSpread = mode === "pnl" ? 20 : 10;
    const pad = Math.max(range * 0.15, minSpread / 2);
    return [lo - pad, hi + pad];
  }, [visibleData, mode]);

  // Jump dots (significant P&L jumps > $50)
  const jumpDots = React.useMemo(() => {
    const threshold = 50;
    return chartData.flatMap((pt, i) => {
      if (i === 0) return [];
      const prev = chartData[i - 1];
      const dots: Array<{ time: string; val: number; color: string }> = [];
      if (pt.a != null && prev.a != null && Math.abs(pt.a - prev.a) > threshold)
        dots.push({ time: pt.time, val: pt.a, color: CHART_COLORS.botA });
      if (pt.b != null && prev.b != null && Math.abs(pt.b - prev.b) > threshold)
        dots.push({ time: pt.time, val: pt.b, color: CHART_COLORS.botB });
      return dots;
    });
  }, [chartData]);

  const pillBtn = (active: boolean) =>
    `rounded px-2.5 py-0.5 text-[11px] font-medium transition-colors ${
      active
        ? "bg-foreground text-background"
        : "text-muted-foreground hover:text-foreground"
    }`;

  if (displayData.length < 2) {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-1.5">
                <ActivityIcon size={16} />
                Live Session
              </CardTitle>
              <CardDescription>
                Bot A vs B · sampled every 30s since page load
                {data.length > 0 && ` · ${data.length} pts`}
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex h-48 flex-col items-center justify-center gap-2">
            {historyMode && (
              <div className="flex items-center gap-2 mb-2">
                <button
                  onClick={() => {
                    setHistoryMode(false);
                    setDateRange(undefined);
                    setHistoryData([]);
                  }}
                  className={pillBtn(false)}
                >
                  ← Live
                </button>
                <Popover open={calendarOpen} onOpenChange={setCalendarOpen}>
                  <PopoverTrigger asChild>
                    <button className="rounded border border-border px-2 py-0.5 text-[11px] text-muted-foreground hover:text-foreground transition-colors">
                      {dateRange?.from ? fmtDateLabel(dateRange.from) : "Pick dates"}
                    </button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0" align="start">
                    <Calendar
                      mode="range"
                      selected={dateRange}
                      onSelect={(range) => {
                        setDateRange(range);
                        if (range?.from) {
                          loadHistory({
                            from: range.from,
                            to: range.to ?? range.from,
                          });
                          setCalendarOpen(false);
                        }
                      }}
                      disabled={{ after: new Date() }}
                      numberOfMonths={2}
                    />
                  </PopoverContent>
                </Popover>
              </div>
            )}
            <p className="text-xs text-muted-foreground">
              {historyMode
                ? historyLoading
                  ? "Loading history…"
                  : "No data for selected range — pick different dates"
                : displayData.length === 0
                  ? "Waiting for first poll…"
                  : `Collecting data — ${displayData.length}/2 points`}
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
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-1.5">
              <ActivityIcon size={16} />
              Live Session
            </CardTitle>
            <CardDescription>
              Bot A vs B · sampled every 30s since page load
              {data.length > 0 && ` · ${data.length} pts`}
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            {data.length > 0 && !historyMode && (
              <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
                LIVE
              </span>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {/* Control bar */}
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <div className="flex items-center gap-2 flex-wrap">
              {/* Mode */}
              <div className="flex items-center gap-1 rounded-md border border-border p-0.5">
                {(["pnl", "spread"] as SessionMode[]).map((m) => (
                  <button
                    key={m}
                    onClick={() => setMode(m)}
                    className={pillBtn(mode === m)}
                  >
                    {m === "pnl" ? "P&L Δ" : "B−A"}
                  </button>
                ))}
              </div>

              {/* Time window */}
              {!historyMode && (
                <div className="flex items-center gap-1 rounded-md border border-border p-0.5">
                  {(["30m", "1h", "2h", "4h", "all"] as TimeWindowKey[]).map(
                    (w) => (
                      <button
                        key={w}
                        onClick={() => setTimeWindow(w)}
                        className={pillBtn(timeWindow === w)}
                      >
                        {w === "all" ? "All" : w}
                      </button>
                    )
                  )}
                </div>
              )}

              {/* History toggle */}
              <div className="flex items-center gap-1">
                <button
                  onClick={() => {
                    setHistoryMode((v) => !v);
                    setDateRange(undefined);
                    setHistoryData([]);
                  }}
                  title={
                    historyMode ? "Back to live session" : "Browse historical sessions"
                  }
                  className={pillBtn(historyMode)}
                >
                  History
                </button>
                {historyMode && (
                  <Popover open={calendarOpen} onOpenChange={setCalendarOpen}>
                    <PopoverTrigger asChild>
                      <button className="rounded border border-border px-2 py-0.5 text-[11px] text-muted-foreground hover:text-foreground transition-colors">
                        {dateRange?.from
                          ? dateRange.to &&
                            dateRange.to.getTime() !== dateRange.from.getTime()
                            ? `${fmtDateLabel(dateRange.from)} – ${fmtDateLabel(dateRange.to)}`
                            : fmtDateLabel(dateRange.from)
                          : "Pick dates"}
                      </button>
                    </PopoverTrigger>
                    <PopoverContent className="w-auto p-0" align="start">
                      <Calendar
                        mode="range"
                        selected={dateRange}
                        onSelect={(range) => {
                          setDateRange(range);
                          if (range?.from) {
                            loadHistory({
                              from: range.from,
                              to: range.to ?? range.from,
                            });
                            setCalendarOpen(false);
                          }
                        }}
                        disabled={{ after: new Date() }}
                        numberOfMonths={2}
                      />
                    </PopoverContent>
                  </Popover>
                )}
              </div>

              {/* MA toggle */}
              <button
                onClick={() => setShowMA((v) => !v)}
                title={`${showMA ? "Hide" : "Show"} ${MA_WINDOW}-point moving average`}
                className={`rounded border border-border px-2 py-0.5 text-[11px] font-medium transition-colors ${
                  showMA
                    ? "bg-foreground text-background"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                MA{MA_WINDOW}
              </button>
            </div>

            {/* Right-side actions */}
            <div className="flex items-center gap-1">
              {!historyMode && (
                <button
                  onClick={handleTogglePause}
                  title={paused ? "Resume live data" : "Pause chart"}
                  className={`rounded border border-border px-2 py-0.5 text-[11px] font-medium transition-colors ${
                    paused
                      ? "bg-yellow-500/20 text-yellow-400 border-yellow-500/40"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {paused ? "▶" : "⏸"}
                </button>
              )}
              {!historyMode && (
                <button
                  onClick={onClear}
                  title="Clear session data"
                  className="rounded border border-border px-2 py-0.5 text-[11px] text-muted-foreground hover:text-foreground transition-colors"
                >
                  ×
                </button>
              )}
              {historyMode && historyLoading && (
                <span className="text-[10px] text-muted-foreground animate-pulse">
                  Loading…
                </span>
              )}
              {historyMode && !historyLoading && historyData.length > 0 && (
                <span className="text-[10px] text-muted-foreground tabular-nums">
                  {historyData.length.toLocaleString()} pts
                </span>
              )}
            </div>
          </div>

          {/* OHLC strip */}
          {chartData.length >= 2 && (
            <div className="flex items-center gap-4 text-[10px] text-muted-foreground tabular-nums flex-wrap">
              {mode !== "spread" ? (
                <>
                  {stats.a && (
                    <span className="flex items-center gap-2">
                      <span className="font-medium" style={{ color: CHART_COLORS.botA }}>
                        A
                      </span>
                      <span>
                        O <span className="text-foreground">{fmtStat(stats.a.open)}</span>
                      </span>
                      <span>
                        H <span className="text-green-500">{fmtStat(stats.a.high)}</span>
                      </span>
                      <span>
                        L <span className="text-red-500">{fmtStat(stats.a.low)}</span>
                      </span>
                      <span>
                        Now{" "}
                        <span className={stats.a.now >= 0 ? "text-green-500" : "text-red-500"}>
                          {fmtStat(stats.a.now)}
                        </span>
                      </span>
                    </span>
                  )}
                  {stats.b && (
                    <span className="flex items-center gap-2">
                      <span className="font-medium" style={{ color: CHART_COLORS.botB }}>
                        B
                      </span>
                      <span>
                        O <span className="text-foreground">{fmtStat(stats.b.open)}</span>
                      </span>
                      <span>
                        H <span className="text-green-500">{fmtStat(stats.b.high)}</span>
                      </span>
                      <span>
                        L <span className="text-red-500">{fmtStat(stats.b.low)}</span>
                      </span>
                      <span>
                        Now{" "}
                        <span className={stats.b.now >= 0 ? "text-green-500" : "text-red-500"}>
                          {fmtStat(stats.b.now)}
                        </span>
                      </span>
                    </span>
                  )}
                </>
              ) : (
                stats.spread && (
                  <span className="flex items-center gap-2">
                    <span className="font-medium" style={{ color: CHART_COLORS.spread }}>
                      B−A
                    </span>
                    <span>
                      O <span className="text-foreground">{fmtStat(stats.spread.open)}</span>
                    </span>
                    <span>
                      H <span className="text-green-500">{fmtStat(stats.spread.high)}</span>
                    </span>
                    <span>
                      L <span className="text-red-500">{fmtStat(stats.spread.low)}</span>
                    </span>
                    <span>
                      Now{" "}
                      <span
                        className={
                          stats.spread.now >= 0 ? "text-green-500" : "text-red-500"
                        }
                      >
                        {fmtStat(stats.spread.now)}
                      </span>
                    </span>
                  </span>
                )
              )}
            </div>
          )}

          {/* Legend */}
          <div className="flex items-center gap-4 text-[11px] text-muted-foreground">
            {mode !== "spread" ? (
              <>
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
              </>
            ) : (
              <span className="flex items-center gap-1.5">
                <span
                  className="inline-block h-0.5 w-4 rounded-full"
                  style={{ backgroundColor: CHART_COLORS.spread }}
                />
                Bot B − Bot A (positive = B winning)
              </span>
            )}
            {paused && <span className="text-yellow-400 text-[10px]">⏸ paused</span>}
            {chartData.length > 0 && (
              <>
                {mode === "spread" ? (
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  (() => {
                    const s = (chartData[chartData.length - 1] as any).spread;
                    if (s == null) return null;
                    return (
                      <span
                        className={`ml-auto tabular-nums font-medium ${
                          s >= 0 ? "text-green-500" : "text-red-500"
                        }`}
                      >
                        {s >= 0 ? "+" : ""}${s.toFixed(2)}
                      </span>
                    );
                  })()
                ) : (
                  <>
                    {chartData[chartData.length - 1].a != null && (
                      <span
                        className={`ml-auto tabular-nums font-medium ${
                          chartData[chartData.length - 1].a! >= 0
                            ? "text-green-500"
                            : "text-red-500"
                        }`}
                      >
                        A:{" "}
                        {chartData[chartData.length - 1].a! >= 0 ? "+" : ""}
                        ${Math.abs(chartData[chartData.length - 1].a!).toFixed(0)}
                      </span>
                    )}
                    {chartData[chartData.length - 1].b != null && (
                      <span
                        className={`tabular-nums font-medium ${
                          chartData[chartData.length - 1].b! >= 0
                            ? "text-green-500"
                            : "text-red-500"
                        }`}
                      >
                        B:{" "}
                        {chartData[chartData.length - 1].b! >= 0 ? "+" : ""}
                        ${Math.abs(chartData[chartData.length - 1].b!).toFixed(0)}
                      </span>
                    )}
                  </>
                )}
              </>
            )}
          </div>

          <ChartContainer config={liveChartConfig} className="h-56 w-full">
            <ComposedChart
              data={chartData}
              margin={{ top: 4, right: 4, left: 0, bottom: 0 }}
            >
              <defs>
                <linearGradient id="liveGradA" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={aFill} stopOpacity={0.2} />
                  <stop offset="100%" stopColor={aFill} stopOpacity={0} />
                </linearGradient>
                <linearGradient id="liveGradB" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={bFill} stopOpacity={0.2} />
                  <stop offset="100%" stopColor={bFill} stopOpacity={0} />
                </linearGradient>
                <linearGradient id="liveGradSpread" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={sFill} stopOpacity={0.24} />
                  <stop offset="100%" stopColor={sFill} stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="time" hide />
              <YAxis
                yAxisId="pnl"
                tickLine={false}
                axisLine={false}
                fontSize={9}
                width={58}
                domain={yDomain}
                tick={{ fill: "var(--muted-foreground)" }}
                tickFormatter={fmtPnl}
              />

              <ReferenceLine
                yAxisId="pnl"
                y={0}
                stroke="var(--border)"
                strokeDasharray="3 3"
                strokeWidth={1}
              />
              {sessionHigh != null && sessionHigh !== 0 && (
                <ReferenceLine
                  yAxisId="pnl"
                  y={sessionHigh}
                  stroke={CHART_COLORS.positive}
                  strokeDasharray="2 3"
                  strokeWidth={1}
                  strokeOpacity={0.55}
                />
              )}
              {sessionLow != null && sessionLow !== 0 && (
                <ReferenceLine
                  yAxisId="pnl"
                  y={sessionLow}
                  stroke={CHART_COLORS.negative}
                  strokeDasharray="2 3"
                  strokeWidth={1}
                  strokeOpacity={0.55}
                />
              )}
              <ChartTooltip
                cursor={<ChartCrosshair />}
                content={
                  <ChartTooltipContent
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    formatter={(value: any, name: any) => {
                      const k = String(name);
                      if (k.startsWith("ma_")) return null;
                      const v = Number(value);
                      if (k === "spread")
                        return (
                          <span className="tabular-nums">
                            B−A: {v >= 0 ? "+" : ""}${v.toFixed(2)}
                          </span>
                        );
                      const label = k === "a" ? "Bot A" : "Bot B";
                      return (
                        <span className="tabular-nums">
                          {label}: {v >= 0 ? "+" : ""}${Math.abs(v).toFixed(2)}
                        </span>
                      );
                    }}
                  />
                }
              />
              <Area
                yAxisId="pnl"
                type="monotone"
                dataKey="a"
                stroke={CHART_COLORS.botA}
                strokeWidth={1.5}
                fill="url(#liveGradA)"
                connectNulls
                dot={false}
                isAnimationActive={false}
              />
              <Area
                yAxisId="pnl"
                type="monotone"
                dataKey="b"
                stroke={CHART_COLORS.botB}
                strokeWidth={1.5}
                fill="url(#liveGradB)"
                connectNulls
                dot={false}
                isAnimationActive={false}
                strokeDasharray="4 3"
              />
              <Area
                yAxisId="pnl"
                type="monotone"
                dataKey="spread"
                stroke={CHART_COLORS.spread}
                strokeWidth={1.5}
                fill="url(#liveGradSpread)"
                connectNulls
                dot={false}
                isAnimationActive={false}
              />
              <Line
                yAxisId="pnl"
                type="monotone"
                dataKey="ma_a"
                stroke={CHART_COLORS.botA}
                strokeWidth={1}
                strokeOpacity={0.7}
                strokeDasharray="3 2"
                dot={false}
                isAnimationActive={false}
                connectNulls
              />
              <Line
                yAxisId="pnl"
                type="monotone"
                dataKey="ma_b"
                stroke={CHART_COLORS.botB}
                strokeWidth={1}
                strokeOpacity={0.7}
                strokeDasharray="3 2"
                dot={false}
                isAnimationActive={false}
                connectNulls
              />
              <Line
                yAxisId="pnl"
                type="monotone"
                dataKey="ma_spread"
                stroke={CHART_COLORS.spread}
                strokeWidth={1}
                strokeOpacity={0.7}
                strokeDasharray="3 2"
                dot={false}
                isAnimationActive={false}
                connectNulls
              />
              {jumpDots.map((d, i) => (
                <ReferenceDot
                  key={i}
                  yAxisId="pnl"
                  x={d.time}
                  y={d.val}
                  r={3}
                  fill={d.color}
                  stroke="var(--background)"
                  strokeWidth={1}
                />
              ))}
              <Brush
                dataKey="time"
                height={18}
                startIndex={Math.max(0, chartData.length - 240)}
                travellerWidth={5}
                stroke="var(--border)"
                fill="var(--card)"
                tickFormatter={(v: string) =>
                  typeof v === "string" ? v.slice(0, 5) : ""
                }
                onChange={(range: { startIndex?: number; endIndex?: number }) => {
                  if (range.startIndex != null && range.endIndex != null) {
                    setBrushRange({ start: range.startIndex, end: range.endIndex });
                  }
                }}
              />
            </ComposedChart>
          </ChartContainer>
        </div>
      </CardContent>
    </Card>
  );
}
