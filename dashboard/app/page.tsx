"use client";

import * as React from "react";
import {
  BotId,
  StatusData,
  Position,
  RiskData,
  TcaData,
  DriftData,
  EquityPoint,
} from "@/lib/types";
import {
  fetchStatus,
  fetchPositions,
  fetchRisk,
  fetchTca,
  fetchDrift,
  fetchEquity,
  fetchLogs,
  fetchHealth,
  storeSessionPoint,
  fetchSessionHistory,
} from "@/lib/api";
import { SessionPoint } from "@/lib/types";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableFooter,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import type { DateRange } from "react-day-picker";
import { Area, Brush, ComposedChart, Line, ReferenceDot, ReferenceLine, XAxis, YAxis } from "recharts";
import { SparklesIcon } from "@/components/ui/sparkles";
import { ClockIcon } from "@/components/ui/clock";
import { RefreshCWIcon } from "@/components/ui/refresh-cw";
import { PauseIcon } from "@/components/ui/pause";
import { BotIcon } from "@/components/ui/bot";
import { DollarSignIcon } from "@/components/ui/dollar-sign";
import { GaugeIcon } from "@/components/ui/gauge";
import { ActivityIcon } from "@/components/ui/activity";
import { ChartLineIcon } from "@/components/ui/chart-line";
import { ShieldCheckIcon } from "@/components/ui/shield-check";
import { ChartBarIncreasingIcon } from "@/components/ui/chart-bar-increasing";
import { LayersIcon } from "@/components/ui/layers";
import { TerminalIcon } from "@/components/ui/terminal";
import { WavesIcon } from "@/components/ui/waves";
import { TrendingUpIcon } from "@/components/ui/trending-up";
import { TrendingDownIcon } from "@/components/ui/trending-down";

// ── Helpers ──────────────────────────────────────────────────────────────

function fmt(n: number | undefined | null, decimals = 2): string {
  if (n == null || isNaN(n)) return "—";
  return n.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function fmtUsd(n: number | undefined | null): string {
  if (n == null || isNaN(n)) return "—";
  return `$${fmt(n)}`;
}

function fmtPct(n: number | undefined | null): string {
  if (n == null || isNaN(n)) return "—";
  return `${fmt(n * 100)}%`;
}

function fmtDate(s: string | undefined | null): string {
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

function regimeColor(regime: string | undefined): string {
  switch (regime) {
    case "normal":
      return "outline";
    case "caution":
      return "secondary";
    case "halt":
      return "destructive";
    default:
      return "outline";
  }
}

const STARTING_EQUITY = 100_000;

// S&P 500 sector map (top holdings — covers most likely positions)
const SECTOR_MAP: Record<string, string> = {
  AAPL: "Technology", MSFT: "Technology", NVDA: "Technology", GOOG: "Technology",
  GOOGL: "Technology", META: "Technology", AVGO: "Technology", ADBE: "Technology",
  CRM: "Technology", CSCO: "Technology", ORCL: "Technology", ACN: "Technology",
  AMD: "Technology", INTC: "Technology", IBM: "Technology", QCOM: "Technology",
  TXN: "Technology", NOW: "Technology", AMAT: "Technology", MU: "Technology",
  INTU: "Technology", LRCX: "Technology", KLAC: "Technology", SNPS: "Technology",
  CDNS: "Technology", MRVL: "Technology", FTNT: "Technology", PANW: "Technology",
  AMZN: "Consumer Disc.", TSLA: "Consumer Disc.", HD: "Consumer Disc.",
  MCD: "Consumer Disc.", NKE: "Consumer Disc.", LOW: "Consumer Disc.",
  SBUX: "Consumer Disc.", TJX: "Consumer Disc.", BKNG: "Consumer Disc.",
  CMG: "Consumer Disc.", ORLY: "Consumer Disc.", MAR: "Consumer Disc.",
  BRK: "Financials", JPM: "Financials", V: "Financials", MA: "Financials",
  BAC: "Financials", WFC: "Financials", GS: "Financials", MS: "Financials",
  SPGI: "Financials", BLK: "Financials", AXP: "Financials", C: "Financials",
  SCHW: "Financials", CB: "Financials", MMC: "Financials", PGR: "Financials",
  UNH: "Healthcare", JNJ: "Healthcare", LLY: "Healthcare", ABBV: "Healthcare",
  MRK: "Healthcare", PFE: "Healthcare", TMO: "Healthcare", ABT: "Healthcare",
  DHR: "Healthcare", BMY: "Healthcare", AMGN: "Healthcare", GILD: "Healthcare",
  ISRG: "Healthcare", VRTX: "Healthcare", SYK: "Healthcare", BSX: "Healthcare",
  MDT: "Healthcare", REGN: "Healthcare", ZTS: "Healthcare", ELV: "Healthcare",
  XOM: "Energy", CVX: "Energy", COP: "Energy", SLB: "Energy", EOG: "Energy",
  MPC: "Energy", PSX: "Energy", VLO: "Energy", OXY: "Energy", HES: "Energy",
  PG: "Consumer Stap.", KO: "Consumer Stap.", PEP: "Consumer Stap.",
  COST: "Consumer Stap.", WMT: "Consumer Stap.", PM: "Consumer Stap.",
  MO: "Consumer Stap.", CL: "Consumer Stap.", MDLZ: "Consumer Stap.",
  LIN: "Materials", APD: "Materials", SHW: "Materials", ECL: "Materials",
  FCX: "Materials", NEM: "Materials", NUE: "Materials", DOW: "Materials",
  NEE: "Utilities", DUK: "Utilities", SO: "Utilities", D: "Utilities",
  AEP: "Utilities", SRE: "Utilities", EXC: "Utilities", XEL: "Utilities",
  AMT: "Real Estate", PLD: "Real Estate", CCI: "Real Estate",
  EQIX: "Real Estate", SPG: "Real Estate", PSA: "Real Estate",
  UNP: "Industrials", RTX: "Industrials", HON: "Industrials", UPS: "Industrials",
  BA: "Industrials", CAT: "Industrials", DE: "Industrials", LMT: "Industrials",
  GE: "Industrials", MMM: "Industrials", GD: "Industrials", NOC: "Industrials",
  T: "Communication", VZ: "Communication", TMUS: "Communication",
  DIS: "Communication", CMCSA: "Communication", NFLX: "Communication",
};

function notify(message: string) {
  if (typeof window === "undefined") return;
  if ("Notification" in window && Notification.permission === "granted") {
    new Notification("Signum", { body: message });
  } else if ("Notification" in window && Notification.permission !== "denied") {
    Notification.requestPermission().then((perm) => {
      if (perm === "granted") new Notification("Signum", { body: message });
    });
  }
}

function fmtTz(date: Date, tz: string, label: string): string {
  return `${label} ${date.toLocaleTimeString("en-US", {
    timeZone: tz,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  })}`;
}

type MarketSession = "Pre-market" | "Open" | "After-hours" | "Closed";

function getMarketStatus(now: Date): {
  session: MarketSession;
  countdown: string;
} {
  const ny = new Date(
    now.toLocaleString("en-US", { timeZone: "America/New_York" })
  );
  const day = ny.getDay();
  const h = ny.getHours();
  const m = ny.getMinutes();
  const mins = h * 60 + m;

  const isWeekday = day >= 1 && day <= 5;

  // Market hours in minutes: pre 4:00-9:30, open 9:30-16:00, after 16:00-20:00
  const PRE = 240; // 4:00
  const OPEN = 570; // 9:30
  const CLOSE = 960; // 16:00
  const AFTER = 1200; // 20:00

  let session: MarketSession = "Closed";
  let targetMins = OPEN; // default: next open
  let targetDay = ny;

  if (isWeekday) {
    if (mins >= OPEN && mins < CLOSE) {
      session = "Open";
      targetMins = CLOSE;
    } else if (mins >= PRE && mins < OPEN) {
      session = "Pre-market";
      targetMins = OPEN;
    } else if (mins >= CLOSE && mins < AFTER) {
      session = "After-hours";
      // Next open is tomorrow (or Monday)
      targetMins = OPEN;
      targetDay = new Date(ny);
      targetDay.setDate(targetDay.getDate() + (day === 5 ? 3 : 1));
    } else {
      session = "Closed";
      if (mins >= AFTER) {
        targetDay = new Date(ny);
        targetDay.setDate(targetDay.getDate() + (day === 5 ? 3 : 1));
      }
      targetMins = OPEN;
    }
  } else {
    // Weekend
    const daysToMon = day === 0 ? 1 : 6 - day + 2;
    targetDay = new Date(ny);
    targetDay.setDate(targetDay.getDate() + daysToMon);
    targetMins = OPEN;
  }

  // Calculate countdown
  let diffMs: number;
  if (session === "Open") {
    // Time until close
    const targetDate = new Date(ny);
    targetDate.setHours(Math.floor(targetMins / 60), targetMins % 60, 0, 0);
    diffMs = targetDate.getTime() - ny.getTime();
  } else {
    // Time until next open
    const targetDate = new Date(targetDay);
    targetDate.setHours(
      Math.floor(targetMins / 60),
      targetMins % 60,
      0,
      0
    );
    if (session !== "Closed" || isWeekday) {
      // Same day target
      if (session === "Pre-market") {
        const t = new Date(ny);
        t.setHours(Math.floor(OPEN / 60), OPEN % 60, 0, 0);
        diffMs = t.getTime() - ny.getTime();
      } else {
        diffMs = targetDate.getTime() - ny.getTime();
      }
    } else {
      diffMs = targetDate.getTime() - ny.getTime();
    }
  }

  if (diffMs < 0) diffMs = 0;
  const totalSec = Math.floor(diffMs / 1000);
  const hours = Math.floor(totalSec / 3600);
  const minutes = Math.floor((totalSec % 3600) / 60);
  const seconds = totalSec % 60;

  const countdown =
    hours > 0
      ? `${hours}h ${String(minutes).padStart(2, "0")}m ${String(seconds).padStart(2, "0")}s`
      : `${minutes}m ${String(seconds).padStart(2, "0")}s`;

  return { session, countdown };
}

function sessionColor(session: MarketSession): string {
  switch (session) {
    case "Open":
      return "outline";
    case "Pre-market":
    case "After-hours":
      return "secondary";
    case "Closed":
      return "destructive";
  }
}

// ── Main Page ────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const [bot, setBot] = React.useState<BotId>("bot-a");
  const [status, setStatus] = React.useState<StatusData | null>(null);
  const [positions, setPositions] = React.useState<Position[]>([]);
  const [risk, setRisk] = React.useState<RiskData | null>(null);
  const [tca, setTca] = React.useState<TcaData | null>(null);
  const [drift, setDrift] = React.useState<DriftData | null>(null);
  const [equity, setEquity] = React.useState<EquityPoint[]>([]);
  const [logs, setLogs] = React.useState<string>("");
  const [healthy, setHealthy] = React.useState<boolean | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [lastRefresh, setLastRefresh] = React.useState<Date>(new Date());
  const [now, setNow] = React.useState<Date>(new Date());
  const [refreshIn, setRefreshIn] = React.useState(30);
  const [prevRegimeA, setPrevRegimeA] = React.useState<string | null>(null);
  const [prevRegimeB, setPrevRegimeB] = React.useState<string | null>(null);
  const [equityA, setEquityA] = React.useState<EquityPoint[]>([]);
  const [equityB, setEquityB] = React.useState<EquityPoint[]>([]);
  const [paused, setPaused] = React.useState(false);

  // Live session accumulator — persisted to localStorage, survives reloads
  const [sessionPoints, setSessionPoints] = React.useState<
    Array<{ time: string; a: number | null; b: number | null; spy_dd: number | null; pos_a: number | null; pos_b: number | null }>
  >(() => {
    try {
      const stored = localStorage.getItem("signum_session_equity");
      if (stored) {
        // Backward-compat: old rows lack spy_dd/pos_a/pos_b — fill with null
        const parsed = JSON.parse(stored) as Array<{ time: string; a: number | null; b: number | null; spy_dd?: number | null; pos_a?: number | null; pos_b?: number | null }>;
        return parsed.map((p) => ({ spy_dd: null, pos_a: null, pos_b: null, ...p }));
      }
    } catch {}
    return [];
  });

  // Persist session equity to localStorage on every update
  React.useEffect(() => {
    try {
      localStorage.setItem("signum_session_equity", JSON.stringify(sessionPoints));
    } catch {}
  }, [sessionPoints]);

  // Dynamic page title
  React.useEffect(() => {
    const eq = status?.account?.equity;
    const label = bot === "bot-a" ? "A" : "B";
    document.title = eq != null ? `${fmtUsd(eq)} (${label}) | Signum` : "Signum";
  }, [status, bot]);

  // 1-second clock tick + refresh countdown
  React.useEffect(() => {
    const tick = setInterval(() => {
      setNow(new Date());
      setRefreshIn((prev) => (prev <= 1 ? 30 : prev - 1));
    }, 1000);
    // Request notification permission
    if ("Notification" in window && Notification.permission === "default") {
      Notification.requestPermission();
    }
    return () => clearInterval(tick);
  }, []);

  const market = getMarketStatus(now);

  // Comparison strip data (always both bots)
  const [statusA, setStatusA] = React.useState<StatusData | null>(null);
  const [statusB, setStatusB] = React.useState<StatusData | null>(null);
  const [healthA, setHealthA] = React.useState<boolean | null>(null);
  const [healthB, setHealthB] = React.useState<boolean | null>(null);

  const loadComparison = React.useCallback(async () => {
    const [sA, sB, hA, hB] = await Promise.all([
      fetchStatus("bot-a"),
      fetchStatus("bot-b"),
      fetchHealth("bot-a"),
      fetchHealth("bot-b"),
    ]);

    // Regime change notifications
    const regimeA = sA?.regime?.regime;
    const regimeB = sB?.regime?.regime;
    if (
      prevRegimeA &&
      regimeA &&
      regimeA !== prevRegimeA &&
      regimeA !== "normal"
    ) {
      notify(`Bot A regime: ${regimeA.toUpperCase()}`);
    }
    if (
      prevRegimeB &&
      regimeB &&
      regimeB !== prevRegimeB &&
      regimeB !== "normal"
    ) {
      notify(`Bot B regime: ${regimeB.toUpperCase()}`);
    }
    if (regimeA) setPrevRegimeA(regimeA);
    if (regimeB) setPrevRegimeB(regimeB);

    setStatusA(sA);
    setStatusB(sB);
    setHealthA(hA != null);
    setHealthB(hB != null);

    // Accumulate live session equity — only during NYSE market hours (9:30–16:00 ET).
    // Outside market hours equity is stale (Alpaca returns last-trade prices), so
    // appending overnight would produce a long flat tail that pollutes the chart.
    const nowET = new Date(
      new Date().toLocaleString("en-US", { timeZone: "America/New_York" })
    );
    const etMinutes = nowET.getHours() * 60 + nowET.getMinutes();
    const isMarketOpen = etMinutes >= 9 * 60 + 30 && etMinutes < 16 * 60;

    if (isMarketOpen) {
      const eqA = sA?.account?.equity;
      const eqB = sB?.account?.equity;
      if (eqA != null || eqB != null) {
        const now = new Date();
        const time = now.toLocaleTimeString("en-US", {
          timeZone: "America/New_York",
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
        });
        const spyDd = sA?.regime?.spy_drawdown ?? sB?.regime?.spy_drawdown ?? null;
        const posA  = sA?.positions_count ?? null;
        const posB  = sB?.positions_count ?? null;
        setSessionPoints((prev) =>
          [...prev, {
            time,
            a:      eqA != null ? +eqA.toFixed(2) : null,
            b:      eqB != null ? +eqB.toFixed(2) : null,
            spy_dd: spyDd != null ? +spyDd.toFixed(6) : null,
            pos_a:  posA,
            pos_b:  posB,
          }].slice(-10_000)
        );
        // Persist to Postgres on Bot B VPS (fire-and-forget — never blocks the UI)
        storeSessionPoint(
          eqA != null ? +eqA.toFixed(2) : null,
          eqB != null ? +eqB.toFixed(2) : null,
          now.toISOString(),
          spyDd != null ? +spyDd.toFixed(6) : null,
          posA,
          posB,
        );
      }
    }
  }, [prevRegimeA, prevRegimeB]);

  const loadBot = React.useCallback(async (b: BotId) => {
    setLoading(true);
    const [s, p, r, t, d, e, l, h] = await Promise.all([
      fetchStatus(b),
      fetchPositions(b),
      fetchRisk(b),
      fetchTca(b),
      fetchDrift(b),
      fetchEquity(b),
      fetchLogs(b, 80),
      fetchHealth(b),
    ]);
    setStatus(s);
    setPositions(Array.isArray(p) ? p : []);
    setRisk(r);
    setTca(t);
    setDrift(d);
    setEquity(Array.isArray(e) ? e : []);
    setLogs(l);
    setHealthy(h != null);
    setLoading(false);
    setLastRefresh(new Date());
    setRefreshIn(30);
  }, []);

  const loadEquityCurves = React.useCallback(async () => {
    const [eA, eB] = await Promise.all([
      fetchEquity("bot-a"),
      fetchEquity("bot-b"),
    ]);
    setEquityA(eA);
    setEquityB(eB);
  }, []);

  // Keyboard shortcuts
  React.useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      switch (e.key) {
        case "1":
          setBot("bot-a");
          break;
        case "2":
          setBot("bot-b");
          break;
        case "r":
        case "R":
          loadComparison();
          loadBot(bot);
          break;
        case " ":
          e.preventDefault();
          setPaused((p) => !p);
          break;
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [bot, loadComparison, loadBot]);

  React.useEffect(() => {
    loadComparison();
    loadBot(bot);
    loadEquityCurves();
    const interval = setInterval(() => {
      if (!paused) {
        loadComparison();
        loadBot(bot);
        loadEquityCurves();
      }
    }, 30000);
    return () => clearInterval(interval);
  }, [bot, paused, loadBot, loadComparison, loadEquityCurves]);

  const switchBot = (b: string) => {
    if (b === "bot-a" || b === "bot-b") setBot(b as BotId);
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className="border-b border-border px-6 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1.5">
              <SparklesIcon size={14} />
              <h1 className="text-sm font-semibold tracking-tight">SIGNUM</h1>
            </div>
            <Separator orientation="vertical" className="h-4" />
            <Badge
              variant={
                sessionColor(market.session) as
                  | "outline"
                  | "secondary"
                  | "destructive"
              }
            >
              {market.session}
            </Badge>
            <span className="text-xs tabular-nums text-muted-foreground">
              {market.session === "Open"
                ? `Closes in ${market.countdown}`
                : `Opens in ${market.countdown}`}
            </span>
          </div>
          <div className="flex items-center gap-4">
            <Tabs value={bot} onValueChange={switchBot}>
              <TabsList variant="line">
                <TabsTrigger value="bot-a">Bot A</TabsTrigger>
                <TabsTrigger value="bot-b">Bot B</TabsTrigger>
              </TabsList>
            </Tabs>
            <Separator orientation="vertical" className="h-4" />
            <div className="flex items-center gap-3 text-[10px] tabular-nums text-muted-foreground">
              <ClockIcon size={12} />
              <span>{fmtTz(now, "America/New_York", "NY")}</span>
              <span>{fmtTz(now, "Asia/Kolkata", "IST")}</span>
              <span>{fmtTz(now, "UTC", "UTC")}</span>
            </div>
            <Separator orientation="vertical" className="h-4" />
            <div className="flex items-center gap-1 text-[10px] tabular-nums text-muted-foreground">
              {paused ? <PauseIcon size={12} /> : <RefreshCWIcon size={12} />}
              <span>{paused ? "paused" : `${refreshIn}s`}</span>
            </div>
          </div>
        </div>
      </header>

      <main className="space-y-4 p-6">
        {/* ── Comparison Strip ──────────────────────────────────── */}
        <div className="grid grid-cols-2 gap-4">
          <ComparisonCard
            label="Bot A"
            sublabel="LightGBM"
            status={statusA}
            healthy={healthA}
            active={bot === "bot-a"}
            onClick={() => setBot("bot-a")}
          />
          <ComparisonCard
            label="Bot B"
            sublabel="Ensemble"
            status={statusB}
            healthy={healthB}
            active={bot === "bot-b"}
            onClick={() => setBot("bot-b")}
          />
        </div>

        {/* ── Hero Metric Cards ────────────────────────────────── */}
        {loading && !status ? (
          <div className="grid grid-cols-4 gap-4">
            <Card className="col-span-2">
              <CardHeader>
                <Skeleton className="h-3 w-24" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-6 w-40" />
              </CardContent>
            </Card>
            {[...Array(2)].map((_, i) => (
              <Card key={i}>
                <CardHeader>
                  <Skeleton className="h-3 w-24" />
                </CardHeader>
                <CardContent>
                  <Skeleton className="h-6 w-32" />
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-4 gap-4">
            <Card className="col-span-2">
              <CardHeader>
                <CardDescription className="flex items-center gap-1.5">
                  <DollarSignIcon size={14} />
                  Portfolio Equity
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex items-baseline justify-between">
                  <CardTitle className="text-2xl font-semibold tabular-nums">
                    {fmtUsd(status?.account?.equity)}
                  </CardTitle>
                  <div className="text-right text-xs text-muted-foreground">
                    <p>Cash: {fmtUsd(status?.account?.cash)}</p>
                    <p>Buying Power: {fmtUsd(status?.account?.buying_power)}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardDescription className="flex items-center gap-1.5">
                  <GaugeIcon size={14} />
                  Market Regime
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-2">
                  <Badge
                    variant={
                      regimeColor(status?.regime?.regime) as
                        | "outline"
                        | "secondary"
                        | "destructive"
                    }
                  >
                    {status?.regime?.regime?.toUpperCase() || "—"}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    Exposure: {fmtPct(status?.regime?.exposure_multiplier)}
                  </span>
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  VIX: {fmt(status?.regime?.vix, 1)} | SPY DD:{" "}
                  {fmtPct(status?.regime?.spy_drawdown)}
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardDescription className="flex items-center gap-1.5">
                  <ActivityIcon size={14} />
                  Bot State
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-2">
                  <span
                    className={`inline-block h-2 w-2 rounded-full ${
                      healthy ? "bg-green-500" : "bg-red-500"
                    }`}
                  />
                  <CardTitle className="text-sm">
                    {healthy ? "Online" : "Offline"}
                  </CardTitle>
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  Positions: {status?.positions_count ?? 0} | Last:{" "}
                  {fmtDate(status?.bot_state?.last_shutdown)}
                </p>
                <p className="text-xs text-muted-foreground">
                  Reason: {status?.bot_state?.reason || "—"}
                </p>
              </CardContent>
            </Card>
          </div>
        )}

        {/* ── Equity Chart + Risk ──────────────────────────────── */}
        <div className="grid grid-cols-4 gap-4">
          <Card className="col-span-3">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-1.5">
                    <ChartLineIcon size={16} />
                    Equity Curve
                  </CardTitle>
                  <CardDescription>
                    Bot A vs Bot B — portfolio value over time
                  </CardDescription>
                </div>
                {paused && (
                  <Badge variant="secondary" className="flex items-center gap-1 text-[10px]">
                    <PauseIcon size={10} />
                    PAUSED
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {equityA.length > 0 || equityB.length > 0 ? (
                <DualEquityChart dataA={equityA} dataB={equityB} />
              ) : (
                <div className="flex h-72 items-center justify-center text-xs text-muted-foreground">
                  No equity data yet — first trade on March 4th
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-1.5">
                <ShieldCheckIcon size={16} />
                Risk Metrics
              </CardTitle>
            </CardHeader>
            <CardContent>
              <dl className="space-y-3 text-xs">
                <MetricRow
                  label="Sharpe Ratio"
                  value={fmt(risk?.sharpe_ratio)}
                />
                <MetricRow
                  label="Sortino Ratio"
                  value={fmt(risk?.sortino_ratio)}
                />
                <MetricRow
                  label="Max Drawdown"
                  value={fmtPct(risk?.max_drawdown)}
                />
                <MetricRow
                  label="Current DD"
                  value={fmtPct(risk?.current_drawdown)}
                />
                <MetricRow label="VaR 95%" value={fmtPct(risk?.var_95)} />
                <MetricRow label="CVaR 95%" value={fmtPct(risk?.cvar_95)} />
                <MetricRow label="Win Rate" value={fmtPct(risk?.win_rate)} />
                <MetricRow
                  label="Total Trades"
                  value={String(risk?.total_trades ?? "—")}
                />
              </dl>
            </CardContent>
          </Card>
        </div>

        {/* ── Live Session P&L ─────────────────────────────────── */}
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
                  {sessionPoints.length > 0 && ` · ${sessionPoints.length} pts`}
                </CardDescription>
              </div>
              <div className="flex items-center gap-2">
                {sessionPoints.length > 0 && (
                  <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
                    <span className="inline-block h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
                    LIVE
                  </span>
                )}
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <LiveSessionChart data={sessionPoints} onClear={() => setSessionPoints([])} />
          </CardContent>
        </Card>

        {/* ── Sector Exposure ──────────────────────────────────── */}
        {positions.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-1.5">
                <ChartBarIncreasingIcon size={16} />
                Sector Exposure
              </CardTitle>
              <CardDescription>Portfolio weight by sector</CardDescription>
            </CardHeader>
            <CardContent>
              <SectorBreakdown positions={positions} />
            </CardContent>
          </Card>
        )}

        {/* ── Positions Table ──────────────────────────────────── */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-1.5">
              <LayersIcon size={16} />
              Open Positions
            </CardTitle>
            <CardDescription>
              {positions.length} position{positions.length !== 1 ? "s" : ""}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {positions.length > 0 ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Symbol</TableHead>
                    <TableHead className="text-right">Qty</TableHead>
                    <TableHead className="text-right">Avg Entry</TableHead>
                    <TableHead className="text-right">Current</TableHead>
                    <TableHead className="text-right">Market Value</TableHead>
                    <TableHead className="text-right">P&L</TableHead>
                    <TableHead className="text-right">P&L %</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {positions.map((p) => (
                    <TableRow key={p.symbol}>
                      <TableCell className="font-medium">{p.symbol}</TableCell>
                      <TableCell className="text-right tabular-nums">
                        {p.qty}
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        {fmtUsd(p.avg_entry_price)}
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        {fmtUsd(p.current_price)}
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        {fmtUsd(p.market_value)}
                      </TableCell>
                      <TableCell
                        className={`text-right tabular-nums ${
                          (p.unrealized_pl ?? 0) >= 0
                            ? "text-green-500"
                            : "text-red-500"
                        }`}
                      >
                        {fmtUsd(p.unrealized_pl)}
                      </TableCell>
                      <TableCell
                        className={`text-right tabular-nums ${
                          (p.unrealized_plpc ?? 0) >= 0
                            ? "text-green-500"
                            : "text-red-500"
                        }`}
                      >
                        {fmtPct(p.unrealized_plpc)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
                <TableFooter>
                  <TableRow>
                    <TableCell className="font-medium">Total</TableCell>
                    <TableCell className="text-right tabular-nums">
                      {positions.reduce((s, p) => s + (p.qty ?? 0), 0)}
                    </TableCell>
                    <TableCell />
                    <TableCell />
                    <TableCell className="text-right tabular-nums font-medium">
                      {fmtUsd(
                        positions.reduce(
                          (s, p) => s + (p.market_value ?? 0),
                          0
                        )
                      )}
                    </TableCell>
                    <TableCell
                      className={`text-right tabular-nums font-medium ${
                        positions.reduce(
                          (s, p) => s + (p.unrealized_pl ?? 0),
                          0
                        ) >= 0
                          ? "text-green-500"
                          : "text-red-500"
                      }`}
                    >
                      {fmtUsd(
                        positions.reduce(
                          (s, p) => s + (p.unrealized_pl ?? 0),
                          0
                        )
                      )}
                    </TableCell>
                    <TableCell />
                  </TableRow>
                </TableFooter>
              </Table>
            ) : (
              <p className="py-8 text-center text-xs text-muted-foreground">
                No open positions
              </p>
            )}
          </CardContent>
        </Card>

        {/* ── Logs + TCA + Drift ───────────────────────────────── */}
        <div className="grid grid-cols-4 gap-4">
          <Card className="col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-1.5">
                <TerminalIcon size={16} />
                Logs
              </CardTitle>
              <CardDescription>Recent bot output</CardDescription>
            </CardHeader>
            <CardContent className="p-0">
              <ScrollArea className="h-72">
                <pre className="p-4 text-[10px] leading-relaxed text-muted-foreground">
                  {logs || "No logs available"}
                </pre>
              </ScrollArea>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-1.5">
                <DollarSignIcon size={16} />
                TCA
              </CardTitle>
              <CardDescription>Transaction cost analysis</CardDescription>
            </CardHeader>
            <CardContent>
              <dl className="space-y-3 text-xs">
                <MetricRow
                  label="Avg IS (bps)"
                  value={fmt(tca?.avg_implementation_shortfall_bps, 1)}
                />
                <MetricRow
                  label="Fill Rate"
                  value={fmtPct(tca?.avg_fill_rate)}
                />
                <MetricRow
                  label="Total Trades"
                  value={String(tca?.total_trades ?? "—")}
                />
              </dl>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-1.5">
                <WavesIcon size={16} />
                Feature Drift
              </CardTitle>
              <CardDescription>KS test + PSI monitoring</CardDescription>
            </CardHeader>
            <CardContent>
              <dl className="space-y-3 text-xs">
                <MetricRow
                  label="Drifted"
                  value={`${drift?.drift_count ?? "—"} / ${drift?.total_features ?? "—"}`}
                />
              </dl>
              {drift?.drifted_features &&
                drift.drifted_features.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {drift.drifted_features.map((f) => (
                      <Badge
                        key={f}
                        variant="secondary"
                        className="text-[10px]"
                      >
                        {f}
                      </Badge>
                    ))}
                  </div>
                )}
            </CardContent>
          </Card>
        </div>

        {/* ── Footer ───────────────────────────────────────────── */}
        <footer className="border-t border-border pt-4 pb-8 text-center text-xs text-muted-foreground">
          Signum — Paper Trading A/B Comparison | Bot A: LightGBM | Bot B:
          Ensemble (LightGBM + CatBoost + RF + Ridge) | Auto-refresh 30s
        </footer>
      </main>
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────

function ComparisonCard({
  label,
  sublabel,
  status,
  healthy,
  active,
  onClick,
}: {
  label: string;
  sublabel: string;
  status: StatusData | null;
  healthy: boolean | null;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <Card
      className={`cursor-pointer transition-colors ${
        active ? "ring-1 ring-foreground/30" : "opacity-60 hover:opacity-80"
      }`}
      onClick={onClick}
    >
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span
              className={`inline-block h-2 w-2 rounded-full ${
                healthy
                  ? "bg-green-500"
                  : healthy === false
                    ? "bg-red-500"
                    : "bg-muted"
              }`}
            />
            <BotIcon size={16} />
            <CardTitle>{label}</CardTitle>
            <Badge variant="outline" className="text-[10px]">
              {sublabel}
            </Badge>
          </div>
          <Badge
            variant={
              regimeColor(status?.regime?.regime) as
                | "outline"
                | "secondary"
                | "destructive"
            }
          >
            {status?.regime?.regime?.toUpperCase() || "—"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex items-baseline justify-between">
          <div>
            <p className="text-lg font-semibold tabular-nums">
              {fmtUsd(status?.account?.equity)}
            </p>
            <div className="flex items-center gap-2">
              <p className="text-xs text-muted-foreground">
                {status?.positions_count ?? 0} positions
              </p>
              {status?.account?.equity != null && (
                <span
                  className={`flex items-center gap-1 text-xs font-medium tabular-nums ${
                    status.account.equity >= STARTING_EQUITY
                      ? "text-green-500"
                      : "text-red-500"
                  }`}
                >
                  {status.account.equity >= STARTING_EQUITY
                    ? <TrendingUpIcon size={12} />
                    : <TrendingDownIcon size={12} />
                  }
                  {status.account.equity >= STARTING_EQUITY ? "+" : ""}
                  {fmtUsd(status.account.equity - STARTING_EQUITY)} (
                  {status.account.equity >= STARTING_EQUITY ? "+" : ""}
                  {fmt(
                    ((status.account.equity - STARTING_EQUITY) /
                      STARTING_EQUITY) *
                      100
                  )}
                  %)
                </span>
              )}
            </div>
          </div>
          <div className="text-right text-xs text-muted-foreground">
            <p>VIX: {fmt(status?.regime?.vix, 1)}</p>
            <p>SPY DD: {fmtPct(status?.regime?.spy_drawdown)}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-medium tabular-nums">{value}</dd>
    </div>
  );
}

const dualEquityChartConfig = {
  botA: { label: "Bot A", color: "hsl(160 60% 50%)" },
  botB: { label: "Bot B", color: "hsl(213 80% 58%)" },
} satisfies ChartConfig;

type EquityDisplayMode = "abs" | "pct";

// Return all Wednesday dates (YYYY-MM-DD) present in the data set.
// Uses the actual date keys so ReferenceLine x values always match.
function extractWednesdays(dates: string[]): string[] {
  return dates.filter((d) => new Date(d + "T12:00:00Z").getUTCDay() === 3);
}

function DualEquityChart({
  dataA,
  dataB,
}: {
  dataA: EquityPoint[];
  dataB: EquityPoint[];
}) {
  const [equityMode, setEquityMode] = React.useState<EquityDisplayMode>("abs");

  // Merge both series by date (last value per day wins)
  const dateMap = new Map<string, { date: string; botA?: number; botB?: number }>();
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
  const merged = Array.from(dateMap.values()).sort((a, b) =>
    a.date.localeCompare(b.date)
  );

  // Insufficient history — need ≥3 unique trading days to draw a useful curve
  if (merged.length < 3) {
    return (
      <div className="flex h-72 flex-col items-center justify-center gap-2">
        <p className="text-xs text-muted-foreground">
          Building history… {merged.length}/3 days
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
    );
  }

  const toDisplay = (equity: number) =>
    equityMode === "abs"
      ? equity
      : +((equity - STARTING_EQUITY) / STARTING_EQUITY * 100).toFixed(3);

  const chartData = merged.map((pt) => ({
    date: pt.date,
    botA: pt.botA != null ? toDisplay(pt.botA) : undefined,
    botB: pt.botB != null ? toDisplay(pt.botB) : undefined,
  }));

  const wednesdays = extractWednesdays(merged.map((d) => d.date));

  const fmtY = (v: number) =>
    equityMode === "abs"
      ? `$${(v / 1000).toFixed(1)}k`
      : `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;

  const yDomain = React.useMemo((): [number, number] => {
    const vals = chartData
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
  }, [chartData, equityMode]);

  // Last-value pinned dot+label rendered only at the final data index
  const makeDot =
    (color: string) =>
    (dotProps: unknown) => {
      const { cx, cy, index, value } = dotProps as {
        cx: number;
        cy: number;
        index: number;
        value?: number;
      };
      if (index !== chartData.length - 1 || value == null)
        return <g key={index} />;
      return (
        <g key={index}>
          <circle cx={cx} cy={cy} r={2.5} fill={color} />
          <text
            x={cx + 5}
            y={cy}
            fontSize={9}
            fill={color}
            dominantBaseline="middle"
          >
            {fmtY(value)}
          </text>
        </g>
      );
    };

  return (
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
              style={{ backgroundColor: "hsl(160 60% 50%)" }}
            />
            Bot A
          </span>
          <span className="flex items-center gap-1.5">
            <span
              className="inline-block h-0.5 w-4 rounded-full"
              style={{
                backgroundColor: "hsl(213 80% 58%)",
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
                  borderColor: "hsl(38 70% 50%)",
                  borderStyle: "dashed",
                }}
              />
              Rebalance
            </span>
          )}
        </div>
      </div>

      <ChartContainer config={dualEquityChartConfig} className="h-72 w-full">
        <ComposedChart data={chartData} margin={{ top: 4, right: 48, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="gradientA" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="hsl(160 60% 50%)" stopOpacity={0.15} />
              <stop offset="100%" stopColor="hsl(160 60% 50%)" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="gradientB" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="hsl(213 80% 58%)" stopOpacity={0.10} />
              <stop offset="100%" stopColor="hsl(213 80% 58%)" stopOpacity={0} />
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
          {/* Wednesday rebalance markers */}
          {wednesdays.map((d) => (
            <ReferenceLine
              key={d}
              x={d}
              stroke="hsl(38 70% 50%)"
              strokeOpacity={0.4}
              strokeDasharray="2 4"
              strokeWidth={1}
            />
          ))}
          <ChartTooltip
            content={
              <ChartTooltipContent
                formatter={(value, name) => (
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
            stroke="hsl(160 60% 50%)"
            strokeWidth={1.5}
            fill="url(#gradientA)"
            connectNulls
            dot={makeDot("hsl(160 60% 50%)")}
            isAnimationActive={false}
          />
          <Area
            type="monotone"
            dataKey="botB"
            stroke="hsl(213 80% 58%)"
            strokeWidth={1.5}
            strokeDasharray="4 3"
            fill="url(#gradientB)"
            connectNulls
            dot={makeDot("hsl(213 80% 58%)")}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ChartContainer>
    </div>
  );
}

// ── Live Session Chart ──────────────────────────────────────────────────

type SessionMode = "pnl" | "spread";
type WindowKey = "30m" | "1h" | "2h" | "4h" | "all";

const WINDOW_LIMITS: Record<WindowKey, number> = {
  "30m": 60,
  "1h":  120,
  "2h":  240,
  "4h":  480,
  "all": Infinity,
};

const MA_WINDOW = 5;

const liveChartConfig = {
  a:         { label: "Bot A",  color: "hsl(160 60% 50%)" },
  b:         { label: "Bot B",  color: "hsl(213 80% 58%)" },
  ma_a:      { label: "MA A",   color: "hsl(160 60% 50%)" },
  ma_b:      { label: "MA B",   color: "hsl(213 80% 58%)" },
  ma_spread: { label: "MA B−A", color: "hsl(38 90% 58%)"  },
} satisfies ChartConfig;

// Rolling mean — requires exactly n non-null values in the window.
function rollingMean(vals: (number | null)[], n: number): (number | null)[] {
  return vals.map((_, i) => {
    if (i < n - 1) return null;
    const slice = vals.slice(i - n + 1, i + 1);
    if (slice.some((v) => v == null)) return null;
    return (slice as number[]).reduce((a, b) => a + b, 0) / n;
  });
}

// Vertical dashed crosshair line rendered via the Recharts `cursor` prop.
function ChartCrosshair({
  points,
  height = 192,
}: {
  points?: Array<{ x: number; y: number }>;
  height?: number;
}) {
  if (!points?.[0]) return null;
  return (
    <line
      x1={points[0].x}
      y1={0}
      x2={points[0].x}
      y2={height}
      stroke="rgba(255,255,255,0.10)"
      strokeWidth={1}
      strokeDasharray="2 3"
    />
  );
}

function LiveSessionChart({
  data,
  onClear,
}: {
  data: Array<{ time: string; a: number | null; b: number | null; spy_dd: number | null; pos_a: number | null; pos_b: number | null }>;
  onClear: () => void;
}) {
  const [mode, setMode]             = React.useState<SessionMode>("pnl");
  const [timeWindow, setTimeWindow] = React.useState<WindowKey>("all");
  const [showMA, setShowMA]         = React.useState(false);
  const [paused, setPaused]         = React.useState(false);
  const [frozenData, setFrozenData] = React.useState<typeof data>([]);

  // ── History mode ──────────────────────────────────────────────────────
  const [historyMode, setHistoryMode]         = React.useState(false);
  const [dateRange, setDateRange]             = React.useState<DateRange | undefined>();
  const [historyData, setHistoryData]         = React.useState<SessionPoint[]>([]);
  const [historyLoading, setHistoryLoading]   = React.useState(false);
  const [calendarOpen, setCalendarOpen]       = React.useState(false);

  const fmtDateLabel = (d: Date) =>
    d.toLocaleDateString("en-US", { month: "short", day: "numeric" });

  const toYMD = (d: Date) => d.toISOString().slice(0, 10);

  const loadHistory = React.useCallback(async (range: DateRange) => {
    if (!range.from) return;
    const from = toYMD(range.from);
    const to   = toYMD(range.to ?? range.from);
    setHistoryLoading(true);
    const res = await fetchSessionHistory(from, to);
    setHistoryData(res?.history ?? []);
    setHistoryLoading(false);
  }, []);

  // Convert Postgres SessionPoint rows → chart-compatible format
  const historyAsPoints = React.useMemo(
    () =>
      historyData.map((p) => ({
        time:   p.ts.slice(11, 19), // HH:MM:SS from ISO string
        a:      p.equity_a,
        b:      p.equity_b,
        spy_dd: p.spy_dd ?? null,
        pos_a:  p.pos_a ?? null,
        pos_b:  p.pos_b ?? null,
      })),
    [historyData],
  );

  const handleTogglePause = () => {
    if (!paused) setFrozenData(data); // snapshot at the moment of pause
    setPaused((p) => !p);
  };

  // When unpaused, keep frozen snapshot in sync so resume is seamless.
  React.useEffect(() => {
    if (!paused) setFrozenData(data);
  }, [data, paused]);

  // In history mode use the DB rows; otherwise use the live in-memory buffer.
  const displayData = historyMode
    ? historyAsPoints
    : paused
      ? frozenData
      : data;

  // Slice to active time window (history mode always shows full range).
  const windowedData = React.useMemo(() => {
    if (historyMode) return displayData;
    const limit = WINDOW_LIMITS[timeWindow];
    return isFinite(limit) ? displayData.slice(-limit) : displayData;
  }, [displayData, timeWindow, historyMode]);

  // Derive chart data. ALL keys always present — null when inactive.
  // Stable key count prevents Recharts losing measured dimensions on switch.
  const chartData = React.useMemo(() => {
    if (windowedData.length === 0) return [];
    const pts = windowedData.map((pt) => {
      const pnlA = pt.a != null ? +(pt.a - STARTING_EQUITY).toFixed(2) : null;
      const pnlB = pt.b != null ? +(pt.b - STARTING_EQUITY).toFixed(2) : null;
      const sp   = pt.a != null && pt.b != null ? +(pt.b - pt.a).toFixed(2) : null;
      return {
        time:   pt.time,
        a:      mode === "pnl"    ? pnlA : null,
        b:      mode === "pnl"    ? pnlB : null,
        spread: mode === "spread" ? sp   : null,
      };
    });
    const maA = rollingMean(pts.map((p) => p.a),      MA_WINDOW);
    const maB = rollingMean(pts.map((p) => p.b),      MA_WINDOW);
    const maS = rollingMean(pts.map((p) => p.spread), MA_WINDOW);
    return pts.map((p, i) => ({
      ...p,
      ma_a:      showMA ? maA[i] : null,
      ma_b:      showMA ? maB[i] : null,
      ma_spread: showMA ? maS[i] : null,
    }));
  }, [windowedData, mode, showMA]);

  // Profitability-aware fill colours.
  const lastPt = chartData.length > 0 ? chartData[chartData.length - 1] : null;
  const aFill  = (lastPt?.a  ?? 0) >= 0 ? "hsl(142 55% 40%)" : "hsl(0 62% 52%)";
  const bFill  = (lastPt?.b  ?? 0) >= 0 ? "hsl(213 70% 52%)" : "hsl(20 70% 50%)";
  const sFill  = ((lastPt as { spread?: number | null } | null)?.spread ?? 0) >= 0
    ? "hsl(38 85% 52%)" : "hsl(0 62% 52%)";

  // Open / High / Low / Now stat strip.
  const stats = React.useMemo(() => {
    type OHLC = { open: number; high: number; low: number; now: number };
    const summarize = (vals: number[]): OHLC | null =>
      vals.length === 0 ? null : {
        open: vals[0],
        high: Math.max(...vals),
        low:  Math.min(...vals),
        now:  vals[vals.length - 1],
      };
    const aVals = chartData.map((d) => d.a).filter((v): v is number => v != null);
    const bVals = chartData.map((d) => d.b).filter((v): v is number => v != null);
    const sVals = chartData
      .map((d) => (d as { spread?: number | null }).spread)
      .filter((v): v is number => v != null);
    return { a: summarize(aVals), b: summarize(bVals), spread: summarize(sVals) };
  }, [chartData]);

  // Session H/L for reference lines.
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

  // Tight Y domain.
  const yDomain = React.useMemo((): [number, number] => {
    let vals: number[];
    if (mode === "spread") {
      vals = chartData
        .map((pt) => (pt as { spread?: number | null }).spread)
        .filter((v): v is number => v != null);
    } else {
      vals = chartData.flatMap((pt) => [pt.a, pt.b]).filter((v): v is number => v != null);
    }
    if (vals.length === 0) return mode === "spread" ? [-50, 50] : [-100, 100];
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const range = hi - lo;
    const minSpread = mode === "pnl" ? 20 : 10;
    const pad = Math.max(range * 0.15, minSpread / 2);
    return [lo - pad, hi + pad];
  }, [chartData, mode]);

  // Drawdown from session peak (for OHLC strip annotation)
  const sessionDdPct = React.useMemo(() => {
    type DD = { a: number | null; b: number | null };
    const result: DD = { a: null, b: null };
    if (chartData.length < 2) return result;
    const aVals = chartData.map((d) => d.a).filter((v): v is number => v != null);
    const bVals = chartData.map((d) => d.b).filter((v): v is number => v != null);
    const calcDd = (vals: number[]) => {
      if (vals.length === 0) return null;
      const peak = Math.max(...vals);
      const now  = vals[vals.length - 1];
      if (peak === 0) return null;
      return ((now - peak) / Math.abs(peak)) * 100;
    };
    result.a = calcDd(aVals);
    result.b = calcDd(bVals);
    return result;
  }, [chartData]);

  // Significant single-step P&L jump markers (|Δ| > $50)
  const jumpDots = React.useMemo(() => {
    const threshold = 50;
    return chartData.flatMap((pt, i) => {
      if (i === 0) return [];
      const prev = chartData[i - 1];
      const dots: Array<{ time: string; val: number; color: string }> = [];
      if (pt.a != null && prev.a != null && Math.abs(pt.a - prev.a) > threshold)
        dots.push({ time: pt.time, val: pt.a, color: "hsl(160 60% 50%)" });
      if (pt.b != null && prev.b != null && Math.abs(pt.b - prev.b) > threshold)
        dots.push({ time: pt.time, val: pt.b, color: "hsl(213 80% 58%)" });
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
      <div className="flex h-48 flex-col items-center justify-center gap-2">
        {/* Keep history controls visible in empty state */}
        {historyMode && (
          <div className="flex items-center gap-2 mb-2">
            <button onClick={() => { setHistoryMode(false); setDateRange(undefined); setHistoryData([]); }} className={pillBtn(false)}>
              ← Live
            </button>
            <Popover open={calendarOpen} onOpenChange={setCalendarOpen}>
              <PopoverTrigger asChild>
                <button className="rounded border border-border px-2 py-0.5 text-[11px] text-muted-foreground hover:text-foreground transition-colors">
                  {dateRange?.from ? fmtDateLabel(dateRange.from) : "Pick dates"}
                </button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0" align="start">
                <Calendar mode="range" selected={dateRange}
                  onSelect={(range) => { setDateRange(range); if (range?.from) { loadHistory({ from: range.from, to: range.to ?? range.from }); setCalendarOpen(false); } }}
                  disabled={{ after: new Date() }} numberOfMonths={2} />
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
    );
  }

  const fmtPnl = (v: number) => `${v >= 0 ? "+" : ""}$${v.toFixed(0)}`;

  return (
    <div className="space-y-3">
      {/* Control bar */}
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-2 flex-wrap">
          {/* Mode */}
          <div className="flex items-center gap-1 rounded-md border border-border p-0.5">
            {(["pnl", "spread"] as SessionMode[]).map((m) => (
              <button key={m} onClick={() => setMode(m)} className={pillBtn(mode === m)}>
                {m === "pnl" ? "P&L Δ" : "B−A"}
              </button>
            ))}
          </div>

          {/* Time window — hidden in history mode (range controls the window) */}
          {!historyMode && (
            <div className="flex items-center gap-1 rounded-md border border-border p-0.5">
              {(["30m", "1h", "2h", "4h", "all"] as WindowKey[]).map((w) => (
                <button key={w} onClick={() => setTimeWindow(w)} className={pillBtn(timeWindow === w)}>
                  {w === "all" ? "All" : w}
                </button>
              ))}
            </div>
          )}

          {/* History mode toggle + date range picker */}
          <div className="flex items-center gap-1">
            <button
              onClick={() => { setHistoryMode((v) => !v); setDateRange(undefined); setHistoryData([]); }}
              title={historyMode ? "Back to live session" : "Browse historical sessions"}
              className={pillBtn(historyMode)}
            >
              History
            </button>
            {historyMode && (
              <Popover open={calendarOpen} onOpenChange={setCalendarOpen}>
                <PopoverTrigger asChild>
                  <button className="rounded border border-border px-2 py-0.5 text-[11px] text-muted-foreground hover:text-foreground transition-colors">
                    {dateRange?.from
                      ? dateRange.to && dateRange.to.getTime() !== dateRange.from.getTime()
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
                        loadHistory({ from: range.from, to: range.to ?? range.from });
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
              showMA ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground"
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
            <span className="text-[10px] text-muted-foreground animate-pulse">Loading…</span>
          )}
          {historyMode && !historyLoading && historyData.length > 0 && (
            <span className="text-[10px] text-muted-foreground tabular-nums">
              {historyData.length.toLocaleString()} pts
            </span>
          )}
        </div>
      </div>

      {/* OHLC stat strip */}
      {chartData.length >= 2 && (() => {
        const fmtStat = (v: number) => `${v >= 0 ? "+" : ""}$${Math.abs(v).toFixed(0)}`;
        if (mode === "spread") {
          const s = stats.spread;
          if (!s) return null;
          return (
            <div className="flex items-center gap-3 text-[10px] text-muted-foreground tabular-nums">
              <span className="font-medium" style={{ color: "hsl(38 90% 58%)" }}>B−A</span>
              <span>O <span className="text-foreground">{fmtStat(s.open)}</span></span>
              <span>H <span className="text-green-500">{fmtStat(s.high)}</span></span>
              <span>L <span className="text-red-500">{fmtStat(s.low)}</span></span>
              <span>Now <span className={s.now >= 0 ? "text-green-500" : "text-red-500"}>{fmtStat(s.now)}</span></span>
            </div>
          );
        }
        return (
          <div className="flex items-center gap-4 text-[10px] text-muted-foreground tabular-nums flex-wrap">
            {stats.a && (
              <span className="flex items-center gap-2">
                <span className="font-medium" style={{ color: "hsl(160 60% 50%)" }}>A</span>
                <span>O <span className="text-foreground">{fmtStat(stats.a.open)}</span></span>
                <span>H <span className="text-green-500">{fmtStat(stats.a.high)}</span></span>
                <span>L <span className="text-red-500">{fmtStat(stats.a.low)}</span></span>
                <span>
                  Now <span className={stats.a.now >= 0 ? "text-green-500" : "text-red-500"}>{fmtStat(stats.a.now)}</span>
                  {sessionDdPct.a != null && Math.abs(sessionDdPct.a) > 0.01 && (
                    <span className="ml-1 text-muted-foreground">
                      ({sessionDdPct.a > 0 ? "+" : ""}{sessionDdPct.a.toFixed(2)}%)
                    </span>
                  )}
                </span>
              </span>
            )}
            {stats.b && (
              <span className="flex items-center gap-2">
                <span className="font-medium" style={{ color: "hsl(213 80% 58%)" }}>B</span>
                <span>O <span className="text-foreground">{fmtStat(stats.b.open)}</span></span>
                <span>H <span className="text-green-500">{fmtStat(stats.b.high)}</span></span>
                <span>L <span className="text-red-500">{fmtStat(stats.b.low)}</span></span>
                <span>
                  Now <span className={stats.b.now >= 0 ? "text-green-500" : "text-red-500"}>{fmtStat(stats.b.now)}</span>
                  {sessionDdPct.b != null && Math.abs(sessionDdPct.b) > 0.01 && (
                    <span className="ml-1 text-muted-foreground">
                      ({sessionDdPct.b > 0 ? "+" : ""}{sessionDdPct.b.toFixed(2)}%)
                    </span>
                  )}
                </span>
              </span>
            )}
          </div>
        );
      })()}

      {/* Legend */}
      <div className="flex items-center gap-4 text-[11px] text-muted-foreground">
        {mode !== "spread" ? (
          <>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-4 rounded-full" style={{ backgroundColor: "hsl(160 60% 50%)" }} />
              Bot A
            </span>
            <span className="flex items-center gap-1.5">
              <span
                className="inline-block h-0.5 w-4 rounded-full"
                style={{
                  backgroundColor: "hsl(213 80% 58%)",
                  backgroundImage: "repeating-linear-gradient(90deg, hsl(213 80% 58%) 0 4px, transparent 4px 7px)",
                }}
              />
              Bot B
            </span>
          </>
        ) : (
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-0.5 w-4 rounded-full" style={{ backgroundColor: "hsl(38 90% 58%)" }} />
            Bot B − Bot A (positive = B winning)
          </span>
        )}
        {paused && (
          <span className="text-yellow-400 text-[10px]">⏸ paused</span>
        )}
        {chartData.length > 0 && (() => {
          const last = chartData[chartData.length - 1];
          if (mode === "spread") {
            const s = (last as { spread?: number | null }).spread;
            if (s == null) return null;
            return (
              <span className={`ml-auto tabular-nums font-medium ${s >= 0 ? "text-green-500" : "text-red-500"}`}>
                {s >= 0 ? "+" : ""}${s.toFixed(2)}
              </span>
            );
          }
          const aVal = last.a;
          const bVal = last.b;
          return (
            <>
              {aVal != null && (
                <span className={`ml-auto tabular-nums font-medium ${aVal >= 0 ? "text-green-500" : "text-red-500"}`}>
                  A: {aVal >= 0 ? "+" : ""}${Math.abs(aVal).toFixed(0)}
                </span>
              )}
              {bVal != null && (
                <span className={`tabular-nums font-medium ${bVal >= 0 ? "text-green-500" : "text-red-500"}`}>
                  B: {bVal >= 0 ? "+" : ""}${Math.abs(bVal).toFixed(0)}
                </span>
              )}
            </>
          );
        })()}
      </div>

      <ChartContainer config={liveChartConfig} className="h-56 w-full">
        <ComposedChart data={chartData} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="liveGradA" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={aFill} stopOpacity={0.20} />
              <stop offset="100%" stopColor={aFill} stopOpacity={0} />
            </linearGradient>
            <linearGradient id="liveGradB" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={bFill} stopOpacity={0.20} />
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

          {/* Zero baseline */}
          <ReferenceLine yAxisId="pnl" y={0} stroke="var(--border)" strokeDasharray="3 3" strokeWidth={1} />
          {/* Session high/low — subtle guides; OHLC strip shows the values */}
          {sessionHigh != null && sessionHigh !== 0 && (
            <ReferenceLine
              yAxisId="pnl"
              y={sessionHigh}
              stroke="hsl(142 55% 40%)"
              strokeDasharray="2 3"
              strokeWidth={1}
              strokeOpacity={0.55}
            />
          )}
          {sessionLow != null && sessionLow !== 0 && (
            <ReferenceLine
              yAxisId="pnl"
              y={sessionLow}
              stroke="hsl(0 62% 52%)"
              strokeDasharray="2 3"
              strokeWidth={1}
              strokeOpacity={0.55}
            />
          )}
          <ChartTooltip
            cursor={<ChartCrosshair />}
            content={
              <ChartTooltipContent
                formatter={(value, name) => {
                  const k = String(name);
                  if (k.startsWith("ma_")) return null;
                  const v = Number(value);
                  if (k === "spread")
                    return <span className="tabular-nums">B−A: {v >= 0 ? "+" : ""}${v.toFixed(2)}</span>;
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
          {/* Area fills — stable key structure; always rendered, null = invisible */}
          <Area yAxisId="pnl" type="monotone" dataKey="a"      stroke="hsl(160 60% 50%)" strokeWidth={1.5} fill="url(#liveGradA)"      connectNulls dot={false} isAnimationActive={false} />
          <Area yAxisId="pnl" type="monotone" dataKey="b"      stroke="hsl(213 80% 58%)" strokeWidth={1.5} fill="url(#liveGradB)"      connectNulls dot={false} isAnimationActive={false} strokeDasharray="4 3" />
          <Area yAxisId="pnl" type="monotone" dataKey="spread" stroke="hsl(38 90% 58%)"  strokeWidth={1.5} fill="url(#liveGradSpread)" connectNulls dot={false} isAnimationActive={false} />
          {/* MA overlay lines — always rendered; null data = invisible */}
          <Line yAxisId="pnl" type="monotone" dataKey="ma_a"      stroke="hsl(160 60% 50%)" strokeWidth={1} strokeOpacity={0.7} strokeDasharray="3 2" dot={false} isAnimationActive={false} connectNulls />
          <Line yAxisId="pnl" type="monotone" dataKey="ma_b"      stroke="hsl(213 80% 58%)" strokeWidth={1} strokeOpacity={0.7} strokeDasharray="3 2" dot={false} isAnimationActive={false} connectNulls />
          <Line yAxisId="pnl" type="monotone" dataKey="ma_spread" stroke="hsl(38 90% 58%)"  strokeWidth={1} strokeOpacity={0.7} strokeDasharray="3 2" dot={false} isAnimationActive={false} connectNulls />

          {/* Significant P&L jump markers */}
          {jumpDots.map((d, i) => (
            <ReferenceDot key={i} yAxisId="pnl" x={d.time} y={d.val} r={3} fill={d.color} stroke="var(--background)" strokeWidth={1} />
          ))}
          {/* Pan/zoom brush — defaults to last 2h (240 × 30s ticks) */}
          <Brush
            dataKey="time"
            height={18}
            startIndex={Math.max(0, chartData.length - 240)}
            travellerWidth={5}
            stroke="var(--border)"
            fill="var(--card)"
            tickFormatter={(v: string) => typeof v === "string" ? v.slice(0, 5) : ""}
          />
        </ComposedChart>
      </ChartContainer>
    </div>
  );
}

// Sector exposure breakdown — horizontal bar using plain divs
function SectorBreakdown({ positions }: { positions: Position[] }) {
  const totalValue = positions.reduce((s, p) => s + Math.abs(p.market_value ?? 0), 0);
  if (totalValue === 0) return null;

  // Group by sector
  const sectorWeights = new Map<string, number>();
  for (const p of positions) {
    const sector = SECTOR_MAP[p.symbol] || "Other";
    sectorWeights.set(
      sector,
      (sectorWeights.get(sector) || 0) + Math.abs(p.market_value ?? 0)
    );
  }

  // Sort descending by weight
  const sorted = Array.from(sectorWeights.entries())
    .map(([sector, value]) => ({ sector, weight: value / totalValue }))
    .sort((a, b) => b.weight - a.weight);

  return (
    <div className="space-y-2">
      {sorted.map(({ sector, weight }) => (
        <div key={sector} className="flex items-center gap-3 text-xs">
          <span className="w-28 shrink-0 text-muted-foreground">{sector}</span>
          <div className="flex-1 h-2 bg-muted overflow-hidden">
            <div
              className="h-full bg-foreground/70"
              style={{ width: `${(weight * 100).toFixed(1)}%` }}
            />
          </div>
          <span className="w-12 text-right tabular-nums font-medium">
            {(weight * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}
