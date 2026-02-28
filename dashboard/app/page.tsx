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
} from "@/lib/api";
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
import { Area, AreaChart, XAxis, YAxis } from "recharts";

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
    setStatusA(sA);
    setStatusB(sB);
    setHealthA(hA != null);
    setHealthB(hB != null);
  }, []);

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
    if (l) {
      setLogs(typeof l === "string" ? l : (l as { logs: string }).logs || "");
    } else {
      setLogs("");
    }
    setHealthy(h != null);
    setLoading(false);
    setLastRefresh(new Date());
  }, []);

  React.useEffect(() => {
    loadComparison();
    loadBot(bot);
    const interval = setInterval(() => {
      loadComparison();
      loadBot(bot);
    }, 30000);
    return () => clearInterval(interval);
  }, [bot, loadBot, loadComparison]);

  const switchBot = (b: string) => {
    if (b === "bot-a" || b === "bot-b") setBot(b as BotId);
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className="border-b border-border px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-sm font-semibold tracking-tight">SIGNUM</h1>
            <Separator orientation="vertical" className="h-4" />
            <span className="text-xs text-muted-foreground">
              Quantitative Equity Trading
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
            <span className="text-xs text-muted-foreground">
              {lastRefresh.toLocaleTimeString()}
            </span>
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
                <CardDescription>Portfolio Equity</CardDescription>
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
                <CardDescription>Market Regime</CardDescription>
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
                <CardDescription>Bot State</CardDescription>
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
              <CardTitle>Equity Curve</CardTitle>
              <CardDescription>Portfolio value over time</CardDescription>
            </CardHeader>
            <CardContent>
              {equity.length > 0 ? (
                <EquityChart data={equity} />
              ) : (
                <div className="flex h-72 items-center justify-center text-xs text-muted-foreground">
                  No equity data yet — first trade on March 4th
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Risk Metrics</CardTitle>
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

        {/* ── Positions Table ──────────────────────────────────── */}
        <Card>
          <CardHeader>
            <CardTitle>Open Positions</CardTitle>
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
              <CardTitle>Logs</CardTitle>
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
              <CardTitle>TCA</CardTitle>
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
              <CardTitle>Feature Drift</CardTitle>
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
            <p className="text-xs text-muted-foreground">
              {status?.positions_count ?? 0} positions
            </p>
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

const equityChartConfig = {
  equity: {
    label: "Equity",
    color: "var(--foreground)",
  },
} satisfies ChartConfig;

function EquityChart({ data }: { data: EquityPoint[] }) {
  const chartData = data.map((d) => ({
    date: fmtDate(d.timestamp),
    equity: d.equity,
  }));

  return (
    <ChartContainer config={equityChartConfig} className="h-72 w-full">
      <AreaChart data={chartData}>
        <defs>
          <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
            <stop
              offset="0%"
              stopColor="var(--foreground)"
              stopOpacity={0.15}
            />
            <stop
              offset="100%"
              stopColor="var(--foreground)"
              stopOpacity={0}
            />
          </linearGradient>
        </defs>
        <XAxis
          dataKey="date"
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => v}
          fontSize={10}
        />
        <YAxis
          tickLine={false}
          axisLine={false}
          tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
          fontSize={10}
          width={50}
        />
        <ChartTooltip content={<ChartTooltipContent />} />
        <Area
          type="monotone"
          dataKey="equity"
          stroke="var(--foreground)"
          strokeWidth={1.5}
          fill="url(#equityGradient)"
        />
      </AreaChart>
    </ChartContainer>
  );
}
