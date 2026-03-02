export type BotId = "bot-a" | "bot-b";

export type MarketSession = "Pre-market" | "Open" | "After-hours" | "Closed";

export type EquityDisplayMode = "abs" | "pct";

export type SessionMode = "pnl" | "spread";

export type TimeWindowKey = "30m" | "1h" | "2h" | "4h" | "all";

export interface AccountData {
  buying_power: number;
  cash: number;
  currency: string;
  equity: number;
  portfolio_value: number;
  status: string;
  timestamp: string;
}

export interface RegimeData {
  exposure_multiplier: number;
  message: string;
  regime: "normal" | "caution" | "halt";
  spy_drawdown: number;
  vix: number;
}

export interface BotState {
  cash: number;
  final_equity: number;
  last_shutdown: string;
  positions: Record<string, unknown>;
  reason: string;
}

export interface StatusData {
  account: AccountData;
  bot_state: BotState;
  has_backtest_results: boolean;
  positions_count: number;
  regime: RegimeData;
  timestamp: string;
}

export interface Position {
  symbol: string;
  qty: number;
  market_value: number;
  avg_entry_price: number;
  current_price: number;
  unrealized_pl: number;
  unrealized_plpc: number;
  side: string;
}

export interface RiskData {
  max_drawdown?: number;
  current_drawdown?: number;
  sharpe_ratio?: number;
  sortino_ratio?: number;
  var_95?: number;
  cvar_95?: number;
  total_trades?: number;
  win_rate?: number;
  [key: string]: unknown;
}

export interface TcaData {
  avg_implementation_shortfall_bps?: number;
  total_trades?: number;
  avg_fill_rate?: number;
  [key: string]: unknown;
}

export interface DriftData {
  drifted_features?: string[];
  drift_count?: number;
  total_features?: number;
  [key: string]: unknown;
}

export interface EquityPoint {
  timestamp: string;
  equity: number;
}

export interface HealthData {
  status: string;
  uptime?: number;
  [key: string]: unknown;
}

/** One intraday equity snapshot stored in Postgres on Bot B VPS. */
export interface SessionPointDb {
  /** ISO-8601 timestamp in ET (from the DB's AT TIME ZONE conversion). */
  ts: string;
  equity_a: number | null;
  equity_b: number | null;
  /** SPY drawdown from rolling peak (negative fraction, e.g. -0.012 = -1.2%). Null for pre-schema rows. */
  spy_dd: number | null;
  /** Open position count for Bot A. Null for pre-schema rows. */
  pos_a: number | null;
  /** Open position count for Bot B. Null for pre-schema rows. */
  pos_b: number | null;
}

/** Live session point for chart display */
export interface SessionPoint {
  time: string;
  a: number | null;
  b: number | null;
  spy_dd: number | null;
  pos_a: number | null;
  pos_b: number | null;
}

export interface SessionHistoryResponse {
  history: SessionPointDb[];
  count: number;
  from: string;
  to: string;
}

export interface MarketStatus {
  session: MarketSession;
  countdown: string;
}

export interface ComparisonBotData {
  label: string;
  sublabel: string;
  status: StatusData | null;
  healthy: boolean | null;
}

export interface ChartDataPoint {
  date: string;
  botA?: number;
  botB?: number;
}

export interface LiveChartDataPoint {
  time: string;
  a: number | null;
  b: number | null;
  spread: number | null;
  ma_a?: number | null;
  ma_b?: number | null;
  ma_spread?: number | null;
}

export interface JumpDot {
  time: string;
  val: number;
  color: string;
}

export interface OHLCStats {
  open: number;
  high: number;
  low: number;
  now: number;
}

export interface SessionStats {
  a: OHLCStats | null;
  b: OHLCStats | null;
  spread: OHLCStats | null;
}
