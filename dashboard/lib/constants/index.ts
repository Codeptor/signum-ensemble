import { TimeWindowKey } from "@/types/dashboard";

/**
 * Dashboard constants
 */

export const STARTING_EQUITY = 100_000;

export const AUTO_REFRESH_INTERVAL = 30_000; // 30 seconds

export const LOG_LINES_DEFAULT = 80;

export const SESSION_POINT_LIMIT = 10_000;

export const MA_WINDOW = 5;

export const MARKET_HOURS = {
  PRE_MARKET_START: 240, // 4:00 AM in minutes
  OPEN: 570, // 9:30 AM in minutes
  CLOSE: 960, // 4:00 PM in minutes
  AFTER_HOURS_END: 1200, // 8:00 PM in minutes
} as const;

export const WINDOW_LIMITS: Record<TimeWindowKey, number> = {
  "30m": 60,
  "1h": 120,
  "2h": 240,
  "4h": 480,
  "all": Infinity,
} as const;

// S&P 500 sector map (top holdings — covers most likely positions)
export const SECTOR_MAP: Record<string, string> = {
  // Technology
  AAPL: "Technology", MSFT: "Technology", NVDA: "Technology", GOOG: "Technology",
  GOOGL: "Technology", META: "Technology", AVGO: "Technology", ADBE: "Technology",
  CRM: "Technology", CSCO: "Technology", ORCL: "Technology", ACN: "Technology",
  AMD: "Technology", INTC: "Technology", IBM: "Technology", QCOM: "Technology",
  TXN: "Technology", NOW: "Technology", AMAT: "Technology", MU: "Technology",
  INTU: "Technology", LRCX: "Technology", KLAC: "Technology", SNPS: "Technology",
  CDNS: "Technology", MRVL: "Technology", FTNT: "Technology", PANW: "Technology",
  // Consumer Discretionary
  AMZN: "Consumer Disc.", TSLA: "Consumer Disc.", HD: "Consumer Disc.",
  MCD: "Consumer Disc.", NKE: "Consumer Disc.", LOW: "Consumer Disc.",
  SBUX: "Consumer Disc.", TJX: "Consumer Disc.", BKNG: "Consumer Disc.",
  CMG: "Consumer Disc.", ORLY: "Consumer Disc.", MAR: "Consumer Disc.",
  // Financials
  BRK: "Financials", JPM: "Financials", V: "Financials", MA: "Financials",
  BAC: "Financials", WFC: "Financials", GS: "Financials", MS: "Financials",
  SPGI: "Financials", BLK: "Financials", AXP: "Financials", C: "Financials",
  SCHW: "Financials", CB: "Financials", MMC: "Financials", PGR: "Financials",
  // Healthcare
  UNH: "Healthcare", JNJ: "Healthcare", LLY: "Healthcare", ABBV: "Healthcare",
  MRK: "Healthcare", PFE: "Healthcare", TMO: "Healthcare", ABT: "Healthcare",
  DHR: "Healthcare", BMY: "Healthcare", AMGN: "Healthcare", GILD: "Healthcare",
  ISRG: "Healthcare", VRTX: "Healthcare", SYK: "Healthcare", BSX: "Healthcare",
  MDT: "Healthcare", REGN: "Healthcare", ZTS: "Healthcare", ELV: "Healthcare",
  // Energy
  XOM: "Energy", CVX: "Energy", COP: "Energy", SLB: "Energy", EOG: "Energy",
  MPC: "Energy", PSX: "Energy", VLO: "Energy", OXY: "Energy", HES: "Energy",
  // Consumer Staples
  PG: "Consumer Stap.", KO: "Consumer Stap.", PEP: "Consumer Stap.",
  COST: "Consumer Stap.", WMT: "Consumer Stap.", PM: "Consumer Stap.",
  MO: "Consumer Stap.", CL: "Consumer Stap.", MDLZ: "Consumer Stap.",
  // Materials
  LIN: "Materials", APD: "Materials", SHW: "Materials", ECL: "Materials",
  FCX: "Materials", NEM: "Materials", NUE: "Materials", DOW: "Materials",
  // Utilities
  NEE: "Utilities", DUK: "Utilities", SO: "Utilities", D: "Utilities",
  AEP: "Utilities", SRE: "Utilities", EXC: "Utilities", XEL: "Utilities",
  // Real Estate
  AMT: "Real Estate", PLD: "Real Estate", CCI: "Real Estate",
  EQIX: "Real Estate", SPG: "Real Estate", PSA: "Real Estate",
  // Industrials
  UNP: "Industrials", RTX: "Industrials", HON: "Industrials", UPS: "Industrials",
  BA: "Industrials", CAT: "Industrials", DE: "Industrials", LMT: "Industrials",
  GE: "Industrials", MMM: "Industrials", GD: "Industrials", NOC: "Industrials",
  // Communication
  T: "Communication", VZ: "Communication", TMUS: "Communication",
  DIS: "Communication", CMCSA: "Communication", NFLX: "Communication",
} as const;

// Chart color configurations
export const CHART_COLORS = {
  botA: "hsl(160 60% 50%)",
  botB: "hsl(213 80% 58%)",
  spread: "hsl(38 90% 58%)",
  rebalance: "hsl(38 70% 50%)",
  positive: "hsl(142 55% 40%)",
  negative: "hsl(0 62% 52%)",
  neutral: "hsl(20 70% 50%)",
} as const;
