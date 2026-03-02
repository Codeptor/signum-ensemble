import { BotId, StatusData, Position, RiskData, TcaData, DriftData, EquityPoint, HealthData, SessionHistoryResponse } from "./types";

const BASE = "/api/bot";

async function fetchBot<T>(bot: BotId, endpoint: string): Promise<T | null> {
  try {
    const res = await fetch(`${BASE}/${bot}/${endpoint}`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function fetchStatus(bot: BotId) {
  return fetchBot<StatusData>(bot, "api/status");
}

export async function fetchPositions(bot: BotId): Promise<Position[]> {
  const data = await fetchBot<{ positions: Position[] } | Position[]>(bot, "api/positions");
  if (!data) return [];
  if (Array.isArray(data)) return data;
  if (Array.isArray(data.positions)) return data.positions;
  return [];
}

export async function fetchRisk(bot: BotId) {
  return fetchBot<RiskData>(bot, "api/risk");
}

export async function fetchTca(bot: BotId) {
  return fetchBot<TcaData>(bot, "api/tca");
}

export async function fetchDrift(bot: BotId) {
  return fetchBot<DriftData>(bot, "api/drift");
}

export async function fetchEquity(bot: BotId): Promise<EquityPoint[]> {
  type RawPoint = { date?: string; timestamp?: string; equity: number };
  const data = await fetchBot<
    RawPoint[] |
    { history: RawPoint[] } |
    { equity: RawPoint[] } |
    { records: RawPoint[] }
  >(bot, "api/equity");
  if (!data) return [];

  // Unwrap whichever envelope key the backend uses
  let raw: RawPoint[] = [];
  if (Array.isArray(data))                                    raw = data;
  else if ("history" in data && Array.isArray(data.history)) raw = data.history;
  else if ("equity"  in data && Array.isArray(data.equity))  raw = data.equity;
  else if ("records" in data && Array.isArray(data.records)) raw = data.records;

  // Normalize: backend sends "date", EquityPoint type expects "timestamp"
  return raw.map((pt) => ({
    timestamp: pt.timestamp ?? pt.date ?? "",
    equity: pt.equity,
  }));
}

export async function fetchHealth(bot: BotId) {
  return fetchBot<HealthData>(bot, "healthz");
}

/**
 * Persist one merged equity snapshot to Bot B's local Postgres.
 * Always routes to bot-b regardless of the active tab — Bot B is the
 * central session store for both bots' intraday data.
 */
export async function storeSessionPoint(
  equityA: number | null,
  equityB: number | null,
  ts?: string,
): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/bot-b/api/session/store`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ts, equity_a: equityA, equity_b: equityB }),
      cache: "no-store",
    });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Fetch historical intraday session data from Bot B's Postgres.
 * @param from YYYY-MM-DD inclusive start (ET)
 * @param to   YYYY-MM-DD inclusive end   (ET)
 */
export async function fetchSessionHistory(
  from: string,
  to: string,
): Promise<SessionHistoryResponse | null> {
  try {
    const res = await fetch(
      `${BASE}/bot-b/api/session/history?from=${from}&to=${to}`,
      { cache: "no-store" },
    );
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function fetchLogs(bot: BotId, lines = 50): Promise<string> {
  const data = await fetchBot<{ log: string[]; logs: string } | string>(bot, `api/logs?lines=${lines}`);
  if (!data) return "";
  if (typeof data === "string") return data;
  if (Array.isArray(data.log)) return data.log.join("\n");
  if (data.logs) return data.logs;
  return "";
}
