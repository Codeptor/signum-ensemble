import { BotId, StatusData, Position, RiskData, TcaData, DriftData, EquityPoint, HealthData } from "./types";

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

export async function fetchPositions(bot: BotId) {
  return fetchBot<Position[]>(bot, "api/positions");
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

export async function fetchEquity(bot: BotId) {
  return fetchBot<EquityPoint[]>(bot, "api/equity");
}

export async function fetchHealth(bot: BotId) {
  return fetchBot<HealthData>(bot, "healthz");
}

export async function fetchLogs(bot: BotId, lines = 50) {
  return fetchBot<{ logs: string } | string>(bot, `api/logs?lines=${lines}`);
}
