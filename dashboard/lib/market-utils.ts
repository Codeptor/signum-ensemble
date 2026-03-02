import { MarketSession, MarketStatus } from "@/types/dashboard";
import { MARKET_HOURS } from "@/lib/constants";

/**
 * Get current market session and countdown to next event
 */
export function getMarketStatus(now: Date): MarketStatus {
  const ny = new Date(
    now.toLocaleString("en-US", { timeZone: "America/New_York" })
  );
  const day = ny.getDay();
  const h = ny.getHours();
  const m = ny.getMinutes();
  const mins = h * 60 + m;

  const isWeekday = day >= 1 && day <= 5;

  const OPEN = MARKET_HOURS.OPEN;
  const CLOSE = MARKET_HOURS.CLOSE;
  const PRE = MARKET_HOURS.PRE_MARKET_START;
  const AFTER = MARKET_HOURS.AFTER_HOURS_END;

  let session: MarketSession = "Closed";
  let targetMins: number = OPEN; // default: next open
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

/**
 * Get badge variant based on market session
 */
export function getSessionVariant(
  session: MarketSession
): "outline" | "secondary" | "destructive" {
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

/**
 * Get badge variant based on regime
 */
export function getRegimeVariant(
  regime: string | undefined
): "outline" | "secondary" | "destructive" {
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

/**
 * Check if market is currently open (9:30-16:00 ET)
 */
export function isMarketOpen(now: Date = new Date()): boolean {
  const ny = new Date(
    now.toLocaleString("en-US", { timeZone: "America/New_York" })
  );
  const etMinutes = ny.getHours() * 60 + ny.getMinutes();
  return etMinutes >= MARKET_HOURS.OPEN && etMinutes < MARKET_HOURS.CLOSE;
}
