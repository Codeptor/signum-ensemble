"use client";

import { useState, useCallback, useRef } from "react";
import { fetchStatus, fetchHealth, fetchEquity, storeSessionPoint } from "@/lib/api";
import { StatusData, EquityPoint, SessionPoint } from "@/types/dashboard";
import { SESSION_POINT_LIMIT } from "@/lib/constants";

export interface ComparisonDataState {
  statusA: StatusData | null;
  statusB: StatusData | null;
  healthA: boolean | null;
  healthB: boolean | null;
  equityA: EquityPoint[];
  equityB: EquityPoint[];
}

export interface UseComparisonDataReturn extends ComparisonDataState {
  loadComparison: () => Promise<void>;
  loadEquityCurves: () => Promise<void>;
  accumulateSessionPoints: (currentPoints: SessionPoint[]) => SessionPoint[];
}

export function useComparisonData(
  onNotify: (message: string) => void
): UseComparisonDataReturn {
  const [state, setState] = useState<ComparisonDataState>({
    statusA: null,
    statusB: null,
    healthA: null,
    healthB: null,
    equityA: [],
    equityB: [],
  });

  // Always-fresh mirror of latest state — lets callbacks read current values
  // without becoming stale or needing state in their dep arrays
  const stateRef = useRef(state);

  // Regime change tracking — owned entirely inside this hook
  const prevRegimeARef = useRef<string | null>(null);
  const prevRegimeBRef = useRef<string | null>(null);

  const loadComparison = useCallback(async () => {
    const [sA, sB, hA, hB] = await Promise.all([
      fetchStatus("bot-a"),
      fetchStatus("bot-b"),
      fetchHealth("bot-a"),
      fetchHealth("bot-b"),
    ]);

    // Regime change notifications — use fresh API data, not stale state
    const regimeA = sA?.regime?.regime;
    const regimeB = sB?.regime?.regime;

    if (
      prevRegimeARef.current &&
      regimeA &&
      regimeA !== prevRegimeARef.current &&
      regimeA !== "normal"
    ) {
      onNotify(`Bot A regime: ${regimeA.toUpperCase()}`);
    }
    prevRegimeARef.current = regimeA ?? null;

    if (
      prevRegimeBRef.current &&
      regimeB &&
      regimeB !== prevRegimeBRef.current &&
      regimeB !== "normal"
    ) {
      onNotify(`Bot B regime: ${regimeB.toUpperCase()}`);
    }
    prevRegimeBRef.current = regimeB ?? null;

    const newState: ComparisonDataState = {
      ...stateRef.current,
      statusA: sA,
      statusB: sB,
      healthA: hA != null,
      healthB: hB != null,
    };
    stateRef.current = newState;
    setState(newState);
  }, [onNotify]);

  const loadEquityCurves = useCallback(async () => {
    const [eA, eB] = await Promise.all([
      fetchEquity("bot-a"),
      fetchEquity("bot-b"),
    ]);

    const newState: ComparisonDataState = {
      ...stateRef.current,
      equityA: eA,
      equityB: eB,
    };
    stateRef.current = newState;
    setState(newState);
  }, []);

  // Reads from stateRef — stable reference, no state in dep array
  const accumulateSessionPoints = useCallback(
    (currentPoints: SessionPoint[]): SessionPoint[] => {
      const { statusA, statusB } = stateRef.current;
      const eqA = statusA?.account?.equity;
      const eqB = statusB?.account?.equity;

      if (eqA == null && eqB == null) {
        return currentPoints;
      }

      const now = new Date();
      const time = now.toLocaleTimeString("en-US", {
        timeZone: "America/New_York",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
      });

      const spyDd =
        statusA?.regime?.spy_drawdown ??
        statusB?.regime?.spy_drawdown ??
        null;
      const posA = statusA?.positions_count ?? null;
      const posB = statusB?.positions_count ?? null;

      // Persist to Postgres (fire-and-forget)
      storeSessionPoint(
        eqA != null ? +eqA.toFixed(2) : null,
        eqB != null ? +eqB.toFixed(2) : null,
        now.toISOString(),
        spyDd != null ? +spyDd.toFixed(6) : null,
        posA,
        posB
      );

      return [
        ...currentPoints,
        {
          time,
          a: eqA != null ? +eqA.toFixed(2) : null,
          b: eqB != null ? +eqB.toFixed(2) : null,
          spy_dd: spyDd != null ? +spyDd.toFixed(6) : null,
          pos_a: posA,
          pos_b: posB,
        },
      ].slice(-SESSION_POINT_LIMIT);
    },
    [] // no deps — reads stateRef which is always current
  );

  return {
    ...state,
    loadComparison,
    loadEquityCurves,
    accumulateSessionPoints,
  };
}
