"use client";

import { useState, useEffect, useCallback } from "react";
import { SessionPoint } from "@/types/dashboard";

const STORAGE_KEY = "signum_session_equity";

export interface UseSessionPointsReturn {
  points: SessionPoint[];
  setPoints: (update: SessionPoint[] | ((prev: SessionPoint[]) => SessionPoint[])) => void;
  clearPoints: () => void;
}

export function useSessionPoints(): UseSessionPointsReturn {
  const [points, setPointsState] = useState<SessionPoint[]>(() => {
    if (typeof window === "undefined") return [];
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        // Backward-compat: old rows lack spy_dd/pos_a/pos_b — fill with null
        const parsed = JSON.parse(stored) as Array<{
          time: string;
          a: number | null;
          b: number | null;
          spy_dd?: number | null;
          pos_a?: number | null;
          pos_b?: number | null;
        }>;
        return parsed.map((p) => ({
          spy_dd: null,
          pos_a: null,
          pos_b: null,
          ...p,
        }));
      }
    } catch {
      // Ignore parse errors
    }
    return [];
  });

  // Persist to localStorage on every update
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(points));
    } catch {
      // Ignore storage errors
    }
  }, [points]);

  // Accepts both direct value and functional updater
  const setPoints = useCallback(
    (update: SessionPoint[] | ((prev: SessionPoint[]) => SessionPoint[])) => {
      if (typeof update === "function") {
        setPointsState(update);
      } else {
        setPointsState(update);
      }
    },
    []
  );

  const clearPoints = useCallback(() => {
    setPointsState([]);
  }, []);

  return { points, setPoints, clearPoints };
}
