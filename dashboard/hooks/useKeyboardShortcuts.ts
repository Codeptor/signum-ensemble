"use client";

import { useEffect, useCallback } from "react";

export interface KeyboardShortcutHandlers {
  onSwitchBotA: () => void;
  onSwitchBotB: () => void;
  onRefresh: () => void;
  onTogglePause: () => void;
}

export function useKeyboardShortcuts(
  handlers: KeyboardShortcutHandlers,
  enabled: boolean = true
) {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!enabled) return;

      // Ignore if typing in an input
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      switch (e.key) {
        case "1":
          handlers.onSwitchBotA();
          break;
        case "2":
          handlers.onSwitchBotB();
          break;
        case "r":
        case "R":
          handlers.onRefresh();
          break;
        case " ":
          e.preventDefault();
          handlers.onTogglePause();
          break;
      }
    },
    [handlers, enabled]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);
}
