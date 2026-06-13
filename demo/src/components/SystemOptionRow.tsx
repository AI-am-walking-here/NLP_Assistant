"use client";

import type { SystemInfo } from "@/lib/types";
import { displayForSystem } from "@/lib/system-display";
import { MODULE_LABELS, modulesForSystem } from "@/lib/system-capabilities";

const MODULE_ICONS: Record<string, string> = {
  vector: "◎",
  graph: "⬡",
  rank: "⇅",
  sft: "✦",
};

type Props = {
  system: SystemInfo;
  selected: boolean;
  onSelect: () => void;
};

export function SystemOptionRow({ system, selected, onSelect }: Props) {
  const display = displayForSystem(system);
  const active = new Set(modulesForSystem(system.id));

  return (
    <button
      type="button"
      role="option"
      aria-selected={selected}
      onClick={onSelect}
      className={`flex w-full items-start gap-3 rounded-lg border px-3 py-3 text-left transition-colors ${
        selected
          ? "border-accent/50 bg-accent/10"
          : "border-transparent bg-slide-bg hover:border-slide-border hover:bg-slide-elevated"
      }`}
    >
      <div className="min-w-0 flex-1 space-y-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-base font-medium text-slide-ink">
            {display.title}
          </span>
          {display.badge && (
            <span className="rounded-full border border-accent/40 bg-accent/15 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wide text-accent">
              {display.badge}
            </span>
          )}
          {system.factscore_mean != null && (
            <span className="ml-auto font-mono text-sm font-semibold text-accent">
              {system.factscore_mean.toFixed(2)}
            </span>
          )}
        </div>
        <p className="text-sm leading-snug text-slide-muted">{display.subtitle}</p>
        <div className="flex items-center gap-1 pt-0.5">
          {(["vector", "graph", "rank", "sft"] as const).map((mod) => (
            <span
              key={mod}
              title={MODULE_LABELS[mod]}
              className={`flex h-7 w-7 items-center justify-center rounded-full border text-xs ${
                active.has(mod)
                  ? "border-accent/40 bg-accent/10 text-accent"
                  : "border-slide-border/60 text-slide-muted/30"
              }`}
            >
              {MODULE_ICONS[mod]}
            </span>
          ))}
        </div>
      </div>
    </button>
  );
}
