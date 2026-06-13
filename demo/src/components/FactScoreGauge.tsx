"use client";

import { factScorePct } from "@/lib/system-capabilities";

type Props = {
  score: number | null;
  label?: string;
  compact?: boolean;
};

export function FactScoreGauge({
  score,
  label = "FAct",
  compact = false,
}: Props) {
  const pct = factScorePct(score);
  const display = score != null ? score.toFixed(2) : "—";

  if (compact) {
    return (
      <div
        className="flex min-w-[5.5rem] flex-col gap-1"
        title={`FActScore: ${display}`}
      >
        <div className="flex items-center justify-between gap-1">
          <span className="font-mono text-[10px] uppercase text-slide-muted">
            {label}
          </span>
          <span className="font-mono text-xs font-semibold text-accent">
            {display}
          </span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-slide-bg">
          <div
            className="h-full rounded-full bg-accent/80"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-1.5" title={`FActScore: ${display}`}>
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-xs uppercase tracking-wide text-slide-muted">
          {label}
        </span>
        <span className="font-mono text-sm font-semibold text-accent">
          {display}
        </span>
      </div>
      <div className="relative h-3 overflow-hidden rounded-full bg-slide-bg">
        <div
          className="h-full rounded-full bg-gradient-to-r from-accent/40 to-accent"
          style={{ width: `${pct}%` }}
        />
        {[25, 50, 75].map((tick) => (
          <div
            key={tick}
            className="absolute top-0 h-full w-px bg-slide-border/80"
            style={{ left: `${tick}%` }}
          />
        ))}
      </div>
    </div>
  );
}
