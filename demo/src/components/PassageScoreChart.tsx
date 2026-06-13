"use client";

import { memo } from "react";
import type { Passage } from "@/lib/types";

type Props = {
  passages: Passage[];
};

export const PassageScoreChart = memo(function PassageScoreChart({
  passages,
}: Props) {
  if (passages.length === 0) return null;

  const maxScore = Math.max(...passages.map((p) => p.score), 0.001);

  return (
    <div className="mb-3 shrink-0 rounded-lg border border-slide-border bg-slide-bg p-3">
      <div className="mb-2 flex items-end gap-1" style={{ height: "3.5rem" }}>
        {passages.map((p, i) => {
          const h = Math.max(12, Math.round((p.score / maxScore) * 100));
          return (
            <div
              key={`${p.paper_id}-${i}`}
              className="group flex flex-1 flex-col items-center justify-end gap-1"
              title={`#${i + 1} · ${p.score.toFixed(3)}`}
            >
              <div
                className="w-full min-w-[4px] rounded-t bg-gradient-to-t from-accent/30 to-accent transition-opacity group-hover:opacity-100"
                style={{ height: `${h}%` }}
              />
            </div>
          );
        })}
      </div>
      <div className="flex justify-between font-mono text-[10px] text-slide-muted">
        <span>#1</span>
        <span>retrieval rank</span>
        <span>#{passages.length}</span>
      </div>
    </div>
  );
});
