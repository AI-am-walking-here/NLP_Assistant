"use client";

import { memo, useMemo } from "react";
import type { Passage } from "@/lib/types";
import { buildPaperColorMap } from "@/lib/paper-colors";
import { PassageCard } from "./PassageCard";
import { PassageScoreChart } from "./PassageScoreChart";

type Props = {
  passages: Passage[];
  loading?: boolean;
  selectedIndex?: number | null;
  onSelectPassage?: (index: number) => void;
};

function EmptyState() {
  return (
    <div className="flex min-h-[8rem] flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-slide-border bg-slide-bg p-4 text-center">
      <div className="flex gap-1">
        {[1, 2, 3, 4].map((n) => (
          <div
            key={n}
            className="h-8 w-3 rounded-t bg-slide-elevated"
            style={{ height: `${12 + n * 8}px` }}
          />
        ))}
      </div>
      <span className="font-mono text-xs text-slide-muted">Awaiting retrieval</span>
    </div>
  );
}

export const PassagesPanel = memo(function PassagesPanel({
  passages,
  loading,
  selectedIndex,
  onSelectPassage,
}: Props) {
  const scoreWidths = useMemo(() => {
    const maxScore = Math.max(...passages.map((p) => p.score), 0.001);
    return passages.map((p) => Math.round((p.score / maxScore) * 100));
  }, [passages]);

  const colorMap = useMemo(() => buildPaperColorMap(passages), [passages]);

  if (loading) {
    return (
      <div className="space-y-3">
        <div className="flex h-16 items-end gap-1 rounded-lg border border-slide-border bg-slide-bg p-3">
          {[40, 65, 50, 30, 20].map((h, i) => (
            <div
              key={i}
              className="flex-1 rounded-t bg-accent/25"
              style={{ height: `${h}%` }}
            />
          ))}
        </div>
        {[1, 2].map((i) => (
          <div
            key={i}
            className="h-16 rounded-lg border border-slide-border bg-slide-elevated"
          />
        ))}
      </div>
    );
  }

  if (passages.length === 0) {
    return <EmptyState />;
  }

  return (
    <div>
      <PassageScoreChart passages={passages} />
      <div className="space-y-3">
        {passages.map((p, i) => (
          <PassageCard
            key={`${p.paper_id}-${i}`}
            passage={p}
            index={i}
            scoreWidthPct={scoreWidths[i]}
            colorMap={colorMap}
            selected={selectedIndex === i}
            onSelect={onSelectPassage}
          />
        ))}
      </div>
    </div>
  );
});
