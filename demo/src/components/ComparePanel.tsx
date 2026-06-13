"use client";

import type { GenerateResponse } from "@/lib/types";
import { wordOverlapPct } from "@/lib/passage-highlight";
import { AbstractContent } from "./AbstractContent";

type Props = {
  resultA: GenerateResponse | null;
  resultB: GenerateResponse | null;
  labelA: string;
  labelB: string;
  loading?: boolean;
};

export function ComparePanel({
  resultA,
  resultB,
  labelA,
  labelB,
  loading,
}: Props) {
  const overlap =
    resultA?.abstract && resultB?.abstract
      ? wordOverlapPct(resultA.abstract, resultB.abstract)
      : null;

  return (
    <div className="slide-panel-elevated col-span-12 grid min-h-0 grid-cols-2 gap-3 overflow-hidden p-3">
      <header className="col-span-2 flex shrink-0 items-center justify-between border-b border-slide-border pb-2">
        <h2 className="section-kicker">A/B compare</h2>
        {overlap != null ? (
          <span className="font-mono text-xs text-slide-muted">
            Abstract overlap:{" "}
            <span className="font-semibold text-accent">{overlap}%</span>
          </span>
        ) : loading ? (
          <span className="font-mono text-xs text-accent">Running both systems…</span>
        ) : null}
      </header>
      {[resultA, resultB].map((result, idx) => {
        const label = idx === 0 ? labelA : labelB;
        return (
          <div
            key={label}
            className="flex min-h-0 flex-col overflow-hidden rounded-lg border border-slide-border bg-slide-bg"
          >
            <div className="flex shrink-0 items-center justify-between border-b border-slide-border px-3 py-2">
              <span className="text-sm font-medium text-slide-ink">{label}</span>
              {result ? (
                <span className="font-mono text-[10px] text-slide-muted">
                  {result.passages.length} passages
                </span>
              ) : null}
            </div>
            <div className="scroll-panel min-h-0 flex-1 p-3">
              {loading && !result ? (
                <p className="text-sm text-slide-muted">Generating…</p>
              ) : result ? (
                <AbstractContent
                  content={result.abstract}
                  passages={result.passages}
                />
              ) : (
                <p className="text-sm text-slide-muted">Awaiting run</p>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
