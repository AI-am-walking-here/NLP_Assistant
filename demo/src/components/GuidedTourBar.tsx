"use client";

import { GUIDED_ACTS, type GuidedAct } from "@/lib/guided-tour";

type Props = {
  actIndex: number;
  running: boolean;
  onStart: () => void;
  onNext: () => void;
  onReset: () => void;
};

export function GuidedTourBar({
  actIndex,
  running,
  onStart,
  onNext,
  onReset,
}: Props) {
  const act: GuidedAct | undefined = GUIDED_ACTS[actIndex];
  const isLast = actIndex >= GUIDED_ACTS.length - 1;

  return (
    <div className="mb-2 shrink-0 rounded-lg border border-accent/30 bg-accent/5 px-4 py-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="font-mono text-[10px] uppercase tracking-wide text-accent">
            Guided tour · Act {actIndex + 1}/{GUIDED_ACTS.length}
          </p>
          <p className="text-sm font-medium text-slide-ink">
            {act?.title ?? "Complete"}
          </p>
          <p className="text-xs text-slide-muted">{act?.caption}</p>
        </div>
        <div className="flex shrink-0 gap-2">
          {actIndex === 0 && !running ? (
            <button type="button" onClick={onStart} className="btn-primary px-4 py-2 text-sm">
              Start tour
            </button>
          ) : (
            <>
              <button
                type="button"
                onClick={onNext}
                disabled={running || isLast}
                className="btn-primary px-4 py-2 text-sm"
              >
                {running ? "Running…" : isLast ? "Done" : "Next act →"}
              </button>
              <button
                type="button"
                onClick={onReset}
                disabled={running}
                className="btn-secondary px-3 py-2 text-xs"
              >
                Reset
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
