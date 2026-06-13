"use client";

import { memo, useMemo } from "react";
import type { Passage } from "@/lib/types";
import { MODULE_ORDER, modulesForSystem } from "@/lib/system-capabilities";
import { AbstractContent } from "./AbstractContent";
import type { PipelineStage } from "@/lib/types";
import { PipelineStageProgress } from "./PipelineStageProgress";

type Props = {
  content: string | null;
  passages: Passage[];
  loading: boolean;
  system: string;
  mockGeneration?: boolean;
  fastPath?: boolean;
  stages?: PipelineStage[];
  stageIndex?: number;
  highlightTokens?: Set<string>;
  presentation?: boolean;
};

function ModuleBadge({ icon, active }: { icon: string; active: boolean }) {
  return (
    <span
      className={`flex h-7 w-7 items-center justify-center rounded-full border text-xs ${
        active
          ? "border-accent/50 bg-accent/15 text-accent"
          : "border-slide-border bg-slide-bg text-slide-muted opacity-40"
      }`}
    >
      {icon}
    </span>
  );
}

function EmptyState() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 p-6 text-center">
      <div className="flex h-24 w-24 items-center justify-center rounded-2xl border-2 border-dashed border-slide-border bg-slide-elevated text-4xl text-slide-muted">
        📄
      </div>
      <div className="flex gap-2">
        {["◎", "⬡", "⇅", "✦"].map((icon) => (
          <span
            key={icon}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-slide-border bg-slide-bg text-sm text-slide-muted"
          >
            {icon}
          </span>
        ))}
      </div>
    </div>
  );
}

const MODULE_ICONS: Record<string, string> = {
  vector: "◎",
  graph: "⬡",
  rank: "⇅",
  sft: "✦",
};

export const AbstractPanel = memo(function AbstractPanel({
  content,
  passages,
  loading,
  system,
  mockGeneration,
  fastPath,
  stages,
  stageIndex = 0,
  highlightTokens,
  presentation,
}: Props) {
  const activeModules = useMemo(
    () => new Set(modulesForSystem(system)),
    [system],
  );
  const wordCount = content?.trim().split(/\s+/).filter(Boolean).length ?? 0;
  const wordPct = Math.min(100, Math.round((wordCount / 250) * 100));

  return (
    <section
      className={`slide-panel-elevated flex min-h-0 flex-col overflow-hidden p-4 ${
        presentation ? "col-span-7" : "col-span-5"
      }`}
    >
      <div className="mb-2 flex shrink-0 items-center justify-between gap-2">
        <h2 className="section-kicker">Abstract</h2>
        <div className="flex items-center gap-1">
          {MODULE_ORDER.map((mod) => (
            <ModuleBadge
              key={mod}
              icon={MODULE_ICONS[mod]}
              active={activeModules.has(mod)}
            />
          ))}
          {mockGeneration && (
            <span
              className="flex h-7 w-7 items-center justify-center rounded-full border border-grounded-warn/50 bg-grounded-warn/10 text-xs text-grounded-warn"
              title="Mock"
            >
              ◇
            </span>
          )}
        </div>
      </div>

      <div
        className={`scroll-panel min-h-0 flex-1 rounded-lg border ${
          loading
            ? "border-accent/30 bg-accent/5"
            : content
              ? "border-slide-border bg-slide-bg"
              : "border-dashed border-slide-border bg-slide-bg/50"
        }`}
      >
        {loading ? (
          <PipelineStageProgress
            stages={stages ?? []}
            activeIndex={stageIndex}
            mock={mockGeneration}
            fast={fastPath}
          />
        ) : content ? (
          <div className={presentation ? "text-base sm:text-lg" : ""}>
            <AbstractContent
              content={content}
              passages={passages}
              highlightTokens={highlightTokens}
            />
          </div>
        ) : (
          <EmptyState />
        )}
      </div>

      {content && !loading && (
        <div className="mt-2 shrink-0 space-y-1">
          <div className="flex justify-between font-mono text-[10px] text-slide-muted">
            <span>Length</span>
            <span>{wordCount}w</span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-slide-bg">
            <div
              className="h-full rounded-full bg-accent/80"
              style={{ width: `${wordPct}%` }}
            />
          </div>
        </div>
      )}
    </section>
  );
});
