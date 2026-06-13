"use client";

import type { PipelineStage } from "@/lib/types";

const ICONS: Record<string, string> = {
  compose: "✎",
  vector: "◎",
  graph: "⬡",
  merge: "⊕",
  rerank: "⇅",
  generate: "✦",
};

type Props = {
  stages: PipelineStage[];
  activeIndex: number;
  mock?: boolean;
  fast?: boolean;
};

export function PipelineStageProgress({
  stages,
  activeIndex,
  mock,
  fast,
}: Props) {
  const visible =
    stages.length > 0
      ? stages
      : fast
        ? [{ id: "generate", label: "8B generate", detail: "fast path" }]
        : mock
          ? [
              { id: "vector", label: "Mock retrieve", detail: "" },
              { id: "generate", label: "Mock generate", detail: "" },
            ]
          : [
              { id: "vector", label: "BGE + FAISS", detail: "" },
              { id: "rerank", label: "RankRAG", detail: "" },
              { id: "generate", label: "8B generate", detail: "" },
            ];

  return (
    <div className="flex h-full flex-col items-center justify-center gap-6 p-6">
      <div className="relative h-20 w-20">
        <div className="absolute inset-0 rounded-full border-2 border-accent/20" />
        <div className="absolute inset-0 animate-spin rounded-full border-2 border-transparent border-t-accent" />
        <div className="absolute inset-0 flex items-center justify-center font-mono text-sm font-semibold text-accent">
          8B
        </div>
      </div>
      <div className="w-full max-w-md space-y-2">
        {visible.map((stage, i) => {
          const done = i < activeIndex;
          const active = i === activeIndex;
          const icon = ICONS[stage.id] ?? "•";
          return (
            <div
              key={`${stage.id}-${i}`}
              className={`flex items-center gap-3 rounded-lg border px-3 py-2 transition-colors ${
                active
                  ? "border-accent/50 bg-accent/10"
                  : done
                    ? "border-accent/30 bg-accent/5"
                    : "border-slide-border bg-slide-bg/50 opacity-60"
              }`}
            >
              <span
                className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full border text-sm ${
                  active || done
                    ? "border-accent/50 text-accent"
                    : "border-slide-border text-slide-muted"
                }`}
              >
                {icon}
              </span>
              <div className="min-w-0 flex-1">
                <div className="text-sm font-medium text-slide-ink">
                  {stage.label}
                  {stage.count != null ? (
                    <span className="ml-2 font-mono text-xs text-accent">
                      {stage.count}
                    </span>
                  ) : null}
                </div>
                {stage.detail ? (
                  <div className="truncate text-xs text-slide-muted">
                    {stage.detail}
                  </div>
                ) : null}
              </div>
              {active ? (
                <span className="font-mono text-[10px] uppercase text-accent">
                  running
                </span>
              ) : done ? (
                <span className="font-mono text-[10px] uppercase text-accent/70">
                  done
                </span>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}
