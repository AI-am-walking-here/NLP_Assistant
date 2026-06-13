"use client";

import type { PipelineModule } from "@/lib/system-capabilities";
import { MODULE_LABELS } from "@/lib/system-capabilities";

type Props = {
  module: PipelineModule;
  active: boolean;
  compact?: boolean;
};

const ICONS: Record<PipelineModule, string> = {
  vector: "◎",
  graph: "⬡",
  rank: "⇅",
  sft: "✦",
};

export function ModuleNode({ module, active, compact = false }: Props) {
  if (compact) {
    return (
      <div
        className={`flex h-8 w-8 items-center justify-center rounded-full border text-sm ${
          active
            ? "border-accent bg-accent/15 text-accent"
            : "border-slide-border bg-slide-bg text-slide-muted opacity-40"
        }`}
        title={MODULE_LABELS[module]}
      >
        {ICONS[module]}
      </div>
    );
  }

  return (
    <div
      className={`flex flex-col items-center gap-1 ${
        active ? "opacity-100" : "opacity-35"
      }`}
      title={MODULE_LABELS[module]}
    >
      <div
        className={`flex h-10 w-10 items-center justify-center rounded-full border-2 text-base ${
          active
            ? "border-accent bg-accent/15 text-accent shadow-[0_0_12px_rgba(79,209,237,0.25)]"
            : "border-slide-border bg-slide-bg text-slide-muted"
        }`}
      >
        {ICONS[module]}
      </div>
      <span
        className={`font-mono text-[10px] uppercase tracking-wide ${
          active ? "text-accent" : "text-slide-muted"
        }`}
      >
        {MODULE_LABELS[module]}
      </span>
    </div>
  );
}
