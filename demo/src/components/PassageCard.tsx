"use client";

import { memo, useState } from "react";
import type { Passage } from "@/lib/types";
import { PaperIdTag } from "./PaperIdTag";

type Props = {
  passage: Passage;
  index: number;
  scoreWidthPct: number;
  colorMap: Map<string, number>;
  selected?: boolean;
  onSelect?: (index: number) => void;
};

const RANK_STYLES = [
  "border-amber-400/60 bg-amber-400/15 text-amber-300",
  "border-slate-300/50 bg-slate-300/10 text-slate-200",
  "border-orange-600/50 bg-orange-600/10 text-orange-300",
];

export const PassageCard = memo(function PassageCard({
  passage,
  index,
  scoreWidthPct,
  colorMap,
  selected,
  onSelect,
}: Props) {
  const [expanded, setExpanded] = useState(false);
  const rankStyle =
    index < 3 ? RANK_STYLES[index] : "border-slide-border bg-slide-bg text-slide-muted";

  return (
    <article
      role={onSelect ? "button" : undefined}
      tabIndex={onSelect ? 0 : undefined}
      onClick={() => onSelect?.(index)}
      onKeyDown={(e) => {
        if (onSelect && (e.key === "Enter" || e.key === " ")) {
          e.preventDefault();
          onSelect(index);
        }
      }}
      className={`rounded-lg border bg-slide-surface p-3 transition-colors ${
        selected
          ? "border-accent/60 bg-accent/10 ring-1 ring-accent/30"
          : expanded
            ? "passage-card-expanded border-accent/45 bg-slide-elevated"
            : "passage-card border-slide-border"
      } ${onSelect ? "cursor-pointer hover:border-accent/40" : ""}`}
    >
      <div className="mb-2 flex items-start gap-3">
        <div
          className={`flex h-10 w-10 shrink-0 flex-col items-center justify-center rounded-full border-2 font-mono text-xs font-bold ${rankStyle}`}
        >
          <span className="text-[9px] opacity-70">#</span>
          {index + 1}
        </div>
        <div className="min-w-0 flex-1">
          <div className="mb-1 flex flex-wrap items-center gap-2 text-xs sm:text-sm">
            <PaperIdTag
              paperId={passage.paper_id}
              colorMap={colorMap}
              className="text-sm"
            />
            {passage.section_heading && (
              <span className="truncate rounded bg-slide-bg px-2 py-0.5 text-slide-muted">
                {passage.section_heading}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-slide-bg">
              <div
                className="h-full rounded-full bg-gradient-to-r from-accent/50 to-accent"
                style={{ width: `${scoreWidthPct}%` }}
              />
            </div>
            <span className="w-12 text-right font-mono text-xs font-semibold text-accent">
              {passage.score.toFixed(2)}
            </span>
          </div>
        </div>
        <button
          type="button"
          aria-expanded={expanded}
          aria-label={
            expanded
              ? `Collapse passage from ${passage.paper_id}`
              : `Expand passage from ${passage.paper_id}`
          }
          onClick={(e) => {
            e.stopPropagation();
            setExpanded((open) => !open);
          }}
          className={`flex h-10 min-w-[2.75rem] shrink-0 items-center justify-center rounded-lg border px-2 font-mono text-xs font-semibold uppercase tracking-wide transition-colors ${
            expanded
              ? "border-accent/50 bg-accent/15 text-accent"
              : "border-slide-border bg-slide-bg text-slide-muted hover:border-accent/40 hover:text-accent"
          }`}
        >
          {expanded ? "−" : "+"}
        </button>
      </div>
      <p
        className={`whitespace-pre-wrap text-sm leading-relaxed text-slide-body ${
          expanded ? "max-h-[min(18rem,40vh)] overflow-y-auto pr-1" : "line-clamp-2"
        }`}
      >
        {passage.text}
      </p>
    </article>
  );
});
