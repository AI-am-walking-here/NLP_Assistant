"use client";

import { memo, useMemo } from "react";
import type { Passage } from "@/lib/types";
import { buildPaperColorMap } from "@/lib/paper-colors";
import { splitAbstractForHighlight } from "@/lib/passage-highlight";
import { PaperIdTag } from "./PaperIdTag";

type Props = {
  content: string;
  passages: Passage[];
  highlightTokens?: Set<string>;
};

const CITE_SPLIT_RE = /(\(\d{4}\.\d{4,5}\))/g;

export const AbstractContent = memo(function AbstractContent({
  content,
  passages,
  highlightTokens,
}: Props) {
  const colorMap = useMemo(
    () => buildPaperColorMap(passages, content),
    [passages, content],
  );

  const parts = content.split(CITE_SPLIT_RE);
  const useTokenHighlight = highlightTokens && highlightTokens.size > 0;

  return (
    <p className="whitespace-pre-wrap p-4 text-sm leading-relaxed text-slide-ink sm:text-base">
      {parts.map((part, i) => {
        const match = part.match(/^\((\d{4}\.\d{4,5})\)$/);
        if (match) {
          return (
            <PaperIdTag
              key={`${match[1]}-${i}`}
              paperId={match[1]}
              colorMap={colorMap}
              className="mx-0.5 align-baseline text-sm"
            />
          );
        }
        if (useTokenHighlight) {
          return (
            <span key={i}>
              {splitAbstractForHighlight(part, highlightTokens!).map((seg, j) =>
                seg.highlight ? (
                  <mark
                    key={j}
                    className="rounded bg-accent/25 px-0.5 text-slide-ink"
                  >
                    {seg.text}
                  </mark>
                ) : (
                  <span key={j}>{seg.text}</span>
                ),
              )}
            </span>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </p>
  );
});
