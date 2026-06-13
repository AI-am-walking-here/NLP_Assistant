"use client";

import { memo } from "react";
import {
  normalizePaperId,
  styleForPaper,
  type PaperTagStyle,
} from "@/lib/paper-colors";

type Props = {
  paperId: string;
  colorMap: Map<string, number>;
  className?: string;
  style?: PaperTagStyle;
};

export const PaperIdTag = memo(function PaperIdTag({
  paperId,
  colorMap,
  className = "",
  style,
}: Props) {
  const id = normalizePaperId(paperId);
  const tagStyle = style ?? styleForPaper(id, colorMap);

  return (
    <span
      className={`inline-flex items-center rounded-md border px-1.5 py-0.5 font-mono text-xs font-semibold leading-none ${tagStyle.bg} ${tagStyle.border} ${tagStyle.text} ${className}`}
      title={`Paper ${id}`}
    >
      {id}
    </span>
  );
});
