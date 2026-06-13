"use client";

import type { OutlineQuality } from "@/lib/outline-quality";

type Props = {
  quality: OutlineQuality | null;
};

const STYLES: Record<OutlineQuality["level"], string> = {
  good: "border-emerald-400/30 bg-emerald-400/5 text-emerald-300",
  warn: "border-amber-400/30 bg-amber-400/5 text-amber-200",
  poor: "border-red-400/30 bg-red-400/5 text-red-300",
};

export function OutlineGuardrail({ quality }: Props) {
  if (!quality || quality.level === "good") return null;

  return (
    <p
      className={`rounded-lg border px-3 py-2 text-xs ${STYLES[quality.level]}`}
    >
      {quality.message}
    </p>
  );
}
