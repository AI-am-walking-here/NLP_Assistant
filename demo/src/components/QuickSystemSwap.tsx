"use client";

const SWAP_TARGETS = [
  { id: "zero_shot", label: "Zero-shot" },
  { id: "full_minus_sft", label: "Champion" },
  { id: "full", label: "+ SFT" },
  { id: "rankrag_only", label: "RankRAG" },
  { id: "naive_rag", label: "Naive RAG" },
];

type Props = {
  current: string;
  disabled?: boolean;
  onSwap: (systemId: string) => void;
};

export function QuickSystemSwap({ current, disabled, onSwap }: Props) {
  return (
    <div className="mt-2 flex flex-wrap items-center gap-2">
      <span className="font-mono text-[10px] uppercase text-slide-muted">
        Same prompt:
      </span>
      {SWAP_TARGETS.map((t) => (
        <button
          key={t.id}
          type="button"
          disabled={disabled || t.id === current}
          onClick={() => onSwap(t.id)}
          className={`rounded-full border px-2.5 py-1 font-mono text-[10px] uppercase tracking-wide transition-colors ${
            t.id === current
              ? "border-accent/50 bg-accent/15 text-accent"
              : "border-slide-border bg-slide-bg text-slide-muted hover:border-accent/40 hover:text-accent disabled:opacity-40"
          }`}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
