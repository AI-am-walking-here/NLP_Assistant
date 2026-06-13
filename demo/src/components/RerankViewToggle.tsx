"use client";

type Props = {
  mode: "post" | "pre";
  hasPre: boolean;
  onChange: (mode: "post" | "pre") => void;
};

export function RerankViewToggle({ mode, hasPre, onChange }: Props) {
  if (!hasPre) return null;

  return (
    <div className="mb-2 flex gap-1 rounded-lg border border-slide-border bg-slide-bg p-1">
      <button
        type="button"
        onClick={() => onChange("post")}
        className={`flex-1 rounded-md px-2 py-1 font-mono text-[10px] uppercase ${
          mode === "post"
            ? "bg-accent/15 text-accent"
            : "text-slide-muted hover:text-slide-ink"
        }`}
      >
        After RankRAG
      </button>
      <button
        type="button"
        onClick={() => onChange("pre")}
        className={`flex-1 rounded-md px-2 py-1 font-mono text-[10px] uppercase ${
          mode === "pre"
            ? "bg-accent/15 text-accent"
            : "text-slide-muted hover:text-slide-ink"
        }`}
      >
        Before RankRAG
      </button>
    </div>
  );
}
