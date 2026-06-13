"use client";

type Props = {
  goldAbstract: string;
  open: boolean;
  onToggle: () => void;
};

export function GoldAbstractReveal({ goldAbstract, open, onToggle }: Props) {
  if (!goldAbstract) return null;

  return (
    <div className="mt-2 shrink-0 rounded-lg border border-amber-400/30 bg-amber-400/5">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center justify-between px-3 py-2 text-left"
      >
        <span className="font-mono text-xs uppercase tracking-wide text-amber-300">
          Gold holdout abstract
        </span>
        <span className="font-mono text-xs text-amber-300">{open ? "−" : "+"}</span>
      </button>
      {open ? (
        <p className="scroll-panel max-h-40 border-t border-amber-400/20 px-3 py-2 text-sm leading-relaxed text-slide-body">
          {goldAbstract}
        </p>
      ) : null}
    </div>
  );
}
