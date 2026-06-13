"use client";

type Props = {
  value: number;
  min?: number;
  max?: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  compact?: boolean;
};

export function TopKSlider({
  value,
  min = 1,
  max = 20,
  onChange,
  disabled,
  compact = false,
}: Props) {
  if (compact) {
    return (
      <div className="flex flex-1 items-center gap-3">
        <span className="shrink-0 font-mono text-xs uppercase tracking-wide text-slide-muted">
          Top-k
        </span>
        <input
          type="range"
          min={min}
          max={max}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(Number(e.target.value))}
          className="range-slider min-w-0 flex-1"
        />
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-accent/40 bg-accent/10 font-mono text-sm font-semibold text-accent">
          {value}
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="font-mono text-xs uppercase tracking-wide text-slide-muted">
          Top-k
        </span>
        <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-accent/40 bg-accent/10 font-mono text-sm font-semibold text-accent">
          {value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="range-slider w-full"
      />
      <div className="flex justify-between px-0.5">
        {[min, 5, 10, 15, max].map((tick) => (
          <span key={tick} className="font-mono text-[10px] text-slide-muted">
            {tick}
          </span>
        ))}
      </div>
    </div>
  );
}
