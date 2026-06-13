"use client";

type Props = {
  step: "compose" | "retrieve" | "generate";
};

const STEPS = [
  { id: "compose" as const, icon: "✎", label: "Compose" },
  { id: "retrieve" as const, icon: "◎", label: "Retrieve" },
  { id: "generate" as const, icon: "✦", label: "Generate" },
];

export function PipelineStrip({ step }: Props) {
  const activeIdx = STEPS.findIndex((s) => s.id === step);

  return (
    <div className="hidden shrink-0 items-center md:flex">
      {STEPS.map((s, i) => {
        const done = i < activeIdx;
        const active = i === activeIdx;
        return (
          <div key={s.id} className="flex items-center">
            <div className="flex flex-col items-center gap-1">
              <div
                className={`flex h-11 w-11 items-center justify-center rounded-full border-2 text-base transition-colors ${
                  active
                    ? "border-accent bg-accent/15 text-accent shadow-[0_0_14px_rgba(79,209,237,0.2)]"
                    : done
                      ? "border-accent/50 bg-accent/5 text-accent"
                      : "border-slide-border bg-slide-surface text-slide-muted"
                }`}
                title={s.label}
              >
                {s.icon}
              </div>
              <span
                className={`font-mono text-[10px] uppercase tracking-wide ${
                  active || done ? "text-accent" : "text-slide-muted"
                }`}
              >
                {s.label}
              </span>
            </div>
            {i < STEPS.length - 1 && (
              <div
                className={`mx-2 mb-4 h-0.5 w-8 rounded-full sm:w-12 ${
                  i < activeIdx ? "bg-accent/70" : "bg-slide-border"
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
