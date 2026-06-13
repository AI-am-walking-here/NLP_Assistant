"use client";

type Props = {
  mock?: boolean;
  fast?: boolean;
};

const STEPS = [
  { id: "retrieve", icon: "◎" },
  { id: "rerank", icon: "⇅" },
  { id: "generate", icon: "✦" },
];

export function LoadingVisualizer({ mock, fast }: Props) {
  const activeSteps = fast ? 1 : mock ? 2 : 3;

  return (
    <div className="flex h-full flex-col items-center justify-center gap-8 p-6">
      <div className="relative h-20 w-20">
        <div className="absolute inset-0 rounded-full border-2 border-accent/20" />
        <div className="absolute inset-0 animate-spin rounded-full border-2 border-transparent border-t-accent" />
        <div className="absolute inset-3 rounded-full bg-accent/10" />
        <div className="absolute inset-0 flex items-center justify-center font-mono text-sm font-semibold text-accent">
          8B
        </div>
      </div>
      <div className="flex items-center gap-2">
        {STEPS.slice(0, activeSteps).map((step, i) => (
          <div key={step.id} className="flex items-center gap-2">
            <div
              className={`flex h-11 w-11 items-center justify-center rounded-full border-2 text-base ${
                i === activeSteps - 1
                  ? "border-accent bg-accent/15 text-accent"
                  : "border-accent/40 bg-accent/5 text-accent/80"
              }`}
            >
              {step.icon}
            </div>
            {i < activeSteps - 1 && (
              <div className="h-0.5 w-6 rounded-full bg-accent/50" />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
