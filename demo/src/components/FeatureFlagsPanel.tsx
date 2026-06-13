"use client";

import {
  DEFAULT_FEATURE_FLAGS,
  FEATURE_LABELS,
  type DemoFeatureFlags,
  type DemoFeatureId,
  saveFeatureFlags,
} from "@/lib/feature-flags";

type Props = {
  flags: DemoFeatureFlags;
  onChange: (flags: DemoFeatureFlags) => void;
  open: boolean;
  onClose: () => void;
};

const ORDER: DemoFeatureId[] = [
  "abCompare",
  "guidedTour",
  "evalShowcase",
  "goldReveal",
  "quickSystemSwap",
  "liveFactScore",
  "passageLinking",
  "rerankToggle",
  "snapshots",
  "presentationMode",
  "outlineGuardrails",
  "qrCode",
];

export function FeatureFlagsPanel({ flags, onChange, open, onClose }: Props) {
  if (!open) return null;

  function toggle(id: DemoFeatureId) {
    const next = { ...flags, [id]: !flags[id] };
    onChange(next);
    saveFeatureFlags(next);
  }

  function reset() {
    onChange({ ...DEFAULT_FEATURE_FLAGS });
    saveFeatureFlags({ ...DEFAULT_FEATURE_FLAGS });
  }

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-end bg-black/50 p-4">
      <div className="slide-panel-elevated flex max-h-[90dvh] w-full max-w-md flex-col overflow-hidden shadow-2xl">
        <div className="flex items-center justify-between border-b border-slide-border px-4 py-3">
          <div>
            <h2 className="font-display text-lg font-semibold text-slide-ink">
              Demo features
            </h2>
            <p className="text-xs text-slide-muted">
              Pipeline stage progress is always on.
            </p>
          </div>
          <button type="button" onClick={onClose} className="btn-secondary px-3 py-1 text-xs">
            Close
          </button>
        </div>
        <div className="scroll-panel flex-1 space-y-2 p-4">
          {ORDER.map((id) => {
            const meta = FEATURE_LABELS[id];
            return (
              <label
                key={id}
                className="flex cursor-pointer items-start gap-3 rounded-lg border border-slide-border bg-slide-bg px-3 py-2.5 hover:border-accent/40"
              >
                <input
                  type="checkbox"
                  checked={flags[id]}
                  onChange={() => toggle(id)}
                  className="mt-1 h-4 w-4 accent-accent"
                />
                <span>
                  <span className="block text-sm font-medium text-slide-ink">
                    {meta.label}
                  </span>
                  <span className="block text-xs text-slide-muted">
                    {meta.description}
                  </span>
                </span>
              </label>
            );
          })}
        </div>
        <div className="border-t border-slide-border p-3">
          <button type="button" onClick={reset} className="btn-secondary w-full text-xs">
            Reset all to off
          </button>
        </div>
      </div>
    </div>
  );
}
