"use client";

import type { EvalShowcasePrompt } from "@/lib/eval-showcase";

type Props = {
  prompts: EvalShowcasePrompt[];
  value: string;
  onChange: (prompt: EvalShowcasePrompt) => void;
  disabled?: boolean;
};

export function EvalShowcasePicker({
  prompts,
  value,
  onChange,
  disabled,
}: Props) {
  if (prompts.length === 0) return null;

  return (
    <div className="space-y-1">
      <label className="section-kicker">Eval challenge</label>
      <select
        value={value}
        disabled={disabled}
        onChange={(e) => {
          const picked = prompts.find((p) => p.id === e.target.value);
          if (picked) onChange(picked);
        }}
        className="input-field py-2 text-sm"
      >
        <option value="">— pick held-out prompt —</option>
        {prompts.map((p) => (
          <option key={p.id} value={p.id}>
            {p.label} ({p.id})
          </option>
        ))}
      </select>
    </div>
  );
}
