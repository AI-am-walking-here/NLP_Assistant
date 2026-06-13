"use client";

import type { SystemInfo } from "@/lib/types";
import { FactScoreGauge } from "./FactScoreGauge";
import { SystemPicker } from "./SystemPicker";
import { SystemPipelineViz } from "./SystemPipelineViz";

type Props = {
  systems: SystemInfo[];
  value: string;
  onChange: (id: string) => void;
  disabled?: boolean;
};

export function SystemSelector({ systems, value, onChange, disabled }: Props) {
  const selected = systems.find((s) => s.id === value);

  if (systems.length === 0) {
    return (
      <div className="shrink-0 space-y-2">
        <p className="section-kicker">System</p>
        <div className="h-11 rounded-lg border border-slide-border bg-slide-elevated" />
      </div>
    );
  }

  return (
    <div className="shrink-0 space-y-2">
      <p className="section-kicker">System</p>
      <SystemPicker
        systems={systems}
        value={value}
        onChange={onChange}
        disabled={disabled}
      />
      <div className="flex items-center gap-3 rounded-lg border border-slide-border bg-slide-bg px-3 py-2">
        <SystemPipelineViz systemId={value} compact />
        <FactScoreGauge score={selected?.factscore_mean ?? null} compact />
      </div>
    </div>
  );
}
