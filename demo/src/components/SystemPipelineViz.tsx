"use client";

import {
  MODULE_ORDER,
  modulesForSystem,
} from "@/lib/system-capabilities";
import { ModuleNode } from "./visual/ModuleNode";

type Props = {
  systemId: string;
  compact?: boolean;
};

function Connector({ lit, compact }: { lit: boolean; compact?: boolean }) {
  return (
    <div
      className={`shrink-0 rounded-full ${
        compact ? "mx-0.5 h-0.5 w-2" : "mx-0.5 mb-4 h-0.5 w-4 sm:w-5"
      } ${lit ? "bg-accent/60" : "bg-slide-border"}`}
    />
  );
}

export function SystemPipelineViz({ systemId, compact = false }: Props) {
  const active = new Set(modulesForSystem(systemId));
  const anyActive = active.size > 0;

  if (compact) {
    return (
      <div
        className="flex items-center gap-0.5"
        title="Active pipeline modules"
      >
        {MODULE_ORDER.map((mod, i) => {
          const next = MODULE_ORDER[i + 1];
          const linkLit =
            next != null
              ? active.has(mod) && active.has(next)
              : active.has(mod) && anyActive;
          return (
            <div key={mod} className="flex items-center">
              <ModuleNode module={mod} active={active.has(mod)} compact />
              <Connector lit={linkLit} compact />
            </div>
          );
        })}
        <div
          className="flex h-8 w-8 items-center justify-center rounded-full border border-accent/50 bg-accent/10 font-mono text-[10px] text-accent"
          title="8B generator"
        >
          8B
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-slide-border bg-slide-bg px-2 py-3">
      <div className="flex items-center justify-center">
        {MODULE_ORDER.map((mod, i) => {
          const next = MODULE_ORDER[i + 1];
          const linkLit =
            next != null
              ? active.has(mod) && active.has(next)
              : active.has(mod) && anyActive;
          return (
            <div key={mod} className="flex items-center">
              <ModuleNode module={mod} active={active.has(mod)} />
              <Connector lit={linkLit} />
            </div>
          );
        })}
        <div className="flex flex-col items-center gap-1">
          <div className="flex h-10 w-10 items-center justify-center rounded-full border-2 border-accent/50 bg-accent/10 font-mono text-sm text-accent">
            8B
          </div>
          <span className="font-mono text-[10px] uppercase tracking-wide text-accent">
            Gen
          </span>
        </div>
      </div>
    </div>
  );
}
