"use client";

import type { HealthResponse } from "@/lib/types";

type Props = {
  health: HealthResponse | null;
  bootError: string | null;
  apiReady: boolean;
};

function Badge({
  active,
  label,
  icon,
  title,
}: {
  active: boolean;
  label: string;
  icon: string;
  title?: string;
}) {
  return (
    <span
      title={title ?? label}
      className={`inline-flex h-8 min-w-[2rem] items-center justify-center gap-1 rounded-lg border px-2 font-mono text-xs ${
        active
          ? "border-accent/40 bg-accent/10 text-accent"
          : "border-slide-border bg-slide-bg text-slide-muted"
      }`}
    >
      <span>{icon}</span>
      <span className="hidden lg:inline">{label}</span>
    </span>
  );
}

export function StatusIndicator({ health, bootError, apiReady }: Props) {
  const loading =
    health?.status === "loading" || health?.stack_ready === false;
  const mock = health?.mock_mode === true;
  const fast = health?.demo_fast === true;
  const gpu = health?.gpu_mode;

  const dotClass = bootError
    ? "bg-red-400"
    : apiReady
      ? "bg-grounded-success"
      : loading
        ? "bg-grounded-warn"
        : "bg-slide-muted";

  return (
    <div
      className="status-chip shrink-0"
      title={
        bootError ??
        (loading ? health?.error ?? "Stack loading" : "API ready")
      }
    >
      <span className={`h-3 w-3 shrink-0 rounded-full ${dotClass}`} />
      <div className="flex items-center gap-1.5">
        <Badge
          active={apiReady && !mock}
          icon="8B"
          label="Live"
          title={mock ? "Mock generation" : "Real 8B models"}
        />
        {mock && <Badge active icon="◇" label="Mock" title="Mock mode" />}
        {fast && !mock && (
          <Badge active icon="⚡" label="Fast" title="Demo fast mode: smaller retrieval + shorter generation" />
        )}
        {gpu && (
          <Badge
            active
            icon="⬢"
            label={gpu === "parallel" ? "GPU∥" : "GPU"}
            title={health?.gpu_layout ?? gpu}
          />
        )}
        {loading && (
          <Badge active icon="…" label="Load" title="Stack preloading" />
        )}
      </div>
    </div>
  );
}
