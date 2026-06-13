"use client";

import type { VerifyResponse } from "@/lib/types";
import { FactScoreGauge } from "./FactScoreGauge";

type Props = {
  result: VerifyResponse | null;
  loading: boolean;
  error: string | null;
  onScore: () => void;
  disabled?: boolean;
};

const LABEL_COLORS: Record<string, string> = {
  yes: "text-emerald-400",
  partial: "text-amber-300",
  no: "text-red-400",
};

export function LiveFactScorePanel({
  result,
  loading,
  error,
  onScore,
  disabled,
}: Props) {
  return (
    <div className="mt-2 shrink-0 rounded-lg border border-slide-border bg-slide-bg p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="section-kicker text-xs">Live FActScore</span>
        <button
          type="button"
          onClick={onScore}
          disabled={disabled || loading}
          className="btn-secondary px-3 py-1.5 text-xs"
        >
          {loading ? "Scoring…" : "Score abstract"}
        </button>
      </div>
      {error ? (
        <p className="text-xs text-red-300">{error}</p>
      ) : result ? (
        <div className="space-y-2">
          <FactScoreGauge score={result.factscore} label="Live" />
          <p className="font-mono text-[10px] text-slide-muted">
            {result.n_claims} claims · verifier: {result.verifier}
          </p>
          <div className="scroll-panel max-h-32 space-y-1.5">
            {result.details.map((d, i) => (
              <div
                key={i}
                className="rounded border border-slide-border bg-slide-elevated px-2 py-1.5 text-xs"
              >
                <span
                  className={`font-mono font-semibold uppercase ${LABEL_COLORS[d.supported] ?? "text-slide-muted"}`}
                >
                  {d.supported}
                </span>
                <span className="ml-2 text-slide-body">{d.claim}</span>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <p className="text-xs text-slide-muted">
          Uses 70B verifier when available; falls back to lexical mock.
        </p>
      )}
    </div>
  );
}
