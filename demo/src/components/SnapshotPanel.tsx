"use client";

import type { DemoSnapshot } from "@/lib/snapshots";

type Props = {
  snapshots: DemoSnapshot[];
  onLoad: (snap: DemoSnapshot) => void;
  onDelete: (id: string) => void;
  onSave: () => void;
  canSave: boolean;
};

export function SnapshotPanel({
  snapshots,
  onLoad,
  onDelete,
  onSave,
  canSave,
}: Props) {
  return (
    <div className="mb-2 shrink-0 rounded-lg border border-slide-border bg-slide-elevated px-3 py-2">
      <div className="mb-2 flex items-center justify-between">
        <span className="section-kicker text-xs">Snapshots</span>
        <button
          type="button"
          onClick={onSave}
          disabled={!canSave}
          className="btn-secondary px-2 py-1 text-[10px]"
        >
          Save run
        </button>
      </div>
      {snapshots.length === 0 ? (
        <p className="text-xs text-slide-muted">No saved runs yet.</p>
      ) : (
        <div className="flex flex-wrap gap-2">
          {snapshots.map((s) => (
            <div
              key={s.id}
              className="flex items-center gap-1 rounded-full border border-slide-border bg-slide-bg pl-2"
            >
              <button
                type="button"
                onClick={() => onLoad(s)}
                className="py-1 font-mono text-[10px] text-accent hover:underline"
                title={s.title}
              >
                {s.system} · {new Date(s.savedAt).toLocaleTimeString()}
              </button>
              <button
                type="button"
                onClick={() => onDelete(s.id)}
                className="px-2 py-1 text-[10px] text-slide-muted hover:text-red-300"
                aria-label="Delete snapshot"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
