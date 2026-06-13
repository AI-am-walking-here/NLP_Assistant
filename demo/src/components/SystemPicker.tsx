"use client";

import { useEffect, useRef, useState } from "react";
import type { SystemInfo } from "@/lib/types";
import { displayForSystem, groupSystems } from "@/lib/system-display";
import { SystemOptionRow } from "./SystemOptionRow";

type Props = {
  systems: SystemInfo[];
  value: string;
  onChange: (id: string) => void;
  disabled?: boolean;
};

export function SystemPicker({ systems, value, onChange, disabled }: Props) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const selected = systems.find((s) => s.id === value);
  const display = selected ? displayForSystem(selected) : null;
  const groups = groupSystems(systems);

  useEffect(() => {
    if (!open) return;
    function onPointerDown(e: MouseEvent) {
      if (!rootRef.current?.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return (
    <div
      ref={rootRef}
      className={`relative ${open ? "z-50" : ""}`}
    >
      <div className={open ? "system-picker-open" : ""}>
        <button
          type="button"
          disabled={disabled}
          aria-haspopup="listbox"
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
          className={
            open
              ? "flex min-h-[44px] w-full items-center justify-between gap-3 rounded-none border-0 bg-transparent px-4 py-3 text-left focus:ring-0"
              : "input-field flex w-full items-center justify-between gap-3 text-left"
          }
        >
          <span className="min-w-0">
            <span className="block truncate text-base font-medium text-slide-ink">
              {display?.title ?? "Select system"}
            </span>
            {display?.subtitle && (
              <span className="block truncate text-sm text-slide-muted">
                {display.subtitle}
              </span>
            )}
          </span>
          <span
            className={`shrink-0 text-slide-muted transition-transform ${
              open ? "rotate-180" : ""
            }`}
          >
            ▾
          </span>
        </button>

        {open && (
          <div
            role="listbox"
            className="scroll-panel max-h-[min(22rem,50vh)] overflow-y-auto border-t border-slide-border p-2"
          >
            {groups.map(({ group, label, items }) => (
              <div key={group} className="mb-2 last:mb-0">
                <p className="sticky top-0 z-10 bg-slide-elevated px-2 py-2 font-mono text-xs font-semibold uppercase tracking-[0.16em] text-accent">
                  {label}
                </p>
                <div className="space-y-1">
                  {items.map((sys) => (
                    <SystemOptionRow
                      key={sys.id}
                      system={sys}
                      selected={sys.id === value}
                      onSelect={() => {
                        onChange(sys.id);
                        setOpen(false);
                      }}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
