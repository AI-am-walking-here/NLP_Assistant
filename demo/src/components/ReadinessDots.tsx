"use client";

type Props = {
  titleOk: boolean;
  outlineOk: boolean;
};

function Dot({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className="flex items-center gap-1.5" title={label}>
      <span
        className={`h-2.5 w-2.5 rounded-full border-2 ${
          ok
            ? "border-grounded-success bg-grounded-success/40"
            : "border-slide-border bg-slide-bg"
        }`}
      />
      <span className={`text-xs ${ok ? "text-slide-body" : "text-slide-muted"}`}>
        {label}
      </span>
    </span>
  );
}

export function ReadinessDots({ titleOk, outlineOk }: Props) {
  return (
    <div className="flex items-center gap-3">
      <Dot ok={titleOk} label="Title" />
      <Dot ok={outlineOk} label="Outline" />
    </div>
  );
}
