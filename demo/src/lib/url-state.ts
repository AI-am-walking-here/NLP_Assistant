import type { DemoFeatureFlags } from "./feature-flags";

export type UrlDemoState = {
  compareA?: string;
  compareB?: string;
  example?: string;
  present?: boolean;
  system?: string;
};

export function readUrlState(): UrlDemoState {
  if (typeof window === "undefined") return {};
  const params = new URLSearchParams(window.location.search);
  const compare = params.get("compare");
  const [compareA, compareB] = compare?.split(",").map((s) => s.trim()) ?? [];
  return {
    compareA: compareA || undefined,
    compareB: compareB || undefined,
    example: params.get("example") ?? undefined,
    present: params.get("present") === "1" || params.get("mode") === "present",
    system: params.get("system") ?? undefined,
  };
}

export function applyUrlToFlags(
  flags: DemoFeatureFlags,
  url: UrlDemoState,
): DemoFeatureFlags {
  const next = { ...flags };
  if (url.present) next.presentationMode = true;
  if (url.compareA && url.compareB) next.abCompare = true;
  if (url.example) next.evalShowcase = true;
  return next;
}

export function buildShareUrl(opts: {
  compareA?: string;
  compareB?: string;
  example?: string;
  present?: boolean;
}): string {
  if (typeof window === "undefined") return "";
  const params = new URLSearchParams();
  if (opts.compareA && opts.compareB) {
    params.set("compare", `${opts.compareA},${opts.compareB}`);
  }
  if (opts.example) params.set("example", opts.example);
  if (opts.present) params.set("present", "1");
  const qs = params.toString();
  return `${window.location.origin}${window.location.pathname}${qs ? `?${qs}` : ""}`;
}
