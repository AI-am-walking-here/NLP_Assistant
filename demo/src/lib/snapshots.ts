import type { GenerateResponse } from "./types";

export type DemoSnapshot = {
  id: string;
  savedAt: string;
  title: string;
  outline: string;
  system: string;
  topK: number;
  result: GenerateResponse;
  label?: string;
};

const STORAGE_KEY = "nils-jens-snapshots";
const MAX_SNAPSHOTS = 8;

export function loadSnapshots(): DemoSnapshot[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as DemoSnapshot[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveSnapshot(entry: Omit<DemoSnapshot, "id" | "savedAt">): DemoSnapshot[] {
  const snap: DemoSnapshot = {
    ...entry,
    id: crypto.randomUUID(),
    savedAt: new Date().toISOString(),
  };
  const all = [snap, ...loadSnapshots()].slice(0, MAX_SNAPSHOTS);
  if (typeof window !== "undefined") {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
  }
  return all;
}

export function deleteSnapshot(id: string): DemoSnapshot[] {
  const all = loadSnapshots().filter((s) => s.id !== id);
  if (typeof window !== "undefined") {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
  }
  return all;
}
