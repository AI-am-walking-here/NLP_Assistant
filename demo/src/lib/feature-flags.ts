export type DemoFeatureId =
  | "abCompare"
  | "guidedTour"
  | "evalShowcase"
  | "goldReveal"
  | "quickSystemSwap"
  | "liveFactScore"
  | "passageLinking"
  | "rerankToggle"
  | "snapshots"
  | "presentationMode"
  | "outlineGuardrails"
  | "qrCode";

export type DemoFeatureFlags = Record<DemoFeatureId, boolean>;

export const FEATURE_LABELS: Record<
  DemoFeatureId,
  { label: string; description: string }
> = {
  abCompare: {
    label: "A/B compare",
    description: "Side-by-side run of two systems on the same prompt",
  },
  guidedTour: {
    label: "Guided 3-act tour",
    description: "Scripted zero-shot → champion → SFT plot twist",
  },
  evalShowcase: {
    label: "Eval challenges",
    description: "Pick held-out eval prompts with gold abstracts",
  },
  goldReveal: {
    label: "Gold abstract reveal",
    description: "Show human-written holdout abstract after generation",
  },
  quickSystemSwap: {
    label: "Quick system swap",
    description: "Re-run same prompt under another system",
  },
  liveFactScore: {
    label: "Live FActScore",
    description: "Score generated abstract with 70B verifier",
  },
  passageLinking: {
    label: "Passage ↔ abstract link",
    description: "Click a passage to highlight related abstract spans",
  },
  rerankToggle: {
    label: "Pre/post RankRAG",
    description: "Toggle passage list before vs after reranking",
  },
  snapshots: {
    label: "Snapshot replay",
    description: "Save and replay recent successful runs",
  },
  presentationMode: {
    label: "Presentation layout",
    description: "Larger text, fewer controls, fullscreen-friendly",
  },
  outlineGuardrails: {
    label: "Outline guardrails",
    description: "Warn when outline may retrieve poorly",
  },
  qrCode: {
    label: "QR code",
    description: "Share demo URL for Q&A audience",
  },
};

export const DEFAULT_FEATURE_FLAGS: DemoFeatureFlags = {
  abCompare: false,
  guidedTour: false,
  evalShowcase: false,
  goldReveal: false,
  quickSystemSwap: false,
  liveFactScore: false,
  passageLinking: false,
  rerankToggle: false,
  snapshots: false,
  presentationMode: false,
  outlineGuardrails: false,
  qrCode: false,
};

const STORAGE_KEY = "nils-jens-feature-flags";

export function loadFeatureFlags(): DemoFeatureFlags {
  if (typeof window === "undefined") {
    return { ...DEFAULT_FEATURE_FLAGS };
  }
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_FEATURE_FLAGS };
    const parsed = JSON.parse(raw) as Partial<DemoFeatureFlags>;
    return { ...DEFAULT_FEATURE_FLAGS, ...parsed };
  } catch {
    return { ...DEFAULT_FEATURE_FLAGS };
  }
}

export function saveFeatureFlags(flags: DemoFeatureFlags): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(flags));
}
