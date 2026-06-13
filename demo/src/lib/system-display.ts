import type { SystemInfo } from "./types";

export type SystemDisplay = {
  title: string;
  subtitle: string;
  badge?: string;
};

export const GROUP_LABELS: Record<string, string> = {
  full: "Recommended",
  retrieval: "Retrieval",
  baselines: "Baselines",
  ablation: "Ablations",
  other: "Advanced",
};

export const GROUP_ORDER = ["full", "retrieval", "baselines", "ablation", "other"];

const DISPLAY_OVERRIDES: Record<string, SystemDisplay> = {
  full_minus_sft: {
    title: "Full pipeline",
    subtitle: "Vector + Graph + RankRAG → 8B (best factual accuracy)",
    badge: "Top pick",
  },
  full: {
    title: "Full pipeline + SFT writer",
    subtitle: "All modules including domain-tuned LoRA",
  },
  zero_shot: {
    title: "Base 8B only",
    subtitle: "No retrieval — fastest, lowest grounding",
  },
  zero_shot_with_sft: {
    title: "SFT writer only",
    subtitle: "Domain-tuned 8B without retrieval",
  },
  naive_rag: {
    title: "Vector RAG",
    subtitle: "BGE retrieval + base 8B — fastest real inference",
    badge: "Quick",
  },
  naive_rag_with_sft: {
    title: "Vector RAG + SFT",
    subtitle: "BGE retrieval + domain-tuned writer",
  },
  graph_only: {
    title: "Graph RAG",
    subtitle: "Community graph retrieval + base 8B",
  },
  rankrag_only: {
    title: "RankRAG",
    subtitle: "Vector search + reranker + 8B — faster live demo",
    badge: "Quick",
  },
  full_minus_graph: {
    title: "Full pipeline · no graph",
    subtitle: "Vector + RankRAG + SFT (graph disabled)",
  },
  full_minus_rerank: {
    title: "Full pipeline · no rerank",
    subtitle: "Vector + graph + SFT (lexical fallback)",
  },
  naive_rag_sft_prompt: {
    title: "Vector RAG + prompt tuning",
    subtitle: "Retrieval with SFT-style prompts (research baseline)",
  },
};

export function displayForSystem(sys: SystemInfo): SystemDisplay {
  if (DISPLAY_OVERRIDES[sys.id]) {
    return DISPLAY_OVERRIDES[sys.id];
  }
  return {
    title: humanizeLabel(sys.label || sys.id),
    subtitle: sys.description || "Configured pipeline variant",
  };
}

function humanizeLabel(label: string): string {
  return label
    .replace(/_/g, " ")
    .replace(/\s*−\s*/g, " · ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function groupSystems(
  systems: SystemInfo[],
): { group: string; label: string; items: SystemInfo[] }[] {
  const grouped = systems.reduce<Record<string, SystemInfo[]>>((acc, sys) => {
    const key = sys.group || "other";
    acc[key] = acc[key] ?? [];
    acc[key].push(sys);
    return acc;
  }, {});

  return GROUP_ORDER.filter((g) => grouped[g]?.length).map((group) => ({
    group,
    label: GROUP_LABELS[group] ?? group,
    items: [...grouped[group]].sort(
      (a, b) => (b.factscore_mean ?? -1) - (a.factscore_mean ?? -1),
    ),
  }));
}
