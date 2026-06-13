export type PipelineModule = "vector" | "graph" | "rank" | "sft";

export const MODULE_ORDER: PipelineModule[] = ["vector", "graph", "rank", "sft"];

export const MODULE_LABELS: Record<PipelineModule, string> = {
  vector: "BGE",
  graph: "Graph",
  rank: "RankRAG",
  sft: "SFT",
};

const SYSTEM_MODULES: Record<string, PipelineModule[]> = {
  zero_shot: [],
  zero_shot_with_sft: ["sft"],
  naive_rag: ["vector"],
  naive_rag_with_sft: ["vector", "sft"],
  graph_only: ["graph"],
  rankrag_only: ["vector", "rank"],
  full: ["vector", "graph", "rank", "sft"],
  full_minus_sft: ["vector", "graph", "rank"],
  full_minus_graph: ["vector", "rank", "sft"],
  full_minus_rerank: ["vector", "graph", "sft"],
  naive_rag_sft_prompt: ["vector", "sft"],
};

export function modulesForSystem(systemId: string): PipelineModule[] {
  return SYSTEM_MODULES[systemId] ?? [];
}

export function factScorePct(score: number | null, max = 0.6): number {
  if (score == null) return 0;
  return Math.min(100, Math.round((score / max) * 100));
}
