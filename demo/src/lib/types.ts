export type SystemInfo = {
  id: string;
  label: string;
  description: string;
  group: string;
  factscore_mean: number | null;
};

export type Passage = {
  paper_id: string;
  section_heading: string;
  score: number;
  text: string;
};

export type PipelineStage = {
  id: string;
  label: string;
  detail: string;
  count?: number | null;
};

export type GenerateResponse = {
  abstract: string;
  passages: Passage[];
  passages_pre_rerank?: Passage[];
  stages?: PipelineStage[];
  mock_generation: boolean;
  mock_rerank: boolean;
  backend: string;
  system: string;
};

export type ClaimDetail = {
  claim: string;
  supported: string;
  reasoning: string;
};

export type VerifyResponse = {
  factscore: number;
  n_claims: number;
  labels: string[];
  details: ClaimDetail[];
  verifier: string;
};

export type HealthResponse = {
  status: string;
  mock_mode: boolean;
  demo_fast?: boolean;
  stack_ready?: boolean;
  sft_adapter?: string | null;
  rank_adapter?: string | null;
  cuda_visible_devices?: string;
  embed_device?: string;
  gpu_mode?: "parallel" | "sequential";
  gpu_layout?: string;
  error?: string;
};

export type GenerateRequest = {
  title: string;
  outline: string;
  system: string;
  top_k?: number;
};
