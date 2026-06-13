export type GuidedAct = {
  id: string;
  title: string;
  caption: string;
  system: string;
  exampleId: string;
};

export const GUIDED_ACTS: GuidedAct[] = [
  {
    id: "act-1",
    title: "Act 1 — Zero-shot",
    caption:
      "No retrieval: fast, vague abstract. FActScore looks high but there is no evidence sidebar.",
    system: "zero_shot",
    exampleId: "nils-jens-demo",
  },
  {
    id: "act-2",
    title: "Act 2 — Retrieval champion",
    caption:
      "Vector + graph + RankRAG without SFT: passages appear and claims stay closer to evidence.",
    system: "full_minus_sft",
    exampleId: "nils-jens-demo",
  },
  {
    id: "act-3",
    title: "Act 3 — SFT plot twist",
    caption:
      "Same retrieval stack + SFT: abstract sounds more like cs.CL prose but grounding collapses.",
    system: "full",
    exampleId: "nils-jens-demo",
  },
];
