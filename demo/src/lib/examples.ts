/**
 * Demo outlines tuned for high retrieval overlap with the cs.CL index.
 *
 * Design rules (from full_minus_sft eval analysis):
 * - Query = title + outline → BGE + graph + RankRAG
 * - Use concrete NLP task names, standard section headings, and corpus-dense terms
 * - Avoid vague user probes ("NLP is unpredictable") that retrieve generic uncertainty papers
 */

export type ExampleFormat = "bullets" | "prose" | "mixed";

export type DemoExample = {
  id: string;
  label: string;
  format: ExampleFormat;
  title: string;
  outline: string;
};

export const DEMO_EXAMPLES: DemoExample[] = [
  {
    id: "mt-gender-bias",
    label: "MT gender bias",
    format: "bullets",
    title: "Evaluating Gender Bias in Machine Translation",
    outline: [
      "- Introduction",
      "- Challenge Set for Gender Bias in MT",
      "- Evaluation",
      "- MT systems",
      "- Target languages and morphological analysis",
    ].join("\n"),
  },
  {
    id: "goal-dialogue-pomdp",
    label: "Goal-oriented dialogue",
    format: "prose",
    title: "Context-Aware Language Modeling for Goal-Oriented Dialogue Systems",
    outline:
      "Goal-oriented dialogue systems must balance fluent language generation with task-specific control. We formulate dialogue as a partially observed Markov decision process and fine-tune a pretrained language model with task relabeling. Experiments on a flight-booking benchmark report task success and compare against state-of-the-art context-aware dialogue baselines.",
  },
  {
    id: "entity-guided-nlg",
    label: "Entity-guided generation",
    format: "bullets",
    title: "Injecting Entity Types into Entity-Guided Text Generation",
    outline: [
      "- Introduction",
      "- Entity-related Text Generation",
      "- Word-to-text Generation",
      "- Entity-Guided Text Generation",
      "- Task Definition",
    ].join("\n"),
  },
  {
    id: "semantic-parser-alignment",
    label: "Semantic parsing alignments",
    format: "prose",
    title: "Measuring Alignment Bias in Neural Seq2Seq Semantic Parsers",
    outline:
      "Sequence-to-sequence semantic parsers are often assumed to learn word alignments automatically via attention. We annotate a Geo-style dataset with monotonic and non-monotonic alignments and compare parser performance across alignment types. Results show substantially higher accuracy on monotonic alignment cases than on examples requiring complex reordering.",
  },
  {
    id: "mt-quality-estimation",
    label: "MT quality estimation",
    format: "bullets",
    title: "MDQE: A More Accurate Direct Pretraining for Machine Translation Quality Estimation",
    outline: [
      "- Introduction",
      "- Generator",
      "- Estimator",
      "- Experimental Settings",
      "- Benchmark Results",
    ].join("\n"),
  },
  {
    id: "nils-jens-demo",
    label: "NILS-JENS (presentation)",
    format: "mixed",
    title:
      "NILS-JENS: Evidence-Faithful Abstract Generation with Graph RAG and RankRAG",
    outline: [
      "We study cs.CL abstract generation where a title and contribution outline are expanded into a grounded abstract supported by retrieved passages.",
      "- Problem: LLMs hallucinate claims not supported by evidence.",
      "- Method: hybrid vector + pilot graph retrieval, RankRAG reranking, optional domain SFT.",
      "- Evaluation: FActScore on 80 held-out prompts with a 70B verifier.",
      "- Result: full_minus_sft achieves the best faithfulness; SFT alone hurts grounding.",
    ].join("\n"),
  },
  {
    id: "icl-compositional",
    label: "In-context compositional generalization",
    format: "mixed",
    title:
      "Generating Demonstrations for In-Context Compositional Generalization in Grounded Language Learning",
    outline: [
      "In-context learning for compositional generalization is sensitive to support example selection, especially in grounded language settings where relevant training states may not match the query.",
      "- Compositional Generalization and Grounded Language Learning",
      "- In-context Learning",
      "- Support Selection for ICL",
      "- Experiments",
    ].join("\n"),
  },
  {
    id: "doc-mt-eval",
    label: "Document-level MT evaluation",
    format: "prose",
    title:
      "Align-then-Slide: A Complete Evaluation Framework for Ultra-Long Document-Level Machine Translation",
    outline:
      "Document-level machine translation with large language models breaks sentence-aligned evaluation assumptions. We propose Align-then-Slide: first align source and target sentences, then score translations with n-chunk sliding evaluation at multiple granularities. Experiments on WMT-style benchmarks show high correlation with expert MQM rankings and enable preference optimization for doc-MT systems.",
  },
  {
    id: "event-extraction-corpus",
    label: "Event extraction corpus",
    format: "bullets",
    title: "CrudeOilNews: An Annotated Crude Oil News Corpus for Event Extraction",
    outline: [
      "- Introduction",
      "- Annotation Methodologies",
      "- Finance and Economic Domain",
      "- Dataset Collection",
      "- Inter-annotator Agreement",
    ].join("\n"),
  },
  {
    id: "bias-pretrain-downstream",
    label: "Bias eval gaps",
    format: "mixed",
    title:
      "The Gaps between Pre-train and Downstream Settings in Bias Evaluation and Debiasing",
    outline: [
      "Pretrained language models change behavior substantially after fine-tuning, creating a gap between intrinsic bias scores and downstream task outcomes.",
      "- Bias Evaluations",
      "- Debiasing Methods",
      "- Downstream Task Evaluations",
      "- In-context Learning vs Fine-tuning",
    ].join("\n"),
  },
  {
    id: "pragmatic-speakers",
    label: "Pragmatic communication",
    format: "prose",
    title:
      "Calibrate your listeners! Robust Communication-Based Training for Pragmatic Speakers",
    outline:
      "Neural speakers trained with a listener objective often suffer from semantic drift away from natural language. We regularize speaker training with a population of neural listeners to improve uncertainty calibration and reduce drift. Reference-game experiments show that ensemble listeners enable pragmatic utterance generation while scaling to large vocabularies and unseen partners.",
  },
  {
    id: "chinese-spelling-ime",
    label: "Chinese spelling (IME)",
    format: "bullets",
    title: "CSCD-IME: Correcting Spelling Errors Generated by Pinyin IME",
    outline: [
      "- Introduction",
      "- CSCD-IME",
      "- Data Collection",
      "- Data Selection",
      "- Pseudo-data Construction",
    ].join("\n"),
  },
  {
    id: "car-answer-generation",
    label: "Complex answer generation",
    format: "mixed",
    title:
      "Does Structure Matter? Leveraging Data-to-Text Generation for Answering Complex Information Needs",
    outline: [
      "Complex information needs require structured answers rather than a single retrieved passage. We treat answer generation as data-to-text with an explicit content selection and planning stage before surface realization.",
      "- Dataset",
      "- Model variants and baselines",
      "- Metrics",
      "- Planning-based vs end-to-end generation",
    ].join("\n"),
  },
  {
    id: "low-resource-asr",
    label: "Low-resource ASR",
    format: "prose",
    title:
      "Improving Low-Resource Speech Recognition with Pretrained Speech Models: Continued Pretraining vs. Semi-Supervised Training",
    outline:
      "Self-supervised speech models such as wav2vec 2.0 and HuBERT improve low-resource ASR but depend on pretraining language coverage. We compare continued pretraining on in-language unlabeled audio against semi-supervised training with pseudo-labels across several low-resource languages. Continued pretraining matches or beats pseudo-labeling while avoiding costly decoding passes over unlabeled speech.",
  },
  {
    id: "multilingual-idioms",
    label: "Multilingual idioms",
    format: "mixed",
    title: "Generating Continuations in Multilingual Idiomatic Contexts",
    outline: [
      "We evaluate whether generative language models can produce contextually appropriate continuations when narratives contain idiomatic or literal multiword expressions in English and Portuguese.",
      "- Idioms-related Classification Tasks",
      "- Idioms-related Generative Tasks",
      "- Zero-shot, few-shot, and fine-tuned training settings",
    ].join("\n"),
  },
];
