const VAGUE_PATTERNS = [
  /\bnlp\b.*\b(hard|difficult|unpredictable|challenging)\b/i,
  /\bai\b.*\b(future|revolution)\b/i,
  /^.{0,40}$/,
  /\b(interesting|cool|nice)\b/i,
];

const STRONG_TERMS = [
  "retrieval",
  "transformer",
  "benchmark",
  "dataset",
  "evaluation",
  "fine-tuning",
  "machine translation",
  "semantic",
  "parsing",
  "dialogue",
  "embedding",
  "abstract",
  "graph",
  "rerank",
];

export type OutlineQuality = {
  level: "good" | "warn" | "poor";
  message: string;
  score: number;
};

export function assessOutlineQuality(outline: string): OutlineQuality | null {
  const text = outline.trim();
  if (text.length < 10) return null;

  let score = 0.4;
  const lower = text.toLowerCase();

  for (const term of STRONG_TERMS) {
    if (lower.includes(term)) score += 0.08;
  }
  if (text.includes("\n-") || text.includes("\n•")) score += 0.1;
  if (text.length > 120) score += 0.1;

  for (const pattern of VAGUE_PATTERNS) {
    if (pattern.test(text)) score -= 0.25;
  }

  score = Math.max(0, Math.min(1, score));

  if (score >= 0.55) {
    return {
      level: "good",
      message: "Outline looks corpus-friendly — concrete NLP terms detected.",
      score,
    };
  }
  if (score >= 0.35) {
    return {
      level: "warn",
      message:
        "Outline is a bit vague — retrieval may miss precise evidence. Try an example or add task/method terms.",
      score,
    };
  }
  return {
    level: "poor",
    message:
      "Low retrieval overlap expected — use ↻ Example or add section headings + benchmark names.",
    score,
  };
}
