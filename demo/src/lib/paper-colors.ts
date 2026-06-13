import type { Passage } from "./types";

export type PaperTagStyle = {
  bg: string;
  border: string;
  text: string;
};

export const PAPER_TAG_PALETTE: PaperTagStyle[] = [
  { bg: "bg-cyan-500/20", border: "border-cyan-400/55", text: "text-cyan-300" },
  { bg: "bg-emerald-500/20", border: "border-emerald-400/55", text: "text-emerald-300" },
  { bg: "bg-amber-500/20", border: "border-amber-400/55", text: "text-amber-300" },
  { bg: "bg-rose-500/20", border: "border-rose-400/55", text: "text-rose-300" },
  { bg: "bg-violet-500/20", border: "border-violet-400/55", text: "text-violet-300" },
  { bg: "bg-orange-500/20", border: "border-orange-400/55", text: "text-orange-300" },
  { bg: "bg-teal-500/20", border: "border-teal-400/55", text: "text-teal-300" },
  { bg: "bg-fuchsia-500/20", border: "border-fuchsia-400/55", text: "text-fuchsia-300" },
  { bg: "bg-sky-500/20", border: "border-sky-400/55", text: "text-sky-300" },
  { bg: "bg-lime-500/20", border: "border-lime-400/55", text: "text-lime-300" },
];

const PAPER_ID_RE = /\d{4}\.\d{4,5}/;
const ABSTRACT_CITE_RE = /\((\d{4}\.\d{4,5})\)/g;

export function normalizePaperId(paperId: string): string {
  return paperId.replace(/^arxiv:/i, "").trim();
}

export function extractPaperIdsFromAbstract(text: string): string[] {
  const ids: string[] = [];
  for (const match of text.matchAll(ABSTRACT_CITE_RE)) {
    if (match[1]) ids.push(match[1]);
  }
  return ids;
}

export function buildPaperColorMap(
  passages: Passage[],
  abstractText?: string | null,
): Map<string, number> {
  const map = new Map<string, number>();
  let next = 0;

  const assign = (rawId: string) => {
    const id = normalizePaperId(rawId);
    if (!id || !PAPER_ID_RE.test(id) || map.has(id)) return;
    map.set(id, next % PAPER_TAG_PALETTE.length);
    next += 1;
  };

  for (const passage of passages) {
    assign(passage.paper_id);
  }
  if (abstractText) {
    for (const id of extractPaperIdsFromAbstract(abstractText)) {
      assign(id);
    }
  }
  return map;
}

export function colorIndexForPaper(
  paperId: string,
  colorMap: Map<string, number>,
): number {
  const id = normalizePaperId(paperId);
  return colorMap.get(id) ?? 0;
}

export function styleForPaper(
  paperId: string,
  colorMap: Map<string, number>,
): PaperTagStyle {
  return PAPER_TAG_PALETTE[colorIndexForPaper(paperId, colorMap)];
}
