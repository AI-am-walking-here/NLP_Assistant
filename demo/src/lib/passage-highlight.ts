const TOKEN_RE = /\b[a-z][a-z0-9]{3,}\b/gi;

const STOP = new Set([
  "that",
  "this",
  "with",
  "from",
  "have",
  "been",
  "were",
  "their",
  "which",
  "using",
  "based",
  "paper",
  "work",
  "show",
  "results",
  "model",
  "models",
  "method",
  "methods",
]);

export function passageOverlapTokens(
  abstract: string,
  passageText: string,
): Set<string> {
  const abstractTokens = new Set(
    (abstract.match(TOKEN_RE) ?? [])
      .map((t) => t.toLowerCase())
      .filter((t) => !STOP.has(t)),
  );
  const overlap = new Set<string>();
  for (const token of passageText.match(TOKEN_RE) ?? []) {
    const lower = token.toLowerCase();
    if (abstractTokens.has(lower) && !STOP.has(lower)) {
      overlap.add(lower);
    }
  }
  return overlap;
}

export function splitAbstractForHighlight(
  content: string,
  highlightTokens: Set<string>,
): Array<{ text: string; highlight: boolean }> {
  if (highlightTokens.size === 0) {
    return [{ text: content, highlight: false }];
  }

  const parts: Array<{ text: string; highlight: boolean }> = [];
  const re = /\b[a-z][a-z0-9]{3,}\b/gi;
  let last = 0;
  let match: RegExpExecArray | null;

  while ((match = re.exec(content)) !== null) {
    const token = match[0].toLowerCase();
    const start = match.index;
    if (start > last) {
      parts.push({ text: content.slice(last, start), highlight: false });
    }
    parts.push({
      text: match[0],
      highlight: highlightTokens.has(token) && !STOP.has(token),
    });
    last = start + match[0].length;
  }
  if (last < content.length) {
    parts.push({ text: content.slice(last), highlight: false });
  }
  return parts;
}

export function wordOverlapPct(a: string, b: string): number {
  const tokensA = new Set(
    (a.match(TOKEN_RE) ?? []).map((t) => t.toLowerCase()).filter((t) => !STOP.has(t)),
  );
  const tokensB = new Set(
    (b.match(TOKEN_RE) ?? []).map((t) => t.toLowerCase()).filter((t) => !STOP.has(t)),
  );
  if (tokensA.size === 0 || tokensB.size === 0) return 0;
  let shared = 0;
  for (const t of tokensA) {
    if (tokensB.has(t)) shared += 1;
  }
  return Math.round((shared / Math.max(tokensA.size, tokensB.size)) * 100);
}
