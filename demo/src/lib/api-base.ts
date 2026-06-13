/** Direct FastAPI base for long-running generate (optional). */
export function directApiBase(): string {
  if (typeof window === "undefined") {
    return "";
  }
  return (process.env.NEXT_PUBLIC_GROUNDED_API_URL ?? "").replace(/\/$/, "");
}

/** Same-origin Next.js proxies — return 503 while Python API boots. */
export function proxyUrl(path: string): string {
  return path;
}

/** Generate hits FastAPI directly when configured; otherwise uses Next proxy. */
export function generateUrl(path: string): string {
  const base = directApiBase();
  return base ? `${base}${path}` : path;
}
