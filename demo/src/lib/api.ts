import { generateUrl, proxyUrl } from "./api-base";
import type {
  GenerateRequest,
  GenerateResponse,
  HealthResponse,
  SystemInfo,
  VerifyResponse,
} from "./types";

const GENERATE_TIMEOUT_MS = 15 * 60 * 1000;

const LOADING_HEALTH: HealthResponse = {
  status: "loading",
  mock_mode: false,
  stack_ready: false,
  error: "Python API is starting (model preload takes 1–3 minutes).",
};

async function parseJson<T>(res: Response): Promise<T> {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail =
      typeof data === "object" && data !== null && "detail" in data
        ? String((data as { detail: unknown }).detail)
        : res.statusText;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return data as T;
}

export async function fetchHealth(): Promise<HealthResponse> {
  try {
    const res = await fetch(proxyUrl("/health"), { cache: "no-store" });
    const data = await res.json().catch(() => ({}));
    if (res.status === 503) {
      return {
        ...LOADING_HEALTH,
        ...(typeof data === "object" && data !== null ? data : {}),
      } as HealthResponse;
    }
    return parseJson<HealthResponse>(res);
  } catch {
    return LOADING_HEALTH;
  }
}

export async function fetchSystems(): Promise<SystemInfo[]> {
  try {
    const res = await fetch(proxyUrl("/api/systems"), { cache: "no-store" });
    if (res.status === 503) {
      return [];
    }
    return parseJson<SystemInfo[]>(res);
  } catch {
    return [];
  }
}

export async function generateAbstract(
  body: GenerateRequest,
): Promise<GenerateResponse> {
  const res = await fetch(generateUrl("/api/generate"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(GENERATE_TIMEOUT_MS),
  });
  return parseJson<GenerateResponse>(res);
}

export async function verifyAbstract(body: {
  abstract: string;
  passages: string[];
}): Promise<VerifyResponse> {
  const res = await fetch(generateUrl("/api/verify"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(5 * 60 * 1000),
  });
  return parseJson<VerifyResponse>(res);
}
