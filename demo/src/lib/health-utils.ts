import type { HealthResponse } from "./types";

export function healthChanged(
  prev: HealthResponse | null,
  next: HealthResponse,
): boolean {
  if (!prev) return true;
  return (
    prev.status !== next.status ||
    prev.stack_ready !== next.stack_ready ||
    prev.mock_mode !== next.mock_mode ||
    prev.demo_fast !== next.demo_fast ||
    prev.gpu_mode !== next.gpu_mode ||
    prev.error !== next.error
  );
}
