#!/usr/bin/env bash
# Block until the FastAPI /health endpoint returns JSON (stack may still be loading).
set -euo pipefail

PORT="${GROUNDED_API_PORT:-8080}"
TIMEOUT="${GROUNDED_API_WAIT_TIMEOUT:-600}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
INTERVAL="${GROUNDED_API_WAIT_INTERVAL:-2}"

echo "[wait-for-api] Waiting for ${HEALTH_URL} (timeout ${TIMEOUT}s)..."
start_epoch="$(date +%s)"

while true; do
  body="$(curl -sf --max-time 5 "$HEALTH_URL" 2>/dev/null || true)"
  if [[ -n "$body" ]] && echo "$body" | grep -q '"status"'; then
    echo "[wait-for-api] API accepting HTTP on :${PORT}"
    exit 0
  fi

  now="$(date +%s)"
  if (( now - start_epoch >= TIMEOUT )); then
    echo "[wait-for-api] Timed out after ${TIMEOUT}s — API never bound to :${PORT}" >&2
    exit 1
  fi
  sleep "$INTERVAL"
done
