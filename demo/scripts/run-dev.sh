#!/usr/bin/env bash
# Start API first, wait until :8080 responds, then start Next.js.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MOCK="${GROUNDED_DEMO_MOCK:-0}"
FAST="${GROUNDED_DEMO_FAST:-0}"
API_SCRIPT="dev:api"
if [[ "$MOCK" == "1" ]]; then
  API_SCRIPT="dev:api:mock"
elif [[ "$FAST" == "1" ]]; then
  API_SCRIPT="dev:api:fast"
fi

# shellcheck source=ensure-api-port.sh
source "$ROOT/scripts/ensure-api-port.sh"

STARTED_API=0
API_PID=""

cleanup() {
  if [[ "$STARTED_API" == "1" && -n "$API_PID" ]]; then
    kill "$API_PID" 2>/dev/null || true
  fi
  if [[ -n "${WEB_PID:-}" ]]; then
    kill "$WEB_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${GROUNDED_API_REUSE:-0}" == "1" ]]; then
  echo "[run-dev] Using existing API on :${GROUNDED_API_PORT:-8080}"
else
  echo "[run-dev] Starting Python API (${API_SCRIPT})..."
  bash "$ROOT/scripts/env-node.sh" pnpm "$API_SCRIPT" &
  API_PID=$!
  STARTED_API=1
fi

bash "$ROOT/scripts/wait-for-api.sh"

echo "[run-dev] Starting Next.js (port 3000)..."
bash "$ROOT/scripts/env-node.sh" env \
  NEXT_PUBLIC_GROUNDED_API_URL="http://127.0.0.1:${GROUNDED_API_PORT:-8080}" \
  next dev --hostname 0.0.0.0 &
WEB_PID=$!

if [[ "$STARTED_API" == "1" ]]; then
  while kill -0 "$WEB_PID" 2>/dev/null; do
    if ! kill -0 "$API_PID" 2>/dev/null; then
      echo "[run-dev] API exited — stopping Next.js" >&2
      kill "$WEB_PID" 2>/dev/null || true
      wait "$WEB_PID" 2>/dev/null || true
      exit 1
    fi
    sleep 1
  done
  wait "$API_PID" 2>/dev/null || true
else
  wait "$WEB_PID"
fi
