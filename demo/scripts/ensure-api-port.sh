#!/usr/bin/env bash
# Ensure port 8080 is free or already serving a healthy NILS-JENS API.
set -euo pipefail

PORT="${GROUNDED_API_PORT:-8080}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"

api_responds() {
  local body=""
  body="$(curl -sf --max-time 3 "$HEALTH_URL" 2>/dev/null || true)"
  [[ -n "$body" ]] && echo "$body" | grep -q '"status"'
}

port_pid() {
  local pid=""
  if command -v ss >/dev/null 2>&1; then
    pid="$(ss -tlnp 2>/dev/null | grep ":${PORT} " | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1 || true)"
  elif command -v lsof >/dev/null 2>&1; then
    pid="$(lsof -t -i ":${PORT}" -sTCP:LISTEN 2>/dev/null | head -1 || true)"
  fi
  echo "$pid"
}

if api_responds; then
  echo "[ensure-api] Reusing API already listening on :${PORT}"
  export GROUNDED_API_REUSE=1
  exit 0
fi

stale_pid="$(port_pid)"
if [[ -n "$stale_pid" ]]; then
  echo "[ensure-api] Stale process on :${PORT} (pid ${stale_pid}) — stopping..."
  kill "$stale_pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    [[ -z "$(port_pid)" ]] && break
    sleep 0.25
  done
  stale_pid="$(port_pid)"
  if [[ -n "$stale_pid" ]]; then
    kill -9 "$stale_pid" 2>/dev/null || true
    sleep 0.5
  fi
fi

export GROUNDED_API_REUSE=0
