#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$ROOT/.." && pwd)"
MOCK="${GROUNDED_DEMO_MOCK:-0}"
FAST="${GROUNDED_DEMO_FAST:-0}"

# shellcheck source=ensure-api-port.sh
source "$ROOT/scripts/ensure-api-port.sh"
if [[ "${GROUNDED_API_REUSE:-0}" == "1" ]]; then
  echo "[start-api] API already healthy on :${GROUNDED_API_PORT:-8080}"
  exit 0
fi

cd "$PROJECT_ROOT"
export PYTHONPATH=src
export GROUNDED_MODELS_ROOT="${GROUNDED_MODELS_ROOT:-/data/team1/models}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GROUNDED_DEMO_MOCK="$MOCK"
export GROUNDED_DEMO_FAST="$FAST"
export GROUNDED_DEMO_EMBED_DEVICE="${GROUNDED_DEMO_EMBED_DEVICE:-}"
if [[ "$FAST" == "1" ]]; then
  export GROUNDED_DEMO_PRELOAD_GENERATOR="${GROUNDED_DEMO_PRELOAD_GENERATOR:-1}"
fi

PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi
if [[ -z "$PYTHON" ]]; then
  echo "[start-api] No Python found. Run: bash scripts/setup_venv.sh" >&2
  exit 1
fi

exec "$PYTHON" scripts/serve_demo.py --port "${GROUNDED_API_PORT:-8080}"
