#!/usr/bin/env bash
# Download model weights to GROUNDED_MODELS_ROOT (run on a machine with disk + HF access).
# The Python pipeline does NOT auto-download unless GROUNDED_ALLOW_MODEL_DOWNLOAD=1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_ROOT="${GROUNDED_MODELS_ROOT:-${ROOT}/models}"
mkdir -p "$MODELS_ROOT"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Install: pip install 'huggingface_hub[cli]' && huggingface-cli login" >&2
  exit 1
fi

echo "Downloading to: $MODELS_ROOT"
echo "Requires HF login + Llama 3.1 license acceptance for gated models."

download() {
  local repo="$1"
  local dest="$MODELS_ROOT/$repo"
  if [[ -f "$dest/config.json" ]] && compgen -G "$dest/*.safetensors" >/dev/null; then
    echo "[skip] $repo already present"
    return 0
  fi
  mkdir -p "$dest"
  huggingface-cli download "$repo" --local-dir "$dest"
}

download "BAAI/bge-large-en-v1.5"
download "meta-llama/Llama-3.1-8B-Instruct"

if [[ "${DOWNLOAD_VERIFIER:-0}" == "1" ]]; then
  download "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
else
  echo "[skip] 70B verifier — set DOWNLOAD_VERIFIER=1 to include (~40 GB)"
fi

export GROUNDED_MODELS_ROOT="$MODELS_ROOT"
export PYTHONPATH="$ROOT/src"
python3 "$ROOT/scripts/check_models.py"
