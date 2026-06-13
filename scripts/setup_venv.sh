#!/usr/bin/env bash
# Install all Python deps under /data/team1/llm-assistant-final/.venv only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PIP_CACHE_DIR="${ROOT}/.pip-cache"
export TMPDIR="${ROOT}/.tmp"
export HF_HOME="${ROOT}/.hf-cache"
export GROUNDED_MODELS_ROOT="${GROUNDED_MODELS_ROOT:-/data/team1/models}"
export XDG_CACHE_HOME="${ROOT}/.cache"
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}" "${HF_HOME}" "${XDG_CACHE_HOME}"

if [[ ! -d "${ROOT}/.venv" ]]; then
  python3 -m venv "${ROOT}/.venv" || python3 -m venv --without-pip "${ROOT}/.venv"
fi

if [[ ! -x "${ROOT}/.venv/bin/pip" ]]; then
  curl -sS https://bootstrap.pypa.io/get-pip.py -o "${TMPDIR}/get-pip.py"
  "${ROOT}/.venv/bin/python3" "${TMPDIR}/get-pip.py"
fi

"${ROOT}/.venv/bin/pip" install --upgrade pip
"${ROOT}/.venv/bin/pip" install -e "${ROOT}[dev,ml,train]"
echo "OK: venv at ${ROOT}/.venv (isolated; cache ${PIP_CACHE_DIR})"
echo "Models root: ${GROUNDED_MODELS_ROOT} (see configs/models.yaml)"
