#!/usr/bin/env bash
# Extract libpython3.10-dev headers into .tmp/py310dev (no sudo).
# Required for vLLM / Triton worker JIT (Python.h).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/.tmp/py310dev"
HEADER="${DEST}/usr/include/python3.10/Python.h"
mkdir -p "${ROOT}/.tmp"
cd "${ROOT}/.tmp"

if [[ ! -f "${HEADER}" ]]; then
  rm -rf py310dev
  # python3.10-dev meta-deb on some mirrors is doc-only; headers live in libpython3.10-dev
  apt-get download libpython3.10-dev
  dpkg-deb -x libpython3.10-dev_*.deb py310dev
fi

if [[ ! -f "${HEADER}" ]]; then
  echo "ERROR: ${HEADER} missing after extract" >&2
  exit 1
fi
echo "OK: ${HEADER}"
