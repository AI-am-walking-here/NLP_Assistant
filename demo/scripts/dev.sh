#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NODE_BIN="${ROOT}/.tools/node-v20.18.2-linux-x64/bin"
export PATH="${NODE_BIN}:${PATH}"

cd "${ROOT}"
if [[ ! -d node_modules ]]; then
  npm install
fi
exec npm run dev -- --hostname 0.0.0.0 "$@"
