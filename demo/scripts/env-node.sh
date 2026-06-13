#!/usr/bin/env bash
# Prefer bundled Node 20; fall back to system node/pnpm on PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE_BIN="$ROOT/.tools/node-v20.18.2-linux-x64/bin"
if [[ -x "$NODE_BIN/node" ]]; then
  export PATH="$NODE_BIN:$PATH"
fi
if ! command -v node >/dev/null 2>&1; then
  echo "Node.js not found. Install Node 20+ or restore demo/.tools/" >&2
  exit 1
fi
exec "$@"
