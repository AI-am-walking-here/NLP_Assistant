#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCHIVE_URL="${GROUNDED_ARTIFACTS_URL:-}"
ARCHIVE_FILE="${1:-${ROOT}/../grounded-artifacts.tar.gz}"

if [[ -n "$ARCHIVE_URL" && ! -f "$ARCHIVE_FILE" ]]; then
  echo "Downloading artifacts from $ARCHIVE_URL ..."
  curl -L "$ARCHIVE_URL" -o "$ARCHIVE_FILE"
fi

if [[ ! -f "$ARCHIVE_FILE" ]]; then
  echo "Missing archive: $ARCHIVE_FILE" >&2
  echo "Set GROUNDED_ARTIFACTS_URL or pass path to grounded-artifacts.tar.gz" >&2
  echo "See data/ARTIFACTS_DOWNLOAD.md" >&2
  exit 1
fi

echo "Extracting into $ROOT ..."
tar -xzf "$ARCHIVE_FILE" -C "$ROOT"
echo "OK:"
ls -lh "$ROOT/data/corpus/papers.jsonl.gz" \
  "$ROOT/runs/seg5_sft_train_2026-06-05-0617/adapter" \
  "$ROOT/data/indices/faiss.index" 2>/dev/null || true
