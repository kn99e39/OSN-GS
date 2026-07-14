#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
STREAM_HOST="${STREAM_HOST:-127.0.0.1}"
STREAM_PORT="${STREAM_PORT:-8080}"

exec "$PYTHON_BIN" scripts/train_osn_gs_torch.py \
  --stream_server_host "$STREAM_HOST" \
  --stream_server_port "$STREAM_PORT" \
  "$@"
