#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 scripts/train_osn_gs_torch.py   --stream_server_host 127.0.0.1   --stream_server_port 8080   "$@"
