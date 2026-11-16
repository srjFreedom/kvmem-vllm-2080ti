#!/bin/bash
# Start the frozen KVMem baseline on this 2x2080Ti host.
#   ./deploy/run-kvmem-vllm.sh [start|stop|status]
# Engine: /home/srj/vLLM-2080Ti-Definitive (vLLM 0.2.2.post3 + one 15-line patch)
# Plugin: this checkout (VLLM_PLUGINS=kvmem, --scheduler-cls KVMemScheduler)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=/home/srj/vLLM-2080Ti-Definitive/.venv/bin
CONFIG=${CONFIG:-$REPO/local/serve-2080ti-baseline.json}
RUN_DIR=${RUN_DIR:-/var/tmp/kvmem-run-baseline}
exec "$VENV/python" "$REPO/scripts/serve_watch.py" "${1:-start}" \
  --config "$CONFIG" --run-dir "$RUN_DIR" ${2:-}
