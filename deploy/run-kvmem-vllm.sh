#!/bin/bash
# Start the frozen KVMem baseline on this 2x2080Ti host.
#   ./deploy/run-kvmem-vllm.sh [start|stop|status]
# Engine: the vLLM 0.2.2.post3 SM75 checkout (set KVMEM_ENGINE_DIR or KVMEM_ENGINE_VENV)
# Plugin: this checkout (VLLM_PLUGINS=kvmem, --scheduler-cls KVMemScheduler)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=${CONFIG:-$REPO/local/serve-2080ti-default.json}
RUN_DIR=${RUN_DIR:-${TMPDIR:-/tmp}/kvmem-run-baseline}
# The watchdog is stdlib-only; the engine venv is resolved inside it.
exec python3 "$REPO/scripts/serve_watch.py" "${1:-start}" \
  --config "$CONFIG" --run-dir "$RUN_DIR" ${2:-}
