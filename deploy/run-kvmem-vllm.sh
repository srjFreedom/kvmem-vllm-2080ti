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
ACTION=${1:-start}; shift || true
# GDN prefill kernel: prefer the SM75 FlashQLA build shipped in the engine's
# .deps. The profile requests `flashqla_legacy`; without that extension the
# engine falls back to the portable Triton kernel (~17% slower on long prefill),
# so override the request back to the fallback when the build is absent.
if [ -n "${KVMEM_ENGINE_DIR:-}" ]; then
  FLASHQLA_ROOT=${FLASHQLA_ROOT:-$KVMEM_ENGINE_DIR/.deps/FlashQLA-SM70-SM75}
  if [ -f "$FLASHQLA_ROOT/.torch_extensions_vllm_flashqla_legacy/flash_qla_legacy_gdn/flash_qla_legacy_gdn.so" ]; then
    export FLASHQLA_ROOT
    export PYTHONPATH="$KVMEM_ENGINE_DIR:$FLASHQLA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-"$FLASHQLA_ROOT/.torch_extensions_vllm_flashqla_legacy"}
  elif [ "$ACTION" = "start" ]; then
    set -- --set mm_gdn_prefill_backend=null "$@"
  fi
fi
case "$ACTION" in
  start)
    exec python3 "$REPO/scripts/serve_watch.py" start \
      --config "$CONFIG" --run-dir "$RUN_DIR" "$@" ;;
  stop)
    exec python3 "$REPO/scripts/serve_watch.py" stop "$@" ;;
  *)
    exec python3 "$REPO/scripts/serve_watch.py" "$ACTION" "$@" ;;
esac
