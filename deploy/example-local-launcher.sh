#!/bin/bash
# Example of the machine-local launcher pattern: the repository keeps its default
# profile, and this script (living outside the repo, e.g. ~/run-kvmem-2080ti.sh)
# supplies the paths and any machine-specific overrides through --set.
#
#   ./example-local-launcher.sh [start|stop|status] [--set KEY=VALUE ...]
set -euo pipefail

REPO=${REPO:-/path/to/kvmem-vllm-2080ti}
export KVMEM_ENGINE_DIR=${KVMEM_ENGINE_DIR:-/path/to/vllm-0.2.2.post3-sm75}
export KVMEM_MODEL_DIR=${KVMEM_MODEL_DIR:-/path/to/checkpoint}
export CONFIG=${CONFIG:-$REPO/local/serve-2080ti-default.json}
export RUN_DIR=${RUN_DIR:-/var/tmp/kvmem-run-prod}

ACTION=${1:-start}
shift || true

# Machine-specific overrides; the profile on disk is not modified.
OVERRIDES=(
  --set gpu_memory_utilization=0.92
  --set max_model_len=262144
  --set prefill_chunk_tokens=2048
  --set max_num_batched_tokens=2048
  --set kvmem.select_budget=36864
  --set kvmem.gen_budget=4096
  --set kvmem.cpu_read_cache_bytes=6442450944
)

if [ "$ACTION" = "start" ]; then
  exec "$REPO/deploy/run-kvmem-vllm.sh" start "${OVERRIDES[@]}" "$@"
fi
exec "$REPO/deploy/run-kvmem-vllm.sh" "$ACTION" "$@"
