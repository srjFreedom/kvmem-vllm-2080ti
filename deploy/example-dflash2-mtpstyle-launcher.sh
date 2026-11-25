#!/usr/bin/env bash
# DFlash2, MTP-style lane: no bounded planner, 3.5 GiB pool, 2 concurrent requests,
# 48K window (the MTP lane's window). Per-step cost matches the pristine engine on
# the same checkpoint and draft; see README "DFlash2 route".
#
#   ./run-kvmem-2080ti-dflash2-mtpstyle.sh start|stop|status
#
# Everything else (config validation, launch, health probe, logs) is the shared
# DFlash2 launcher; this file only pins the MTP-style route parameters.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export RUN_DIR=${RUN_DIR:-/var/tmp/kvmem-run-dflash2-mtpstyle}
export KVMEM_MODEL_DIR=${KVMEM_MODEL_DIR:-/home/srj/models/Coder390/FP8-fp8mtp}
# Keep the draft in one sliding group instead of five native per-layer pools.
export VLLM_DFLASH_NATIVE_GROUPS_WITHOUT_APC=${VLLM_DFLASH_NATIVE_GROUPS_WITHOUT_APC:-0}

exec "$HERE/run-kvmem-2080ti-dflash2.sh" "$@" \
  --set max_model_len=262144 \
  --set kv_cache_memory_bytes=3758096384 \
  --set logical_kv_capacity=false \
  --set skip_kv_check=true \
  --set mamba_block_size=1648 \
  --set max_num_seqs=2 \
  --set kvmem.select_budget=49152 \
  --set kvmem.gen_budget=8192 \
  --set kvmem.rolling_active_window=false \
  --set draft_model=/home/srj/models/Coder390/DFlash2-W4A16
