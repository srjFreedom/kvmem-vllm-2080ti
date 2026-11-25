#!/bin/bash
# Example of the DFlash2 launcher: the repository keeps its default profile, and
# this script (living outside the repo, e.g. ~/run-kvmem-2080ti-dflash2.sh)
# supplies the paths and the route overrides through --set.
#
#   ./example-dflash2-launcher.sh [start|stop|status|logs|params] [--set KEY=VALUE ...]
#
# Shape: no bounded planner — the engine sizes the pool and KVMem opens windows at
# runtime, the same layout the MTP lane uses. 262K logical context, B=48K window,
# 3.5 GiB pool, two concurrent requests, DFlash2 k=7 with a 4-bit W4A16 draft.
set -euo pipefail

REPO=${REPO:-/path/to/kvmem-vllm-2080ti}
export KVMEM_ENGINE_DIR=${KVMEM_ENGINE_DIR:-/path/to/vllm-0.2.2.post3-sm75}
export KVMEM_MODEL_DIR=${KVMEM_MODEL_DIR:-/path/to/checkpoint}
export CONFIG=${CONFIG:-$REPO/local/serve-2080ti-dflash2.json}
export RUN_DIR=${RUN_DIR:-/var/tmp/kvmem-run-dflash2}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # needs enable_cumem_allocator=true

# Mixed target/draft page sizes cannot be expressed by LBNHC: block-outermost it is.
export VLLM_KV_CACHE_LAYOUT=BLHNC
# Let the GDN state group use an attention-sized page while APC stays off.
export VLLM_MAMBA_NONE_SMALL_BLOCK=1
# One draft group shares the target pool page instead of five native draft pools.
export VLLM_DFLASH_NATIVE_GROUPS_WITHOUT_APC=0
export VLLM_DFLASH_DRAFT_PAGE_ALIGN=1
export KVMEM_TOKENIZER_PATH=${KVMEM_TOKENIZER_PATH:-$KVMEM_MODEL_DIR}
# Pinned checkpoint prewarm only supports target-only/MTP drafts.
export KVMEM_CHECKPOINT_PINNED_PREWARM=0

ACTION=${1:-start}
shift || true

OVERRIDES=(
  # model and engine
  --set model="$KVMEM_MODEL_DIR"
  --set tokenizer="$KVMEM_MODEL_DIR"
  --set hf_config_path="$KVMEM_MODEL_DIR"
  --set execution_mode=compiled_graph   # the base engine's fast path: mode 3 + FULL_AND_PIECEWISE
  --set max_model_len=262144
  --set host=0.0.0.0
  --set port=18200
  --set mm_image_limit=64

  # pool and scheduling
  --set max_num_seqs=2
  --set kv_cache_memory_bytes=3758096384        # 3.5 GiB: the FP8 target's measured ceiling
  --set logical_kv_capacity=false               # no per-request static reservation
  --set skip_kv_check=true                      # allow max_model_len above the pool
  --set mamba_block_size=1648                   # GDN pages as large as an attention page
  --set mamba_cache_mode=none

  # DFlash2 speculation
  --set draft_method=dflash
  --set draft_model=/path/to/DFlash2-W4A16      # e.g. syvai/Qwen3.8-27B-DFlash2-W4A16
  --set draft_quantization=null                 # auto-detect: W4A16 via compressed-tensors
  --set draft_tokens=7
  --set experimental_dflash_query_planning=true

  # window (resident on the GPU)
  --set kvmem.select_budget=49152               # B: 48K
  --set kvmem.gen_budget=8192                   # R: 8K
  --set kvmem.block_tokens=32
  --set kvmem.sink_blocks=384
  --set kvmem.recent_blocks=64
  --set kvmem.rolling_active_window=false       # engine-managed window, no per-token planning

  # host history and sessions
  --set kvmem.cpu_read_cache_bytes=6442450944
  --set kvmem.repository_blocks=16384
  --set kvmem.archive_backend=cpu
  --set kvmem.session_capacity=2
)

if [ "$ACTION" = "start" ]; then
  exec "$REPO/deploy/run-kvmem-vllm.sh" start "${OVERRIDES[@]}" "$@"
fi
exec "$REPO/deploy/run-kvmem-vllm.sh" "$ACTION" "$@"
