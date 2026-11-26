#!/bin/bash
# Example of the DFlash2 launcher: the repository keeps its default profile, and
# this script (living outside the repo, e.g. ~/run-kvmem-2080ti-dflash2.sh)
# supplies the paths and the route overrides through --set.
#
#   ./example-dflash2-launcher.sh [start|stop|status|logs|params] [--set KEY=VALUE ...]
#
# Shape: no bounded planner — the engine sizes the pool and KVMem opens windows at
# runtime, the same layout the MTP lane uses. 262K logical context, B=36K window,
# 3.0 GiB pool, two concurrent requests, DFlash2 k=7 with a 4-bit W4A16 draft.
#
# 3.0 GiB, not 3.5: the FP8 target plus a 1.19 GiB draft leaves ~3.0 GiB of pool.
# At 3.2 GiB the prefill-completion transient (~192 MiB) no longer fits and the
# engine dies with CUDA OOM; 3.0 GiB keeps ~700 MiB of headroom.
# Do NOT set mamba_block_size: with mamba_cache_mode=none the engine defaults the
# GDN state page to max_model_len, so each request holds 1 state page + 7
# speculative tail pages. Pinning 1,648 makes the state dense (42 pages/group at
# 57K) and the 3 GiB pool cannot hold one request.
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
  --set max_num_batched_tokens=2064            # 2048 + 16: DFlash2 k=7 verification width
  --set prefill_chunk_tokens=1024
  --set chat_prefill_blocks=64
  --set query_replay_budget=2048               # ~180 KiB VRAM per token; 4096 doubles it
  --set kv_cache_memory_bytes=3221225472        # 3.0 GiB: the FP8 target + W4A16 draft's ceiling
  --set logical_kv_capacity=false               # no per-request static reservation
  --set skip_kv_check=true                      # allow max_model_len above the pool
  --set mamba_cache_mode=none                   # GDN state page = max_model_len, 1 page/request

  # DFlash2 speculation
  --set draft_method=dflash
  --set draft_model=/path/to/DFlash2-W4A16      # e.g. syvai/Qwen3.8-27B-DFlash2-W4A16
  --set draft_quantization=null                 # auto-detect: W4A16 via compressed-tensors
  --set draft_tokens=7
  --set experimental_dflash_query_planning=true
  --set experimental_dflash_compiled_multi=true
  --set sm75_spec_sync_mode=safe
  --set allow_mamba_spec_full_cudagraph=true

  # window (resident on the GPU)
  --set kvmem.select_budget=43008               # B: 42K
  --set kvmem.gen_budget=4096                   # R: 4K (window B+R = 47,104 = 29 pages)
  --set kvmem.block_tokens=32
  --set kvmem.sink_blocks=384
  --set kvmem.recent_blocks=64
  --set kvmem.rolling_active_window=false        # engine-managed window, no per-token planning

  # host history and sessions
  --set kvmem.cpu_read_cache_bytes=8589934592    # 8 GiB/rank = 524,288 tokens = repo blocks
  --set kvmem.repository_blocks=16384
  --set kvmem.archive_backend=cpu
  --set kvmem.archive_io_max_bytes=536870912
  --set kvmem.session_capacity=2
)

if [ "$ACTION" = "start" ]; then
  exec "$REPO/deploy/run-kvmem-vllm.sh" start "${OVERRIDES[@]}" "$@"
fi
exec "$REPO/deploy/run-kvmem-vllm.sh" "$ACTION" "$@"
