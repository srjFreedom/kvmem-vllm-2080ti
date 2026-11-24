#!/bin/bash
# Example of the machine-local launcher pattern: the repository keeps its default
# profile, and this script (living outside the repo, e.g. ~/run-kvmem-2080ti.sh)
# supplies the paths and every machine-specific override through --set.
#
#   ./example-local-launcher.sh [start|stop|status|logs|params] [--set KEY=VALUE ...]
#
# The engine log is written to $RUN_DIR/serve.log; `logs` prints it, `params`
# prints the effective values (window, history, sessions, MTP depth).
set -euo pipefail

REPO=${REPO:-/path/to/kvmem-vllm-2080ti}
export KVMEM_ENGINE_DIR=${KVMEM_ENGINE_DIR:-/path/to/vllm-0.2.2.post3-sm75}
export KVMEM_MODEL_DIR=${KVMEM_MODEL_DIR:-/path/to/checkpoint}
export CONFIG=${CONFIG:-$REPO/local/serve-2080ti-default.json}
export RUN_DIR=${RUN_DIR:-/var/tmp/kvmem-run}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # needs enable_cumem_allocator=true

ACTION=${1:-start}
shift || true

# Machine-specific overrides; the profile on disk is not modified.
OVERRIDES=(
  # model and engine
  --set model="$KVMEM_MODEL_DIR"
  --set tokenizer="$KVMEM_MODEL_DIR"
  --set hf_config_path="$KVMEM_MODEL_DIR"
  --set served_model_name=kvmem-2080ti
  --set execution_mode=compiled_graph   # 实测比 full_graph 快：code +12% / repeat +9%
  --set tensor_parallel_size=2
  --set dtype=float16
  --set generation_config=auto     # sampling defaults from the checkpoint's generation_config.json
  --set max_model_len=262144
  --set host=127.0.0.1        # 127.0.0.1 = local only; 0.0.0.0 exposes the API (no auth)
  --set port=18200

  # batching and VRAM
  --set max_num_seqs=1
  --set max_num_batched_tokens=2048
  --set prefill_chunk_tokens=1024      # logits transient ≈ 0.5 GB; 2048 doubles it
  --set gpu_memory_utilization=0.92    # only a ceiling: the pool below is pinned explicitly
  --set kv_cache_memory_bytes=4294967296   # 4 GiB; 200K cold prefill, 0 allocator retries
  --set cudagraph_capture_sizes=[5]
  --set enable_cumem_allocator=true    # required before expandable_segments
  --set disable_custom_all_reduce=true # VMM segments break custom all-reduce

  # MTP speculative decoding
  --set draft_method=mtp
  --set draft_tokens=4
  --set sm75_spec_sync_mode=nosync
  --set allow_mamba_spec_full_cudagraph=true

  # GPU window (matches local/serve-2080ti-default.json; override per host)
  --set kvmem.select_budget=36864
  --set kvmem.gen_budget=8192
  --set query_replay_budget=4096
  --set kvmem.block_tokens=32
  --set kvmem.sink_blocks=256
  --set kvmem.recent_blocks=64

  # host history
  --set kvmem.cpu_read_cache_bytes=6442450944
  --set kvmem.repository_blocks=16384
  --set kvmem.archive_backend=cpu
  --set kvmem.archive_io_max_bytes=134217728

  # sessions
  --set kvmem.session_capacity=1
  --set kvmem.session_checkpoint_bytes=1073741824
  --set kvmem.session_checkpoint_policy=pm
  --set kvmem.session_capture_terminal=true
  --set auto_session=true
)

if [ "$ACTION" = "start" ]; then
  echo "[kvmem] window B+R=40960 ctx=262144 MTP=4 util=0.92 chunk=1024 pool=4G"
  echo "[kvmem] engine log: $RUN_DIR/serve.log  params: $RUN_DIR/launch.json"
  exec "$REPO/deploy/run-kvmem-vllm.sh" start "${OVERRIDES[@]}" "$@"
fi
exec "$REPO/deploy/run-kvmem-vllm.sh" "$ACTION" "$@"
