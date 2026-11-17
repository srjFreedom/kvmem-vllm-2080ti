# Architecture

## Two layers, one process tree

```
python -m vllm_kvmem serve --config local/serve-2080ti-default.json
        │
        ├── builds the vLLM CLI      →  <engine venv>/bin/vllm serve … --max-model-len 262144
        │                                 --scheduler-cls vllm_kvmem.serving.scheduler.KVMemScheduler
        │                                 --kv-transfer-config {KVMemConnector}
        │                                 --kv-cache-dtype fp8_e4m3  (KVMem requires this spelling)
        │
        └── exports the environment   →  VLLM_PLUGINS=kvmem
                                         KVMEM_REQUEST_PLAN_CONFIG={block_tokens, B, R, sink, recent, …}
                                         KVMEM_WINDOW_TOKENS=B+R
                                         KVMEM_SKIP_KV_CHECK=1     (engine patch reads this)
```

The engine (vLLM) owns execution; the plugin owns *what KV is resident*. They meet at three seams:
the scheduler class, the KV connector, and a handful of environment variables.

## Request lifecycle

1. **Admission** — the middleware derives a session namespace (`auto-` + hash of the conversation
   opening when the client sends no `session_id`), the planner turns the request into a plan
   (`select_budget`, `gen_budget`, query span, sink/recent bands).
2. **Prefill** — chunked. Every KV page is written to GPU, then handed to the archive
   (`capture_runtime`), so the full history exists in host RAM even though the GPU keeps a window.
3. **Selection** — before decode, KVMem selects the window: sink head + recent tail + scored middle
   (`mean_k` over stored keys), bounded by `select_budget`.
4. **Decode** — generation writes into the `gen_budget` headroom.
5. **Boundary** — when the window fills (`B+R−block_tokens`) the plan returns `pressure` and KVMem
   runs a **bounded refresh**: append the new tokens to history, re-select, rebuild only the delta,
   keep the same request/sampler. See [CONFIGURATION.md](CONFIGURATION.md#generation-past-the-window).
6. **Retirement** — the request's session becomes an idle checkpoint (P at the prompt boundary,
   M at the terminal position when the policy is `pm`), ready for the next turn.

## Session modes

| mode | settings | behaviour |
|---|---|---|
| frozen-P | `policy=sealed` + `session_freeze_prompt=true` | one frozen prompt checkpoint; good for *independent questions sharing a prefix*; **continuations that need a P replay miss** |
| P/M | `policy=pm` + `session_capture_terminal=true` | keeps P and terminal checkpoints; **continuations inside and beyond the window hit** (this is the default here) |

The default uses `pm`; it costs one extra terminal snapshot per session (measured 60–80 MB each).

## Memory model

| resource | consumed by | bounded by |
|---|---|---|
| VRAM, weights | model | ~15.4 GiB/GPU for the 27B FP8 hybrid |
| VRAM, KV pool | 3 attention cache groups × window | `gpu_memory_utilization`, `B+R` |
| VRAM, query replay buffers | ~180 KiB per `query_replay_budget` token | set it explicitly (default = R) |
| VRAM, CUDA graphs | capture sizes | `cudagraph_capture_sizes` |
| host RAM, history archive | one arena, shared by all sessions | `cpu_read_cache_bytes` (bytes) and `repository_blocks` (blocks); the bytes bind first |
| host RAM, session checkpoints | P/M snapshots | `session_checkpoint_bytes` **total**, not per session |

Per-token archive size comes from `adapters/cache_manifest.py::archive_bytes_per_token`: only
full-attention layers are stored, `(kv_heads/TP) × head_dim × 2` bytes each, **per rank**.
For this model: `17 × 1,024 = 17,408 B/token/rank`; TP = 2 ⇒ double it for host RAM.

## Why the engine needed a patch

vLLM rejects a start where `max_model_len` needs more KV than the pool holds. That check is correct for
a normal server and wrong for KVMem: the context is logical, the pool only has to hold the window.
`patches/0001-engine-skip-kv-check.patch` adds a `KVMEM_SKIP_KV_CHECK` guard around that check — 15
lines, no behaviour change when the variable is unset.
