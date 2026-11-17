# Configuration

All knobs live in one serving profile (`local/*.json`). `deploy/run-kvmem-vllm.sh` reads it, builds
the vLLM command line and exports the plugin environment. Nothing else needs editing.

## The four budgets

| budget | keys | meaning |
|---|---|---|
| GPU window | `kvmem.select_budget` (B), `kvmem.gen_budget` (R) | KV resident for one request |
| logical context | `max_model_len` (top level, = CTX) | longest single conversation |
| host history | `kvmem.cpu_read_cache_bytes`, `kvmem.repository_blocks`, `block_tokens` | total history tokens, shared |
| session checkpoints | `kvmem.session_capacity`, `kvmem.session_checkpoint_bytes` | how many conversations stay warm |

## Capacity formulas

```
KV pages needed          ≈ ceil((B + R) / page_tokens) + 3 × (1 + spec_blocks)
                            # page_tokens = 1,616, page = 26.8 MB, spec_blocks = 4
                            # ⇒ 17,408 B per window token + 0.39 GiB fixed GDN state
history tokens, per rank = min( cpu_read_cache_bytes / 17,408 ,
                                repository_blocks × block_tokens )
host RAM for history     ≈ 2 × history tokens × 17,408     # TP=2 → two archives
checkpoint RAM           ≤ session_checkpoint_bytes        # TOTAL across sessions (60–80 MB each)
```

Measured archive cost per token (this model): **17,408 B/rank** = 17 full-attention layers ×
(4 kv heads / TP 2) × 256 head_dim × 2 (K+V, fp8).

### Worked example — B = 36K, R = 4K (window 40,960), CTX = 256K, one session

| item | value |
|---|---|
| window | 40,960 tokens |
| GPU KV needed | ceil(40,960/1,616) + 15 pages = **41 pages ≈ 1.07 GiB** |
| pool at `gpu_memory_utilization=0.92` | 118 pages ≈ 3.09 GiB (2.9× margin — the window is a product choice, not a limit) |
| history for one 256K session | 262,144 × 17,408 = **4.25 GiB/rank** ⇒ 8.5 GiB host |
| `cpu_read_cache_bytes` | 6 GiB (1.41× margin) |

## Key knobs

| key | default here | notes |
|---|---|---|
| `max_model_len` | 262,144 | logical CTX; must be ≥ B+R; the launcher allows up to 10,000,000 for MTP (needs `request_planning` and `B+R ≤ 262,144`) |
| `gpu_memory_utilization` | **0.92** | weights + activation + graphs + KV pool; 0.90 leaves ~2.8 GiB free for the long-prefill transient (at 0.96 a max-length prefill hit CUDA OOM) |
| `kvmem.select_budget` | 36,864 | selected history per step; must be divisible by `block_tokens`, ≥ sink+recent |
| `kvmem.gen_budget` | 4,096 | generation headroom; also the model's per-request output reserve |
| `kvmem.block_tokens` | 32 | selection grid; sink/recent/`repository_blocks` are counted in these |
| `kvmem.sink_blocks` / `recent_blocks` | 128 / 64 | 4,096 / 2,048 tokens kept unconditionally |
| `kvmem.repository_blocks` | 16,384 | × 32 = 524,288 nominal tokens |
| `kvmem.cpu_read_cache_bytes` | 6 GiB | **the real history limit**, shared by all sessions |
| `query_replay_budget` (top level) | 4,096 | defaults to R; costs ~180 KiB VRAM per token — always set it |
| `kvmem.session_capacity` | 1 | retained conversations (old adapter: `SESSIONS`) |
| `kvmem.session_checkpoint_bytes` | 1 GiB | **total** for all retained checkpoints |
| `kvmem.session_checkpoint_policy` | `pm` | P + terminal checkpoints; required for continuation hits |
| `execution_mode` | `full_graph` | SM75 recipe: `cudagraph_mode=FULL` + `cudagraph_capture_sizes=[K+1]` |
| `draft_tokens` | 4 | MTP drafts; acceptance decides the win (code ≈ 4/5, prose ≈ 2/5) |
| `cudagraph_capture_sizes` | `[5]` | must equal `{query_len × batch}` for batch 1..`max_num_seqs` |
| `prefill_chunk_tokens` / `max_num_batched_tokens` | 2,048 | logits buffer ≈ 1.0 GB; 4,096 ⇒ 1.94 GB and a max-length prefill can OOM at high utilization |
| `max_num_seqs` | 1 | windowed requests are serialized anyway; raise only for spare authority slots |
| `auto_session` | `true` | derive `session_id` from the conversation opening; no frontend change |

### Mapping from the old adapter (v0.5)

| old | new | unit change |
|---|---|---|
| `B` | `kvmem.select_budget` | tokens (same meaning) |
| `R` | `kvmem.gen_budget` | tokens |
| `CTX` | `max_model_len` | tokens |
| `SINK` / `RECENT` | `sink_blocks` / `recent_blocks` | **tokens → blocks of `block_tokens`** (8,192 tok = 256 blocks) |
| `HOST_MIB` | `cpu_read_cache_bytes` + `repository_blocks` | one pool → bytes **and** blocks; bytes bind first |
| `SESSIONS` | `session_capacity` | same |
| `SPECULATIVE_TOKENS` | `draft_tokens` | same |
| `GPU_UTIL` | `gpu_memory_utilization` | same |

## Generation past the window

When the resident window fills (`B+R−block_tokens`), the plan returns `pressure` and KVMem runs a
bounded refresh: append generated tokens to history, re-select B tokens from the full history
(sink + recent + scored middle), rebuild only the delta, keep the same request and sampler.

Measured (B=4,096 / R=2,048, window 6,144, 9,000-token output): 2 refreshes, each rebuilding
~2,600–3,100 tokens (~3 s at 1,000 tok/s), total **100.5 tok/s**, no stall.

Refresh cadence during decode ≈ **R** (not B+R), because B is already full once history is selected.
With R = 16,384 that is one refresh per ~16K generated tokens.

`rolling_active_window=true` replaces this with asynchronous refresh quanta; it requires
`async_refresh=async_restore=true`, `cpu_history_coordination=true`, `maintenance_max_steps>0`,
`session_capture_terminal=true`, accepts only token-ID completions, and measured the same throughput.

## Tuning recipes

| goal | change |
|---|---|
| faster code generation | `draft_tokens=4` + `cudagraph_capture_sizes=[5]` + `execution_mode=full_graph` (80–87 tok/s verified) |
| longer logical context | raise `max_model_len` and `cpu_read_cache_bytes` together; keep `sessions × CTX ≤ history tokens` |
| more warm sessions | `session_capacity` up **and** `cpu_read_cache_bytes` up by 4.25 GiB/rank per extra 256K session |
| free VRAM for another service | lower `gpu_memory_utilization` (0.80 leaves ~4.7 GB) and/or R |
| avoid a hard failure when sessions overflow | `KVMEM_SESSION_IDLE_LRU_EVICTION=1` + small `session_capacity` |

## Beyond 262,144 tokens

The logical context is **not** capped at 262,144. Measured on this host with
`max_model_len = 524,288`, `B = R = 8,192`, `gpu_memory_utilization = 0.90`:

| prompt | result |
|---|---|
| 269,654 tokens (needle at the middle) | answered `ZX-4417-QJ7` correctly, 686 s, 393 tok/s prefill |
| 299,280 tokens | `FINISHED_STOPPED`, no OOM |

Two caveats, both measured:

1. **VRAM headroom, not context, is the wall.** The same 300K prompt failed with CUDA OOM when the
   window was 49,152 at `gpu_memory_utilization` 0.96/0.88: the pool (3 × window) plus the prefill
   transient filled the card. Shrinking the window (16,384 total) or the utilization leaves the room.
2. **The checkpoint declares 262,144 positions** (`rope_type=default`); beyond that RoPE is
   extrapolated. It returns the right answer in our needle tests, but long-range quality is unverified.

Cost: prefill runs at ~390–1,000 tok/s depending on the window, so a 270K prompt takes ~11 minutes.

## Window capacity vs utilization (measured on 2 × 22 GB)

The pool holds `pages × 1,616` attention tokens; the 15 pages of GDN state are fixed.

| `gpu_memory_utilization` | pool | usable window | code-gen decode | max-length (258K) prefill |
|---|---|---|---|---|
| 0.82 | 70,690 tok (43 pages) | ~45K | — | not tested |
| **0.90 (default)** | 167,889 tok (103 pages) | **142K** | 80.1 tok/s @131K window | ✅ 1,017 s |
| 0.92 | 191,453 tok (118 pages) | 166K | — | — |
| **0.96** | 240,053 tok (148 pages) | **215K** | **80.0 tok/s @200,704 window** | ❌ CUDA OOM (engine died) |

Reading: a 200K window boots and decodes at full speed at 0.96 — the old fork's 0.972 → ~201K pool is
matched or beaten — but the **prefill transient of a maximum-length prompt no longer fits** at that
utilization. 0.90 with a 131K window is the setting that survives both. Choose by workload:
long inputs → 0.90; long windows for many short requests → 0.96.

## Overrides without editing the profile

```bash
./deploy/run-kvmem-vllm.sh start \
  --set gpu_memory_utilization=0.92 \
  --set kvmem.select_budget=36864 \
  --set kvmem.gen_budget=4096
```

`--set KEY=VALUE` is applied after the profile is loaded (and after `${VAR}` expansion); dotted paths
reach nested objects, JSON values keep their type, and an unknown key is rejected. This is how a machine
launcher can keep local differences while the committed profile stays the reference default
(see `deploy/example-local-launcher.sh`).
