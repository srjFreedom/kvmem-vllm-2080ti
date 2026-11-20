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
| pool at `kv_cache_memory_bytes=3.5 GiB` | ~120 pages ≈ 3.5 GiB (3.3× margin — the window is a product choice, not a limit) |
| history for one 256K session | 262,144 × 17,408 = **4.25 GiB/rank** ⇒ 8.5 GiB host |
| `cpu_read_cache_bytes` | 6 GiB (1.41× margin) |

## Key knobs

| key | default here | notes |
|---|---|---|
| `max_model_len` | 262,144 | logical CTX; must be ≥ B+R; the launcher allows up to 10,000,000 for MTP (needs `request_planning` and `B+R ≤ 262,144`) |
| `kv_cache_memory_bytes` | **3,758,096,384 (3.5 GiB)** | **the knob that actually sizes the KV pool.** When set, vLLM reserves exactly this many bytes, skips memory profiling, and **ignores `gpu_memory_utilization`** (the engine log says so verbatim). This is the production setting. |
| `gpu_memory_utilization` | 0.92 | **a no-op while `kv_cache_memory_bytes` is set** — vLLM skips profiling and never reads it. It only takes effect if you remove `kv_cache_memory_bytes`, in which case it sizes the pool as a fraction of free VRAM (0.90 leaves ~2.8 GiB for the long-prefill transient; 0.96 OOMs a max-length prefill). Do not tune it expecting a window change. |
| `kvmem.select_budget` | 36,864 | selected history per step; must be divisible by `block_tokens`, ≥ sink+recent |
| `kvmem.sink_blocks` / `kvmem.recent_blocks` | 256 / 64 | 8,192 / 2,048 tokens head and tail kept on every retrieval, prefill and decode alike; the middle band is scored |
| `kvmem.gen_budget` | 8,192 | generation headroom; also the model's per-request output reserve. Keep it at 8,192 or above: after a refresh the window slides by `R − lookahead − block_tokens`, so a smaller R fills the window every ~2K tokens and turns prefill into a refresh loop |
| `kvmem.block_tokens` | 32 | selection grid; sink/recent/`repository_blocks` are counted in these |
| `kvmem.sink_blocks` / `recent_blocks` | 64 / 32 | 2,048 / 1,024 tokens kept unconditionally |
| `kvmem.repository_blocks` | 16,384 | × 32 = 524,288 nominal tokens |
| `kvmem.cpu_read_cache_bytes` | 6 GiB | **the real history limit**, shared by all sessions |
| `query_replay_budget` (top level) | 4,096 | defaults to R; costs ~180 KiB VRAM per token — always set it |
| `kvmem.session_capacity` | 1 | retained conversations (old adapter: `SESSIONS`) |
| `kvmem.session_checkpoint_bytes` | 1 GiB | **total** for all retained checkpoints |
| `kvmem.session_checkpoint_policy` | `pm` | P + terminal checkpoints; required for continuation hits |
| `execution_mode` | `full_graph` | SM75 recipe: `cudagraph_mode=FULL` + `cudagraph_capture_sizes=[K+1]` |
| `draft_tokens` | 4 | MTP drafts; acceptance decides the win (code ≈ 4/5, prose ≈ 2/5) |
| `cudagraph_capture_sizes` | *derived* | leave unset: derived as `(draft_tokens + 1) × 1..max_num_seqs`, which is exactly the set of reachable decode batch sizes. An explicit list is still honoured but must match, or decode silently drops to eager |
| `prefill_chunk_tokens` / `max_num_batched_tokens` | 2,048 | logits buffer ≈ 1.0 GB; 4,096 ⇒ 1.94 GB and a max-length prefill can OOM at high utilization |
| `chat_prefill_blocks` | 32 (1,024 tokens at block 32) | tokens per native-chat prefill step; a chat request carries no request plan, so `prefill_chunk_tokens` never sized it. Bigger steps cut the forward count but publish more per window refresh; 1,024 tokens measures 15–17% faster than 256 on 8K/32K cold prefill and ~5% slower above ~130K. On the 3.5 GiB pool 2,048 is safe and ~1.6–3.7% faster than 1,024 at 200K/8K (2026-11-20 retest); the earlier "2,048 kills the engine on a 200K prompt" was a 4 GiB-pool headroom artifact, not a structural limit |
| `max_num_seqs` | 1 | windowed requests are serialized anyway; raise only for spare authority slots |
| `auto_session` | `true` | derive `session_id` from the conversation opening; no frontend change |
| `generation_config` | `vllm` (repo) / `auto` (launcher) | `auto` applies the checkpoint's `generation_config.json` to requests that omit sampling params; `vllm` keeps vLLM's neutral defaults (temperature 1.0, which degenerates long generations on reasoning models) |

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

## Vision / multimodal (opt-in)

The default profile is **text-only**: the vision tower is not loaded and image inputs are rejected.
Image understanding is a launch choice, enabled by two top-level keys (not in the committed profile —
add them to your profile or pass them with `--set`):

| key | default | meaning |
|---|---|---|
| `mm_image_limit` | `0` | `0` keeps `--language-model-only` (no vision tower, text only). A positive value sets `--limit-mm-per-prompt` for images and loads the vision tower; `64` is the tested value. Video and audio stay `0`. |
| `mm_gdn_prefill_backend` | `null` | `null` lets vLLM pick the GDN prefill backend. `flashqla_legacy` selects the SM75 FlashQLA backend, which the hybrid (GDN + full-attention) checkpoint needs on this hardware. |

Where it runs and at what precision (measured on this checkpoint):

- **On the GPU**, in the same workers as the language model (TP0/TP1), as part of prefill — not offloaded to the host. Image → patch embed → 27 ViT blocks → merger → visual embeddings, then scattered into the KV window.
- **Vision tower: BF16.** Its weights ship in a separate `vision-bf16.safetensors` and the whole `model.visual.*` is excluded from FP8 quantization (`modules_to_not_convert`), so it loads at native BF16. The language model stays FP8. Keeping the ViT unquantized is deliberate — quantizing it measurably degrades image understanding.
- **Dynamic resolution, no fixed input size** (`Qwen2VLImageProcessorFast`). Each visual token covers 32×32 px (`patch_size 16` × `merge_size 2`); the image is scaled so its total pixel count falls in **[65,536, 16,777,216]** with both edges on a multiple of 32. That is 64 visual tokens at 256×256 up to ~16,384 at 4096×4096, any aspect ratio in between. The encoder cache is budgeted at 16,384 tokens, so one maximum-size image fits exactly.

Image conversations reuse sessions exactly like text ones: the visual embeddings enter the KVMem window and M-RoPE positions keep vLLM's native per-axis values, so a picture mid-conversation is retrieved on a later refresh like any other token.

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
| free VRAM for another service | lower `kv_cache_memory_bytes` (the effective pool knob; `gpu_memory_utilization` is ignored while it is set) and/or R |
| avoid a hard failure when sessions overflow | `KVMEM_SESSION_IDLE_LRU_EVICTION=1` + small `session_capacity` |

## Beyond 262,144 tokens

The logical context is **not** capped at 262,144. Measured on this host with
`max_model_len = 524,288`, `B = R = 8,192`, `gpu_memory_utilization = 0.90` (utilization-controlled
mode — `kv_cache_memory_bytes` unset, so the utilization fraction actually sized the pool):

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

> **These rows only apply when `kv_cache_memory_bytes` is unset**, so that
> `gpu_memory_utilization` is the knob sizing the pool. The production profile sets
> `kv_cache_memory_bytes = 3.5 GiB`, which **fixes the pool and ignores
> `gpu_memory_utilization` entirely** (vLLM skips profiling and never reads it). To move the
> window in production, change `kv_cache_memory_bytes`, not the utilization. The table below is
> kept as the reference for what each utilization fraction would buy if you drop the fixed pool.

The pool holds `pages × 1,616` attention tokens; the 15 pages of GDN state are fixed.

| `gpu_memory_utilization` | pool | usable window | code-gen decode | max-length (258K) prefill |
|---|---|---|---|---|
| 0.82 | 70,690 tok (43 pages) | ~45K | — | not tested |
| **0.90 (default)** | 167,889 tok (103 pages) | **142K** | 80.1 tok/s @131K window | ✅ 1,017 s |
| 0.92 | 191,453 tok (118 pages) | 166K | — | — |
| **0.96** | 240,053 tok (148 pages) | **215K** | **80.0 tok/s @200,704 window** | ❌ CUDA OOM (engine died) |

At 0.96 a 200K window starts and decodes at full speed, but the prefill transient of a maximum-length
prompt no longer fits. 0.90 with a 131K window survives both. Choose by workload: long inputs → 0.90;
large windows for many short requests → 0.96.

## Overrides without editing the profile

```bash
./deploy/run-kvmem-vllm.sh start \
  --set kv_cache_memory_bytes=3758096384 \
  --set kvmem.select_budget=36864 \
  --set kvmem.gen_budget=4096
```

`--set KEY=VALUE` is applied after the profile is loaded (and after `${VAR}` expansion); dotted paths
reach nested objects, JSON values keep their type, and an unknown key is rejected. This is how a machine
launcher can keep local differences while the committed profile stays the reference default
(see `deploy/example-local-launcher.sh`).

## Common failures

| symptom | cause | fix |
|---|---|---|
| `ValueError: GPU page pool cannot fit one worst-case KVMem request (admission tokens=40960 …)` at boot | pool smaller than one window plus headroom | raise `kv_cache_memory_bytes`; ~1.5 GiB is the smallest pool that boots, below ~0.8 GiB admission is refused |
| `memory allocation failed … retry` or `CUDA out of memory … 71303168 bytes` during a long prefill | pool pinned above the free VRAM left for the prefill transient, or no VMM allocator | pin 4 GiB with `prefill_chunk_tokens=1024` and `query_replay_budget=2048`, plus `enable_cumem_allocator=true` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| `custom_all_reduce … 'invalid argument'` after enabling expandable segments | VMM segments break the custom all-reduce setup | `disable_custom_all_reduce=true` (NCCL; ~5% of decode throughput) |
| engine refuses the expandable-segments environment | the VMM allocator is off | add `enable_cumem_allocator=true` |
| long answers drift into repetition and MTP acceptance falls to ≈1.0 | the client omits sampling parameters, so vLLM's default (temperature 1.0) applies | `generation_config=auto` with a sane checkpoint `generation_config.json` |
| `400 Explicit Query coordinates require a token-ID completion prompt` | KVMem query coordinates were sent through the chat endpoint | send token IDs to `/v1/completions`, or let the scheduler plan the query |
| `KVMem: skipping the KV capacity check (… pool has X GiB)` | expected: `max_model_len` intentionally exceeds the pool | informational; the window, not the logical context, has to fit |
| an answer starts, then nothing arrives for minutes and the turn ends empty or the client reports a transport error | the stream went silent while the request kept decoding: a degenerating model loops inside an output structure the parser holds back, so no SSE byte reaches the client | the `stream_watchdog` middleware ends the stream after 60 s of body silence and cancels the request (client retries); set `{"stream_watchdog": {"idle_seconds": 0}}` to keep the old behaviour |
