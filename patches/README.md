# Patch set

The patches are plain `git diff` output. Apply them in order on top of the commits below.

| file | base | applies to | size |
|---|---|---|---|
| `0001-engine-skip-kv-check.patch` | engine `93c9013e37` (vLLM `v0.2.2-post3`) | `vLLM-2080Ti-Definitive` | 15 lines |
| `0002-kvmem-sm75-compat.patch` | plugin `8654700` (upstream `kvmem/kvmem-vllm`) | this repository's `src/` and `local/` | ~20 KB |
| `0003-engine-dflash2-w4a16-draft.patch` | engine `01506a5471` (with `0001` applied) | `vLLM-2080Ti-Definitive` | ~840 lines |
| `0004-engine-dflash2-no-apc-gates.patch` | engine `01506a5471` (after `0003`) | `vLLM-2080Ti-Definitive` | ~70 lines |
| `0005-engine-dflash2-pool-floors.patch` | engine `01506a5471` (after `0004`) | `vLLM-2080Ti-Definitive` | ~65 lines |

```bash
# engine (in order; 0003/0004 are only needed for the DFlash2 route)
cd /path/to/vLLM-2080Ti-Definitive
git apply /path/to/patches/0001-engine-skip-kv-check.patch
git apply /path/to/patches/0003-engine-dflash2-w4a16-draft.patch
git apply /path/to/patches/0004-engine-dflash2-no-apc-gates.patch
git apply /path/to/patches/0005-engine-dflash2-pool-floors.patch

# plugin (only when starting from upstream kvmem-vllm)
cd /path/to/kvmem-vllm-2080ti
git apply /path/to/patches/0002-kvmem-sm75-compat.patch
uv pip install --python /path/to/engine/.venv/bin/python -e .
```

`0001` adds a `KVMEM_SKIP_KV_CHECK` guard so the engine starts when `max_model_len` exceeds what the KV
pool can hold — required because KVMem's physical pool is the window (B + R), not the logical context.

`0002` contains the whole SM75/float16 compatibility layer, the auto-session identifier, the guided
page-pool demand fix, the session-miss diagnostics, and the serving profiles.

Verified with these exact patches on 2 × RTX 2080 Ti: see
[`docs/sm75/VERIFICATION.md`](../docs/sm75/VERIFICATION.md).

`0003` is [`weicj/vLLM-2080Ti-Definitive#265`](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/265)
applied on our base: it lets a quantized draft expose its K/V rows (through the quantization kernel
instead of a dense `.weight`), adds the kernel-side `dequantized_weight` row slice, and adds the
companion `dflash2_sm75.py` fixes and the regression test. Without it the **W4A16 draft fails at load**
with `AttributeError: 'QKVParallelLinear' object has no attribute 'weight'`.

`0004` is ours: three opt-in environment gates on the SM75 DFlash2 route.

| gate | what it unlocks |
|---|---|
| `VLLM_DFLASH_NATIVE_GROUPS_WITHOUT_APC=1` | native per-layer draft groups while APC is off (KVMem's bounded route requires APC off) |
| `VLLM_DFLASH_DRAFT_PAGE_ALIGN=1` | draft pages sized like the attention page when APC is off (64-token draft pages cost ~6 ms/step on the engine alone) |
| `VLLM_MAMBA_NONE_SMALL_BLOCK=1` | `--mamba-block-size` without `--enable-prefix-caching` |

`0004` also teaches `_dflash_group_capacities` to honour `KVMEM_SKIP_KV_CHECK`: without it the
DFlash2 route refuses to start whenever `max_model_len` exceeds what the pool can hold, which is
exactly the MTP-style layout KVMem uses everywhere else. With the clamp the draft pools are sized from
the budget and the window covers the difference — measured **slower than the bounded route**
(128K unbounded: 96 tok/s code, 168 repeat; bounded 262K: 106 / 197), so the bounded planner stays the
recommended layout for DFlash2.

The DFlash2 route also needs `VLLM_KV_CACHE_LAYOUT=BLHNC` at runtime (mixed target/draft page sizes) —
that is a flag, not a patch. See the [DFlash2 route](../README.md#dflash2-route) section for the
route's boundaries and for the draft we run.

`0005` fixes the DFlash2 pool sizing itself. Three floors were missing, and each one
made a pool smaller than a single decode step needs:

| pool | was | now |
|---|---|---|
| recurrent (GDN) | `cdiv(max_memory_usage_bytes / page)` = 1 page → 6 blocks | `2 + num_speculative_blocks` (live state + speculation + the pool's null block) |
| draft ring | sized by the sliding window → 52 blocks | sized by `max_in_flight_tokens + lookahead` → 67 blocks, matching what KVMem reserves |
| budget clamp | scaled every pool down proportionally, wiping the floors | reserves the recurrent and draft floors first, then scales the position-indexed pool |

With them the route runs **without the bounded planner**: `max_model_len=262144`, a 4 GiB
pool, `mamba_block_size=1648`, no APC and no static window reservation — i.e. the MTP-style
layout. Measured on this host: code **93.6 tok/s**, repeat **170.6 tok/s**, chat native. The
bounded planner is still the faster layout (106 / 197), so it stays the recommended one; this
patch makes the MTP-style layout possible rather than preferable.
