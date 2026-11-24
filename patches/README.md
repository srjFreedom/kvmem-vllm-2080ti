# Patch set

The patches are plain `git diff` output. Apply them in order on top of the commits below.

| file | base | applies to | size |
|---|---|---|---|
| `0001-engine-skip-kv-check.patch` | engine `93c9013e37` (vLLM `v0.2.2-post3`) | `vLLM-2080Ti-Definitive` | 15 lines |
| `0002-kvmem-sm75-compat.patch` | plugin `8654700` (upstream `kvmem/kvmem-vllm`) | this repository's `src/` and `local/` | ~20 KB |
| `0003-engine-dflash2-w4a16-draft.patch` | engine `01506a5471` (with `0001` applied) | `vLLM-2080Ti-Definitive` | ~840 lines |
| `0004-engine-dflash2-no-apc-gates.patch` | engine `01506a5471` (after `0003`) | `vLLM-2080Ti-Definitive` | ~70 lines |

```bash
# engine (in order; 0003/0004 are only needed for the DFlash2 route)
cd /path/to/vLLM-2080Ti-Definitive
git apply /path/to/patches/0001-engine-skip-kv-check.patch
git apply /path/to/patches/0003-engine-dflash2-w4a16-draft.patch
git apply /path/to/patches/0004-engine-dflash2-no-apc-gates.patch

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

The DFlash2 route also needs `VLLM_KV_CACHE_LAYOUT=BLHNC` at runtime (mixed target/draft page sizes) —
that is a flag, not a patch. See the [DFlash2 route](../README.md#dflash2-route) section for the
route's boundaries and for the draft we run.
