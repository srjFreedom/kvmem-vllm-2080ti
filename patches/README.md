# Patch set

The patches are plain `git diff` output. Apply them in order on top of the commits below.

| file | base | applies to | size |
|---|---|---|---|
| `0001-engine-skip-kv-check.patch` | engine `93c9013e37` (vLLM `v0.2.2-post3`) | `vLLM-2080Ti-Definitive` | 15 lines |
| `0002-kvmem-sm75-compat.patch` | plugin `8654700` (upstream `kvmem/kvmem-vllm`) | this repository's `src/` and `local/` | ~20 KB |
| `0003-engine-dflash2-support.patch` | engine `01506a5471` (with `0001` applied) | `vLLM-2080Ti-Definitive` | ~960 lines |
| `0006-engine-dflash2-block-align.patch` | engine `9a363037a9` (after `0003`) | `vLLM-2080Ti-Definitive` | ~90 lines |

```bash
# engine (in order; 0003/0006 are only needed for the DFlash2 route)
cd /path/to/vLLM-2080Ti-Definitive
git apply /path/to/patches/0001-engine-skip-kv-check.patch
git apply /path/to/patches/0003-engine-dflash2-support.patch
git apply /path/to/patches/0006-engine-dflash2-block-align.patch

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

`0003` is the whole DFlash2 route as one patch: the W4A16 draft support from
[`weicj/vLLM-2080Ti-Definitive#265`](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/265) (a quantized
draft exposes its K/V rows through the quantization kernel instead of a dense `.weight`, plus the
kernel-side `dequantized_weight` row slice, the `dflash2_sm75.py` fixes and the regression test), the
APC-off gates (`VLLM_DFLASH_NATIVE_GROUPS_WITHOUT_APC`, `VLLM_DFLASH_DRAFT_PAGE_ALIGN`,
`VLLM_MAMBA_NONE_SMALL_BLOCK`), and the pool floors. Without it the **W4A16 draft fails at load** with
`AttributeError: 'QKVParallelLinear' object has no attribute 'weight'`.

`0006` is what makes the DFlash2 lane serve two requests at once. The draft used to keep a private
64-token block ring, so every request needed a 4,224-token ring of its own (66 of 67 pages); the
second request was rejected on every decode step once both draft contexts approached the ring limit,
and the engine spun forever with `running=2`, KV 100 % and GPU 0 %. The patch gives the draft the
target's manager block (1,648) so a request costs 4 draft pages instead of 66, floors the
full-attention group at one resident window per concurrent request, and keeps one spare recurrent
page per request. The coarsening helper that already existed could not do this: it may only multiply
the draft block by a factor dividing the target alignment, and `1,648 = 16 × 103` has no multiple
of 64.
