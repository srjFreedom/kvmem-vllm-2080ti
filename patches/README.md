# Patch set

Both patches are plain `git format-patch` output. Apply them in order on top of the commits below.

| file | base | applies to | size |
|---|---|---|---|
| `0001-engine-skip-kv-check.patch` | engine `93c9013e37` (vLLM `v0.2.2-post3`) | `vLLM-2080Ti-Definitive` | 15 lines |
| `0002-kvmem-sm75-compat.patch` | plugin `8654700` (upstream `kvmem/kvmem-vllm`) | this repository's `src/` and `local/` | ~20 KB |

```bash
# engine
cd /path/to/vLLM-2080Ti-Definitive
git apply /path/to/patches/0001-engine-skip-kv-check.patch

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
