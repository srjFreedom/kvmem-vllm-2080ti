# SM75（双 2080Ti）补丁集

两个补丁合起来 = 让上游 kvmem-vllm 在 2080Ti 上跑起来所需的**全部**本地改动。

| 补丁 | 作用仓库 | 基线 |
|---|---|---|
| `0001-engine-kvmem-sm75-runtime.patch` | vLLM fork `/home/srj/vLLM-2080Ti-Definitive`（分支 `kvmem-0.2.2-clean`） | 应用前 HEAD = `6b72693be0c676c4c014e9978ad7c20f186bc8d6`，应用后 = `175950def30f1dd7b2e19fe67ffa521034ff67d0` |
| `0002-kvmem-sm75-compat.patch` | 本仓库（上游 `kvmem/kvmem-vllm` @ `8654700`） | 本仓库 `git diff 8654700..HEAD` |

## 应用顺序

```bash
# 1) 引擎侧（必须在引擎仓库根目录）
cd /home/srj/vLLM-2080Ti-Definitive
git apply --check /home/srj/src/kvmem-vllm-2080ti/patches/0001-engine-kvmem-sm75-runtime.patch
git apply         /home/srj/src/kvmem-vllm-2080ti/patches/0001-engine-kvmem-sm75-runtime.patch

# 2) kvmem 侧（在本仓库根目录）
cd /home/srj/src/kvmem-vllm-2080ti
git apply --check patches/0002-kvmem-sm75-compat.patch   # 已包含在 HEAD 里，仅用于回放/审阅
```

## 内容摘要

**0001 引擎侧（3 文件，+334/−3）**
- `vllm/v1/core/sched/scheduler.py`：KVMem 的双账本（`num_logical_tokens` 已服务 / `num_computed_tokens` 常驻）、
  `kvmem_page_tokens`、`kvmem_true_computed`、`kvmem_window_tokens`
- `vllm/v1/worker/gpu/model_runner.py`：positions/seq_lens 之后的 kvmem ring 钩子（`enabled/batch` 判定）
- `vllm/v1/worker/gpu/kvmem_ring.py`（新增 296 行）：页环维护，发布 `kvmem_view_lens` / 位置偏移

**0002 kvmem 侧（6 文件）**
- `compat_sm75/`：block table 形状兼容（`num_blocks_per_row` / 只有设备行）+ capture runner 晚绑定
- `core/native_pages.py`：只有设备行的表按恒等扩展
- `serving/serve.py` + `DEFAULTS`：`--dtype` 由 profile 决定（SM75 无 bf16）
- `adapters/capture.py` / `serving/capture_runtime.py`：capture 缓冲与激活 dtype 跟随模型（fp16）
- `local/serve-2080ti-smoke.json`：本机启动 profile

## 已验证（2026-11-16）
双 2080Ti、`dtype=float16`、`kv-cache-dtype=fp8_e4m3`、TP=2、MTP1、`enforce-eager`：
4,888 token 的 needle prompt **11.1 s** 答对 `ZX-4417-QJ7`；run-dir 里 `archive.json`（batches=131）等正常生成。
