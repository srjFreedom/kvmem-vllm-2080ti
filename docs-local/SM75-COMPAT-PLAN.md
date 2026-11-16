# SM75（2080Ti）兼容计划 —— 以最新上游 kvmem-vllm 为准

创建：2026-11-16　仓库：`/home/srj/src/kvmem-vllm-2080ti`（克隆自 `github.com/kvmem/kvmem-vllm`，HEAD `8654700`）
目标运行时：`/home/srj/vLLM-2080Ti-Definitive`（分支 `kvmem-0.2.2-clean`，vllm `0.2.2.post3`，torch 2.13.0+cu130，py 3.12.15）

## 1. 已经用实测确认的事（本轮证据）

| 项目 | 结果 |
|---|---|
| 上游 Python 包能否在本机导入 | **可以**：`import vllm_kvmem` 成功；“OK”的有 `serving.scheduler`、`adapters.assembly`、`core.retrieval`、`serving.window_batch`、`apps.planner`、`strategy` |
| 上游依赖的 8 个 vLLM 模块 | fork **全部存在**（`v1/core/sched/scheduler`、`v1/core/single_type_kv_cache_manager`、`v1/kv_cache_interface`、`v1/request`、`v1/worker/gpu_worker`、`kv_transfer/...base`、`kv_transfer/...offloading_connector`、`triton_utils`） |
| 关键符号 | `CUDAGraphMode`、`OffloadingConnector`、`get_kv_transfer_group`、`Qwen3NextAttention`、`QwenGatedDeltaNetAttention`、`is_conv_state_dim_first`、`kernel_warmup`、`NewRequestData`、`tokenizers.get_tokenizer`、`triton_utils.tl/triton` **全部存在** |
| native 扩展 | `setup.py build_ext --inplace` **构建成功**（`_native.so` 2.9M、`_prefill_state_native.so` 429K）——这两个扩展是**纯 C++（CPU）**，GPU 内核走运行时 JIT |
| `register()` | 存在且可调用 |
| CPU 测试子集 | `test_action_attention.py` + `test_action_metadata.py` + `test_action_profile.py`：**38 通过 / 2 失败** |

## 2. 失败的 2 个测试暴露的 API 漂移（要修/要适配）

1. `CaptureRuntime` 缺 `.runner`（`serving/capture_runtime.py:308`）——上游对新版 runner 的属性假设与 fork 不一致。
2. `runner.block_tables` 缺 `.num_blocks`（`core/native_pages.py:56`）——fork 的 block table 形状/字段与 vLLM 0.30 不同。

⇒ 结论：**Python API 大面积兼容，但少量形状假设有漂移**，需要一层薄的 `compat` 适配。

## 3. 修正：SM75 不是硬约束 —— fork 已经提供所需的一切（2026-11-16 复核）

原先我把它写成“硬约束”，是把上游的**可选加速路径**当成了必需路径。复核结论：

| 上游组件 | 依赖 | SM75 可行性 |
|---|---|---|
| `native_attention_sources.json`（FA2 / XQA / `mha_sm90.cu` / `mla_sm120.cu`） | 只有 `action_attention` 特性用 | **默认关闭**（`core/config.py: action_attention = False`），且 `serving/action_attention.py:937` 注释明确有“**without the architecture-specific kernels**”的通用路径 ⇒ 不需要 |
| `adapters/assembly.py`（页装配/恢复） | `vllm.triton_utils` 的 Triton kernel | ✅ Triton 支持 SM75 |
| `adapters/_fused_capture.py`（K/Q 捕获） | Triton | ✅ |
| `capture.py` 的 `torch.ops.kvmem.capture_projection` | `csrc/qw3_mean_k*.cu` JIT | ✅ 源码里**没有** `__nv_fp8`/`cp.async`/`wgmma`/`ldmatrix` 等架构敏感特性，nvcc 可编 sm75 |
| FP8 KV | `adapters/fp8_compat.py`：`software_fp8` / `fp8_storage` | ✅ 上游本身就是**软件量化**路径；fork 也支持 `kv_cache_dtype="fp8_e4m3"`（uint8 存储） |
| 模型执行（注意力） | vLLM 后端 | ✅ fork 的 FlashInfer 明确保留 SM75：`supports_compute_capability >= DeviceCapability(7,5)` |
| 权重 FP8 | — | ✅ fork 用 MarlinFP8 / modelopt（cc>=7.5）在 Turing 上做 w8a16 |

⇒ **不需要重写注意力，也不需要移植内核**。SM75 上要做的只是：两个 API 漂移兼容层 + 保持 `action_attention=False` + 用 fork 的 FlashInfer/FP8 KV。

## 4. 分阶段计划

**P0（本轮完成）**：项目落地 + 宿主核心在本机构建/导入 + 基线测试记录。

**P1 compat 薄层**（预计 1–2 轮）
- 增加 `vllm_kvmem/compat_sm75/`：
  - `pages.py`：`block_tables.num_blocks` → 从 fork 的 block table 形状推导（或直接算 pool 页数）；
  - `capture.py`：`CaptureRuntime.runner` 缺失时的兜底（从 forward context / worker 注入）。
- 目标：`test_action_profile.py` 两个失败用例转绿，CPU 子集 40/40。

**P2 KV 编解码（SM75）**（2–3 轮）
- 上游按 FP8 E4M3 KV 设计；SM75 两条路：
  a) **BF16 KV**（最稳，显存翻倍）；
  b) **FP8 按字节存储 + 软件量化/反量化**（旧适配器已验证可行：页 KV 以 uint8 fp8 字节保存，`_quantize/_dequant` 走软件路径，旋转也在该字节域完成）。
- 建议先 (a) 打通正确性，再切 (b) 拿显存。

**P3 attention 路径（SM75）**（1–2 轮，已大幅简化）
- 保持 `action_attention=False`（默认值）：走 `action_attention.py:937` 的通用 prefill + fork 的 FlashInfer。
- KV 页安装/恢复用上游 Triton 的 `adapters/assembly.py`，相位按 rank 旋转、稠密索引。
- 关键契约（上游 `core/memory.py`）：帧必须**有序稠密**、`history_end == checkpoint.position`、**K 和 V 都要重载**、GDN/draft 状态一起恢复。
- 若以后要 `action_attention` 的性能优势，才需要把 XQA 移植到 sm75 —— 那是独立立项，不进 P3。

**P4 端到端**（2–3 轮）
- 目标模型：`Qwen3.8-27B-Coder390-FP8-MTP`（与上游测试的 Qwen hybrid GDN 27B 同族）。
- 验收：小上下文正确性 → 长上下文 needle → 会话前缀命中 → 吞吐对比。

## 5. 现成资产（迁移来源）

- `vLLM-2080Ti-Definitive` 分支 `kvmem-0.2.2-clean` 有 **20 个未推送提交**（已备份 `/home/srj/vllm-fork-2080ti-20261116.bundle`），其中 `a156cf2bfd` 是“kvmem 集成补丁（同步自旧 adapter 仓库）”，P3 的实现可以直接参考/迁移。
- 旧适配器代码已删除，但**已推送到私有仓库** `srjFreedom/kvmem-vllm-adapter`（999 个提交），需要时 `git clone` 取回。

## 6. 待你拍板

1. KV 编解码先走 **BF16（稳）** 还是直接 **FP8 字节（省显存）**？建议先 BF16。
2. 目标模型确认是 `Coder390-FP8-MTP` 吗？（上游只对 Qwen hybrid GDN 27B 有资格记录）
3. 新项目是否也建一个**私有远端**仓库？

## 7. P1 完成（2026-11-16）：API 漂移兼容层落地

新增 `src/vllm_kvmem/compat_sm75/__init__.py`，把 fork 与上游的**形状差异**集中到一处：
- `block_row_count(tables, group, index)`：兼容 `tables.num_blocks.np[g,i]`（上游）
  与 fork 的 `block_tables[g].num_blocks_per_row[i]`，以及只有设备行的 fixture（取行宽）。
- `optional_runner(owner)`：`owner.runner` 未绑定时返回 None（上游下游容忍 None）。

改动点（保持最小、可 grep）：
| 文件 | 改动 |
|---|---|
| `core/native_pages.py` | `kernel_table()` 用 `block_row_count` |
| `core/native_pages.py` | `stored_expansions()` 在表只有设备行时按恒等扩展 |
| `core/native_pages.py` | `kernel_expansions()` 在无 `block_sizes` 时返回恒等扩展 |
| `serving/capture_runtime.py` | `append_qsa_captures(optional_runner(self), ...)` |

**测试基线（本机，fork venv，CPU-only）**
- `tests/test_action_{attention,metadata,profile}.py` ⇒ **40 passed**
- 20 个核心 CPU 测试文件 ⇒ **257 passed / 1 failed**
- 该失败 `test_historical_compact_index.py::test_authority_sealed_sums_release_with_native_gc_and_live_partial_tail`
  是**上游既有失败**（已用 `git stash` 在未改动的树上复现同样失败）：`memory_storage.py:1264`
  `self.tier` 为 None，属环境/上游问题，不是 P1 引入。

## 8. P2 进行中：服务已能启动到 worker 初始化，卡在 capture 投影几何

### 已修（都是 SM75/本机定制点）
| 问题 | 现象 | 处理 |
|---|---|---|
| venv 里残留**旧适配器**的插件注册 | `ModuleNotFoundError: No module named 'vllm_kvmem_plugin'`（该模块随旧项目目录一起删了，但 dist-info/入口点还在） | `uv pip uninstall vllm-kvmem-plugin`（旧包：`vllm_kvmem_plugin-0.1.0`，入口点 `vllm.general_plugins kvmem = vllm_kvmem_plugin:register`） |
| KV 类型被写死 | `serve.py` 硬编码 `--kv-cache-dtype fp8_e4m3`；`capture_runtime.py:115`、`window_assembly.py:443` 非 fp8 直接 `raise` | **结论：KV 必须 fp8_e4m3**（这是*存储格式*，SM75 上走软件量化/反量化，fork 支持 uint8 存储）。原计划的 BF16 KV 需改 3 处 + fp8 数学，暂不做 |
| 模型 dtype | `Bfloat16 ... requires compute capability >= 8.0`（2080Ti=7.5） | `serve.py` 的 `--dtype` 改为**profile 可配**（`DEFAULTS` 新增 `dtype: bfloat16`），本地 profile 设 `dtype: float16`；`capture_runtime.py:311` 的“BF16 activations”断言放宽为 bf16/fp16 |
| profile 字段 | `KernelConfig.linear_backend_per_quant 意外关键字` | 该字段可选，从 profile 移除 |

### 当前阻塞（下一轮从这里开始）
`ValueError: Content capture output must match native projection geometry`
（`adapters/_fused_capture.py:267`，由 `capture.py:80` 的 `torch.ops.kvmem.capture_projection` 触发）

断言内容：`raw_q/raw_k` 的形状必须等于 `(n_tokens, heads*head_dim)`。
**待查**：调用方分配的 `raw_q_output/raw_k_output` 形状/device 与 native 不一致——高度怀疑是
**TP=2 下 KV/Q 头被切分**（fork 暴露每 rank 的头数，而上游按全局头数分配）。
下一步：在 `capture.py:80` 附近打印 `n_tokens / num_q_heads / num_kv_heads / head_dim / raw_* 形状`
与 `attn` 模块的 `num_heads/num_kv_heads`，判定是“头数口径”还是“dtype/连续性问题”。

### 启动命令（可复现）
```bash
cd /home/srj/src/kvmem-vllm-2080ti
/home/srj/vLLM-2080Ti-Definitive/.venv/bin/vllm-kvmem serve \
  --config local/serve-2080ti-smoke.json --run-dir /tmp/kvmem-run-smoke
```
profile：`local/serve-2080ti-smoke.json`（Coder390 FP8-fp8mtp、TP=2、gpu_mem 0.80、max_model_len 16384、MTP 1 draft、dtype float16、kvmem 预算 2048/2048）

## 9. P2（续）：已启动到 warmup 完成，卡在 warmup 后第一个 forward

### 又修掉的两个 SM75 定制点
| 问题 | 根因 | 处理 |
|---|---|---|
| `Content capture output must match native projection geometry` | `adapters/capture.py:126` 里 `install_persistent_capture` 把缓冲 dtype **写死 bfloat16**，而我们跑 fp16 ⇒ 形状对、dtype 不符 | 改为 `dtype=weight.dtype`（跟随模型） |
| profile `dtype` 属未知字段 | `DEFAULTS` 没有该键 | `DEFAULTS` 增加 `'dtype': 'bfloat16'`，本地 profile 设 `float16` |

### 现在能走到的阶段（日志实证）
- `Injected SystemOneWorker`（两个 rank）
- `Available KV cache memory: 1.76 GiB` → `GPU KV cache size: 65,536 tokens`（16K 上下文 ×4 并发）
- `Creating v1 connector with name: KVMemConnector`（两个 rank）
- `FlashInfer resolved query dtypes: prefill=float16, decode=float16, decode_backend=flashinfer-native`
  ⇒ **fork 的 FlashInfer 在 SM75 上确实用 fp16 跑起来了**
- `Warming up Qwen GDN Triton kernels`、`Warmed M-RoPE Triton kernels`、`Warmed Mamba batch_memcpy_kernel` 全部成功

### 当前阻塞
两个 rank 都停在：
```
[kvmem ring] hook entered, enabled=False batch=False
```
之后没有输出，60 秒后 EngineCore 报
`No available shared memory broadcast block found in 60 seconds`（worker 卡住导致）。

`/dev/shm` 有 32G 空闲，**不是 shm 容量问题**；是 warmup 后第一个 forward 卡死。

**下一步（隔离法）**：在 profile 里分别关掉可能的钩子（`rolling decode`/`original rope`/`packed assembly` 等
`kvmem` 子项），或用 `--layerwise-*` 开关组合，确认是 **ring hook**、**capture hook** 还是 **assembly**
导致 hang；同时在 `systemone_worker` 的 forward 入口加一行日志，确认卡在哪个 hook 之内。

## 10. P2 ✅ 完成：上游 kvmem 在双 2080Ti 上成功服务（真实请求）

**之前判断的“卡死”是误判**：warmup（GDN/M-RoPE/Mamba Triton + KV cache 分配 + 连接器初始化）耗时较长，
超过了我设的等待窗口；实际启动约 90–120 秒完成，**health 200**。

### 实测请求（服务日志与 run-dir 均可复核）
| 请求 | prompt_tokens | finish | 答案 | 耗时 |
|---|---|---|---|---|
| 短（“Reply with exactly: OK-2080TI”） | 63 | stop | `OK-2080TI` ✅ | — |
| 中（含一次性码，~6.3K 字符） | 1,048 | stop | `ZX-4417-QJ7` ✅ | — |
| **长（30.9K 字符，超出 select_budget）** | **4,888** | stop | `ZX-4417-QJ7` ✅ | **11.1 s** |

### kvmem 确实在工作（run-dir 证据）
`/tmp/kvmem-run-smoke/capture/`：
`archive.json`（`batches: 131`、`completed_requests: 4`、`block_tokens: 16384`）、
`archive_resources.json`、`cache-geometry.json`（按层的 page_bytes / payload_page_bytes）、
`bounded_refresh.jsonl`、`scheduled_queries.json`、`model_step_latency.jsonl`（`sequence: 129`，
每步延迟记录）、`public_work.jsonl`、`progress.json`。

### 生效配置（两层，见 `launch.json`）
- **kvmem**：`block_tokens=32`、`select_budget=2048`、`gen_budget=2048`、`sink_blocks=1`、
  `recent_blocks=1`、`archive_backend=cpu`、`session_capacity=8`、`request_planning=true`；
  策略 `query=meank / pressure=recency / picker=native`，`path=global`
- **vLLM**：`--dtype float16`、`--kv-cache-dtype fp8_e4m3`、`--tensor-parallel-size 2`、
  `--gpu-memory-utilization 0.80`、`--max-model-len 16384`、`--enforce-eager`、
  `--enable-prefix-caching False`、`--scheduler-cls KVMemScheduler`、
  `--worker-extension-cls SystemOneWorker`、`--kv-transfer-config KVMemConnector`、
  `--speculative-config {mtp, 1}`
- 启动命令：`vllm-kvmem serve --config local/serve-2080ti-smoke.json --run-dir /tmp/kvmem-run-smoke`

### 相对上游的 SM75 改动（共 5 处，全部有注释）
1. `compat_sm75.block_row_count`（block table 形状）
2. `compat_sm75.optional_runner`（capture runner 晚绑定）
3. `core/native_pages.py`：`stored_expansions` / `kernel_expansions` 容忍只有设备行的表
4. `serving/serve.py`：`--dtype` 由 profile 决定（`DEFAULTS` 新增 `dtype`）
5. `adapters/capture.py`：capture 缓冲 dtype 跟随模型；`serving/capture_runtime.py`：
   激活 dtype 接受 fp16

## 11. 删掉引擎侧的实验改动（页环 + 按页滑动双账本）

判断依据：那次提交里的东西**默认都不生效**（\`KVMEM_RING_CLEAN\` 未开，钩子只打印；
双账本只在 \`plan_d\` 下才走），而它带来的只是额外的 import 与打印。

处理：在 fork 里 \`git revert 175950def3\` → \`3b871c663f\`（\`kvmem_ring.py\` 删除，scheduler/model_runner 复位）。
回滚后重启服务验证：
- health 200，日志里 \`kvmem ring\` 行 **0 条**，无异常
- 短请求 \`OK-2080TI\` ✅
- 4,888-token needle \`ZX-4417-QJ7\` ✅，耗时 **9.5s**（回滚前 11.1s，删掉反而更快）

⇒ 引擎侧现在**不需要任何额外补丁**，只要 \`kvmem-0.2.2-clean\` 分支本身。
补丁集随之只剩 \`patches/0002-kvmem-sm75-compat.patch\`。
