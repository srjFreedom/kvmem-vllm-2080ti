# SM75（2080Ti）兼容计划 —— 以最新上游 kvmem-vllm 为准

创建：2026-11-16　仓库：本 checkout（克隆自 `github.com/kvmem/kvmem-vllm`，HEAD `8654700`）
目标运行时：vLLM `0.2.2.post3` SM75 checkout（`$KVMEM_ENGINE_DIR`），torch 2.13.0+cu130，py 3.12.15

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

- `vLLM-2080Ti-Definitive` 分支 `kvmem-0.2.2-clean` 有 **20 个未推送提交**（已备份为 bundle 文件），其中 `a156cf2bfd` 是“kvmem 集成补丁（同步自旧 adapter 仓库）”，P3 的实现可以直接参考/迁移。
- 旧适配器代码已删除，但**已推送到** `srjFreedom/kvmem-vllm-adapter`，保留在 `legacy-adapter-v0.5` 分支。

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
cd "$REPO"
"$KVMEM_ENGINE_DIR/.venv/bin/vllm-kvmem" serve \
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

## 12. 引擎回到**原版 post3**：那 20 个旧补丁全部不需要

**判断依据**：新项目源码对 fork 专有字段的引用数全为 **0**
（`num_logical_tokens` / `kvmem_view_lens` / `kvmem_rope_offset` / `kvmem_true_computed` /
`num_kvmem_offset` / `kvmem_slot_offset`）。那 20 个提交只改 Python（45 文件 +2407/−147，无 C/CUDA），
所以切分支不需要重编译。

**操作**：新建并切到 `sm75-post3-clean` @ `v0.2.2-post3`（**post3 之上 0 个提交**）。
旧的 `kvmem-0.2.2-clean`（post3 + 20）仅作历史参考，运行时不再使用。

**验证（原版引擎，实测）**
| 请求 | prompt_tokens | 答案 | 耗时 |
|---|---|---|---|
| 短 | 63 | `OK-2080TI` ✅ | — |
| 长 needle | 4,888 | `ZX-4417-QJ7` ✅ | **9.5 s** |

⇒ 结论：**kvmem 逻辑完全在插件层**，引擎保持 2080Ti fork 的原样即可（与上游 `vllm>=0.30` 的插件设计一致）。
补丁集只剩 kvmem 侧的 `0002-kvmem-sm75-compat.patch`。

## 13. 会话命中（RAM 缓存）在原版引擎上验证通过：**12.4×**

旧项目那 20 个引擎提交里有 kv-offload（RAM KV 层）与引擎侧账本；上游把「会话复用」做在**插件层**
（`session_capacity` / `session_checkpoint_bytes` / `session_checkpoint_policy` + `http_sessions` 中间件
+ `archive` 的 CPU 后端）。在原版 post3 引擎上实测：

| 请求 | prompt_tokens | 耗时 | 答案 |
|---|---|---|---|
| turn1（4,848 token，冷） | 4,848 | **7.40 s** | `ZX-4417-QJ7` ✅ |
| turn2（同 session 续写，4,880 token） | 4,880 | **0.60 s** | `ZX-4417-QJ7` ✅ |

**提速 12.4×**。运行目录证据（`capture/scheduler_resources.json` / `archive.json`）：
```
sessions     = {entries: 4, hits: 2, misses: 7, waits: 11}
session_views: ns ['', 'sess-time-1']  history 4889  window 2073  prefix_sha256 ...
session_checkpoints   = 4
session_restored_tokens = 4857      ← turn2 只 prefill 了 4880-4857 = 23 个 token
```

### 复现配方（三个要点，缺一不成）
1. **同一 `session_id`**（可选 `cache_salt`）；不同会话必须换 id。
2. **请求要带完整累计历史**：turn2 = turn1 的 prompt + `assistant` 的**精确输出** + 新回合。
   手工改写的 assistant 文本会因重新分词导致前缀不匹配（实测：漏掉开头两个换行 ⇒ 仍 miss）。
3. 短回答要 `chat_template_kwargs.enable_thinking=false`，否则 `max_tokens` 被思考吃光、`content` 为空。

会话结束用上游的 close 协议（`kv_transfer_params.kvmem_release_session`，1 token 输入/输出）。

## 14. 前缀命中零前端改动：auto-session + P 边界冻结（**8.7×**）

**背景**：上游有两条命中路径 —— 会话命中（按 `session_id` 命名空间 + 精确 token 前缀）和共享前缀
（`prefix image`，键是 `(model, cache_salt, sha256(tokens[:P]))`，与 session 无关，但**明确拒绝 TP>1**）。
用户要的是"前缀 hash 命中、前端不用改"，在 TP=2 下只能走会话机制。

**实现（纯插件层，3 处）**
1. `http_sessions.auto_session_id()`：请求没带 `session_id`/`X-Session-ID`/`vllm_xargs.session_id` 时，
   用**提示前缀文本的 sha256**派生 `auto-<hash24>` 作为命名空间。（文本相同 ⇒ token 相同 ⇒ 真命中；
   文本不同 ⇒ 不同命名空间 ⇒ 不会误命中。`kvmem_release_session` / `kvmem_share_prefix` 不派生。）
2. `serve.py`：`DEFAULTS` 新增 `auto_session` / `auto_session_chars`，向启动环境导出
   `KVMEM_AUTO_SESSION=1`、`KVMEM_AUTO_SESSION_PREFIX_CHARS`。
3. profile 打开 **P 边界冻结**（否则 checkpoint 在"prompt+输出"末端，重复同一 prompt 不算续写）：
   `session_checkpoint_policy=sealed`、`session_freeze_prompt=true`、
   `session_capture_terminal=false`、`session_capture_decode=false`、`query_history=false`。

**实测（完全不带 session_id 的 3 次相同请求）**
| 次 | 耗时 | 答案 |
|---|---|---|
| 1 | 7.66 s | `ZX-4417-QJ7` |
| 2 | **0.89 s** | `ZX-4417-QJ7` |
| 3 | **0.89 s** | `ZX-4417-QJ7` |

⇒ **8.7×**；`scheduler_resources.json → sessions.hits=2`，`archive.json → session_restored_tokens=9216`。

**边界**：派生命名空间会随不同前缀增长，需配 `session_capacity`（当前 8）与
`KVMEM_SESSION_IDLE_LRU_EVICTION`；同一前缀的并发请求会被串行化（上游"同会话串行"语义）。

## 15. auto-id 修正：命名空间 = 会话开头（完整 hash），多轮续写稳定命中且任务隔离

**上一版的缺陷（用户指出）**：用"截断前缀（8192 字符）"做 hash ⇒ 同一 agent 的 system prompt+工具定义
相同，所有任务落进**同一命名空间**；而上游账本"每个会话只保留一个 prompt checkpoint"、
"同会话请求串行化" ⇒ 不同任务互相覆盖/串行。实测证据：第二个任务把第一个会话挤掉（`entries: 1`）。

**修正**：命名空间 = `sha256(system 文本 + **第一条 user 消息**（完整、不截断）)`
- 会话内部：每一轮的开头都不变 ⇒ 命名空间稳定 ⇒ 续写永远命中
- 不同任务：第一条 user 消息不同 ⇒ 命名空间不同 ⇒ 不撞车
- 不再需要"每次续写后改 id"：下一轮的历史在本次应答结束时已确定，而前缀匹配由账本自身完成

**实测（全程不带 `session_id`）**
| 请求 | 耗时 | 说明 |
|---|---|---|
| A 第1轮 | 7.66 s | 冷 |
| A 第2轮（续写） | **1.08 s** | 命中 |
| B 第1轮（不同任务，同样 system 前缀） | 7.17 s | **冷**，答案 `200` ✅（另一问题） |
| A 第3轮（B 之后） | **1.08 s** | **仍命中**（A 未被 B 挤掉） |

`restored_tokens=9216`（两次恢复），`hits=2`。

## 16. 复用容量由两个参数共同决定（用户提问的确认）

| 参数 | 含义 | 我们的值 |
|---|---|---|
| `session_capacity` | 保留多少个可复用会话（**条目数**） | 8 |
| `session_checkpoint_bytes` | **单个** checkpoint 的 payload 字节上限 | 512 MiB |

上游原文（`docs/prompt-sessions.md`）：
> Entry count and retained checkpoint payload bytes are bounded.
> Shared archive capacity includes an additional `session_capacity` authority allowance.
> The checkpoint byte limit … is not a whole-process RAM quota.

⇒ 复用总开销 ≈ `session_capacity × session_checkpoint_bytes` + archive authority 余量；
另有 `cpu_read_cache_bytes`（CPU 读缓存总预算）、`repository_blocks`（历史页数）两个上限。
两个参数必须**同时**非零（`options.py:1011` 会拒绝只设一个）。

## 17. 大上下文实测（原版引擎，双 2080Ti）

**配置调整**（都记在 `local/serve-2080ti-smoke.json`）
| 项 | 值 | 原因 |
|---|---|---|
| `max_model_len` | 131,072 | 262,144 被 vLLM 拒绝：需要 4.42 GiB KV，当时只有 1.74 GiB |
| `gpu_memory_utilization` | 0.88 | 换到 **GPU KV pool = 198,817 tokens**（131K 请求 1.52 并发） |
| `kvmem.repository_blocks` | 16,384 | 1024 装不下长 prompt，实测报 `BufferError: Authority capacity exhausted` |
| `kvmem.cpu_read_cache_bytes` | 4 GiB | 长历史在 CPU archive 的读缓存 |

**请求结果（同一服务实例，单请求）**
| prompt_tokens | 耗时 | finish | 答案 |
|---|---|---|---|
| 4,888 | 9.5 s | stop | `ZX-4417-QJ7` ✅ |
| 65,496 | 2 m 06 s | stop | `ZX-4417-QJ7` ✅ |
| 86,532（冷） | **2 m 49 s** | stop | `ZX-4417-QJ7` ✅ |
| **86,532（同前缀，重复）** | **0.85 s** | stop | `ZX-4417-QJ7` ✅ |

⇒ 大上下文前缀命中 **约 200×**；证据 `sessions.hits=1`、`session_restored_tokens=86,528`。

**边界（诚实记录）**
- 256K（262,144）在本机不可达：vLLM 要求 KV ≥ 4.42 GiB，而 2×22G 在装下 FP8 27B 权重（约 32G）后
  最多给到约 3.5 GiB ⇒ 目前上限 131,072（实际跑通 86.5K）。
- 本轮所有大 prompt 都是「prompt >> 工作窗口（select 2048 + gen 2048）」：65K/86K 的 KV 大部分在 CPU
  archive，正是 KVMem 的目标场景。

## 18. CTX 与物理池解耦：`max_model_len` 不再受池容量约束（**超池 218K 命中 1.36s**）

**用户口径**：物理池是 `B+R`（工作窗口），`max_model_len`（= 逻辑上下文上限 CTX）与它无关，
所以启动时不该做"max_model_len ≤ 池容量"的校验，也不该和模型最大长度绑定。

**实现（两处）**
1. **引擎**（`vllm/v1/core/kv_cache_utils.py`，提交 `01506a5471`）：`KVMEM_SKIP_KV_CHECK=1` 时跳过
   `_check_enough_kv_cache_memory`，并打印
   `KVMem: skipping the KV capacity check (max_model_len=… needs … GiB, pool has … GiB)`。
2. **kvmem**（`serve.py`）：`DEFAULTS` 新增 `skip_kv_check`，导出环境变量；profile 设 `true`。

**实测（CTX=262,144，池 205,752 token，kvmem 窗口仍 2,048+2,048）**
| 请求 | prompt_tokens | 耗时 | 答案 |
|---|---|---|---|
| 4,888 | 4,888 | 9 s 级 | `ZX-4417-QJ7` ✅ |
| **超过池** | **218,220** | **8 m 18 s** | `ZX-4417-QJ7` ✅ |
| **同上，重复（前缀命中）** | 218,220 | **1.36 s** | `ZX-4417-QJ7` ✅ |

⇒ 这是 KVMem 的目标形态：**prompt 比物理池大 6%，靠归档+选择跑通；第二次请求 1.36 s（约 365×）**。
证据：`sessions.hits=1`、`session_restored_tokens=218,112`。

**注意**：跳过校验只去掉"启动拒绝"，不改变显存事实——池仍是 205,752 token，工作窗口仍是 4,096 token；
超长 prompt 的 KV 存在 CPU archive（`repository_blocks=16,384` × 32 = 524,288 token 上限）。

## 19. gpu_memory_utilization → KV 池实测曲线（回答"为什么这套看起来低这么多"）

**实测（同一实例、fp8_e4m3 KV、TP=2、CTX=262,144、kvmem 窗口 2,048+2,048）**
| gpu_mem | 可用 KV | GPU KV 池 | 说明 |
|---|---|---|---|
| 0.80（上游 starter）| 1.74 GiB | 65,536 token | 只够 16K×4 并发 |
| 0.88 | 3.46 GiB | 205,752 token | 本轮早期用的值 |
| **0.96（现在）** | **5.2 GiB** | **307,866 token** | 262,144 请求 1.17 并发，**已能原生容纳 CTX** |
| 0.972（旧项目生产值）| — | ≈201,438 token（旧记录）| 旧档 |

**结论：不是这套低，而是两个原因**
1. **显存比例**：上游 starter 用 `0.80`、我们早期用 `0.88`，旧生产是 `0.972`。池大小几乎线性于 gpu_mem
   （实测 ≈ 5.9 万 token/GiB）。0.96 已比旧档 0.972 的池更大（307,866 vs 201,438）。
2. **vLLM 的启动校验**曾把 `max_model_len` 钉在池容量上（262,144 需 4.42 GiB，0.88 时只有 3.46 GiB），
   于是看起来"上下文上限低"。现已用 `KVMEM_SKIP_KV_CHECK` 解耦：`max_model_len` 只表达逻辑 CTX。

**注意**：0.96 时物理显存 21,794 / 22,528 MiB，已经贴近上限；再往上没有空间，
要更长上下文只能靠"逻辑 CTX + 归档"，而不是更大的池。

## 20. 多会话保留命中（= ninfer 的 `RetainedSessions`）

**对应关系**：ninfer 的 `RetainedSessions` ⇔ 上游的 `session_capacity`（我们现在 = 8）。
每个条目还受 `session_checkpoint_bytes` 限制（现在 512 MiB）。

**开发用小池 profile**（按用户要求：池贴近 B+R，省时间）
| 项 | 值 |
|---|---|
| `select_budget` + `gen_budget` | 4,096 + 2,048 = **6,144**（B+R）|
| `sink_blocks` / `recent_blocks` | 32 / 64 → 1,024 / 2,048 token |
| `gpu_memory_utilization` | **0.74** ⇒ GPU KV 池 **21,845 token**（≈3.5× 窗口）|
| `max_model_len`（CTX） | **32,768**（= 5.3×B+R，>2× 要求）|
| `session_capacity` | 8 |

**实测 AⓘB 会话交替（auto-id，无前端 session）**

*情形①：prompt 2,212 token（< 窗口）*
| 请求 | 耗时 | 答案 |
|---|---|---|
| A1 冷 | 2.69 s | `ZX-4417-QJ7` ✅ |
| B1 冷 | 2.50 s | `QQ-8882-ZZ9` ✅ |
| **A2（切回 A）** | **0.64 s** | `ZX-4417-QJ7` ✅ |
| **B2（切回 B）** | **0.59 s** | `QQ-8882-ZZ9` ✅ |
| **A3（再切回 A）** | **0.63 s** | `ZX-4417-QJ7` ✅ |

`sessions = {entries: 2, hits: 3, misses: 2}`、`restored_tokens = 6,144`
⇒ **两个会话同时保留在 RAM，来回切换不重新 prefill**（≈4×）。

*情形②：prompt 7,254 token（> 窗口，需要装配/压缩）*
| 请求 | 耗时 | 命中 |
|---|---|---|
| A1 / B1 / A2 / B2 / A3 | 全部 **10.3 s**（≈全量 prefill）| `hits` 不增长 |

⇒ **超过窗口的会话目前不会被恢复**（`misses` 增长、`checkpoint_deferrals=0`，
说明 checkpoint 已捕获，是 **lookup/import 侧拒绝**）。已在该路径装探针
（`session_control.lookup` 的三处 miss 分支，日志 `/tmp/kvmem_session_miss.jsonl`，
代码码：`entry_busy` / `plan_changed` / `capacity_hold`），下一轮跑一次 7K 的 A/B 即可定位。

## 21. 池只需要装 B+R（修正一处会计口径），sink/recent 调成 1,024/1,024

**用户问题：为什么 B+R 是 4,096 + 2,048？**
老实说：**那是我随手挑的**，没有推导依据（目标只是"窗口小、调试快、2K 的 prompt 能整段落在窗口里"）。
真正的约束只有三条（`core/config.py`）：`select_budget`/`gen_budget` 能被 `block_tokens` 整除、
`sink+recent ≤ select_budget/block_tokens`、`repository_blocks × block_tokens ≥ select+gen`。

**本轮改动**
| 项 | 值 | 说明 |
|---|---|---|
| `sink_blocks` / `recent_blocks` | **32 / 32**（= **1,024 / 1,024 token**）| 按你的要求 |
| `gpu_memory_utilization` | 0.74 | 池 **21,845 token**（窗口 6,144 的 3.5 倍）|
| CTX | 32,768 | > 2×B+R |

**关键发现：引擎的原生 KV 页是 32,768 token，不是 32**
`cache-geometry.json`：4 个 cache group（3 个 attention 各 16 层、1 个 GDN 17 层），
attention 组的 `block_tokens` 等于 **32,768**（CTX 越小它越小，CTX=262,144 时就是 262,144！）。
但 KVMem 用 `guided_block_tokens = 32` 指导分配。`capacity.py` 的 `_demand()` 原来按**原生大页**算，
于是"一个最坏请求"被算成每 group 一整页 ⇒ 池必须 ≥ 组数 × CTX，池永远压不下来。

**修正**：`_demand()` 改为按 `guided_block_tokens` 计费（没有则退回 `manager.block_size`），
并由 launcher 导出 `KVMEM_WINDOW_TOKENS = select_budget + gen_budget`。
→ 最坏驻留请求 = **B+R**，与 CTX 无关。

**验证**
- 池 21,845 token（0.74）能启动，CTX 32,768 照常（skip-check 仍在用）
- 小 prompt（2,212 token）多会话：A1 2.68s / B1 2.49s → **A2 0.64s / B2 0.59s / A3 0.63s** ✅
- 大 prompt（7,254 token，>窗口）：C1/D1 10.3s（冷）→ **C2/D2/C3 仍 10.3s（未命中）**

## 22. 仍未解决：超过窗口的会话不被恢复

探针（`entry_busy` / `plan_changed` / `capacity_hold`）在这轮**没有触发**，
说明 miss 发生在更早的分支：`uncached_sessions` / `entry.phase != 'idle'` / `namespace is None`。
下一轮把探针补到这些分支（`session_control.lookup` 前 3 个 return None），一次 7K 的 A/B 即可定位。

## 23. 服务看门狗（不再硬等）：`scripts/serve_watch.py`

之前每次实验都用 `sleep 200` 等启动，实际上 **启动只要 54.6 秒**——白等了 2.5 分钟/次。
新增脚本用**可观测信号**代替固定等待：

\`\`\`bash
serve_watch.py start   [--config ...] [--run-dir ...] [--timeout 420] [--fresh]
serve_watch.py stop
serve_watch.py status
serve_watch.py req --file REQ.json [--idle-stall 20] [--timeout 3600]
\`\`\`

| 能力 | 实现 |
|---|---|
| 启动就绪 | 每 0.5s 轮询 `/v1/models`（不是固定 sleep）；同时监控进程退出与日志里的致命行（`ValueError/RuntimeError/EngineDeadError/OOM/Traceback`），失败立即返回 |
| 就绪信息 | 打印 `READY in Xs`、model id、CTX、池大小、以及 skip-check 那行 |
| 请求完成 | 后台线程发请求，主线程按 0.4s 采样 `nvidia-smi`（util + mem）|
| 卡死判定 | 请求在飞但 GPU util ≤5% 持续 `--idle-stall` 秒 ⇒ 打印 `STALL`（不阻塞等待）|
| 结束统计 | wall 时间、prompt/completion tokens、吞吐、GPU 忙的比例、峰值显存、答案 |
| run-dir 安全 | 只清理"明显是我们生成的"目录（空或只含 `capture/`、`launch.json`），否则要求 `--fresh` |

**实测（当前 dev 配置：池 21,845 / 窗口 6,144 / CTX 32,768）**
| 指标 | 值 |
|---|---|
| 启动到 READY | **54.6 s** |
| 4,888-token 冷请求 | **7.87 s**（≈621 tok/s prefill+decode），GPU 忙 17/19 采样，无 STALL |
| 同一请求重复（会话命中） | **2.45 s**（`restored_tokens=4,864`）|
| 2,212-token 会话切回 | 0.59–0.64 s |

**启动时间构成**（后续优化方向）：模型加载 + FlashInfer/GDN/M-RoPE/Mamba 内核预热 + KV 池分配。

## 24. 提速评估：图模式无效、MTP 有限、prefill 是硬件上限（附裸 vLLM 基线）

**① 关掉 eager（`execution_mode=full_graph`）**
- 能启动（`--compilation-config {"mode":0,"cudagraph_mode":"FULL"}`，`READY in 54.6s`）
- **但没有提速**：4,888-token 请求 full_graph **7.86s** vs eager **7.87s**；600-token 生成
  full_graph **24.0 tok/s** vs eager **23.7 tok/s**（差 1%）
- 原因：kvmem 每步改写 KV 块表/窗口，图基本按 eager 走；日志也没有 "Capturing cudagraph" 的实质捕获记录

**② MTP 草稿数（`draft_tokens` 1 → 4）**
- eager+MTP4：600 token **26.4 tok/s**（MTP1 是 23.7）⇒ 仅 +11%
- 日志给出根因：`Fused multi-step draft decode is not supported by attention backend(s) FLASHINFER;
  falling back to rebuilding attention metadata`，且 `Mean acceptance length: 2.07–2.32 / 4`
  ⇒ 草稿接受率约 50–58%，且每步要重建 metadata，收益被吃掉

**③ prefill chunk（128 → 1024）**：4,888 token 冷跑 **622 → 627 tok/s**，**基本无变化**
（`prefill_chunk_tokens` 必须 32 对齐且 < `gen_budget`；改成 2048 会要求 `gen_budget ≥ 8192`）

**④ 裸 vLLM 基线（不带 kvmem，同模型/TP/精度/eager）**
| 负载 | 裸 vLLM | kvmem | 差 |
|---|---|---|---|
| 4,888-token prefill | **653 tok/s** | 627 tok/s | kvmem 开销 ≈ **4%** |
| 600-token 生成 | **16.9 tok/s** | 23.7–26.4 tok/s | kvmem **反而更快**（窗口 4K，注意力更便宜）|

⇒ **结论：慢在硬件/内核上限**（27B FP8、2×2080Ti、SM75、eager ≈ 650 tok/s prefill、17–26 tok/s decode），
不是 KVMem 的开销。旧项目记录里的 46/85 tok/s 是它那套 fork 补丁（MTP4+compiled+自有图集成）的成绩，
本机要接近它，得单独做"SM75 上的 MTP 融合 draft + 有效图捕获"，属于独立性能项目。

**当前 dev 定稿配置**：`select=4,096`、`gen=2,048`（窗口 6,144）、`sink/recent=32/32`（1,024/1,024 token）、
`chunk=1,024`、`draft=1`、`eager`、`gpu_mem=0.78`（池 61,895）、`CTX=32,768`。
启动经看门狗实测 **54.1s**。

## 25. 纠正吞吐口径 + 提速清扫结果（compiled 是真正的杠杆）

**先纠正我上一条的错误**：我之前用 `prompt_tokens / 整个 wall` 当 prefill 速率，把 decode 也算进了分母，
所以报出 622 tok/s。正确做法：`max_tokens=1` 且用**全新（未命中）**的 prompt 测。重测后：

| 配置 | 4.1K 冷 prefill | 600-token decode |
|---|---|---|
| kvmem，eager，torch_reference | **1,021 tok/s** | 23.7 tok/s |
| kvmem，eager，qw3_cuda（CUDA 内核） | 1,024 tok/s | 24.0 tok/s |
| kvmem，**compiled_graph** | **1,063 tok/s** | **40–42 tok/s** |
| 裸 vLLM（无 kvmem），eager | 1,226 tok/s | 16.9 tok/s |
| 裸 vLLM（无 kvmem），**compiled** | **1,312 tok/s** | — |

**结论**
1. **compile 是关键杠杆，而且主要加速 decode**：kvmem decode 23.7 → **40–42 tok/s（+70%）**；
   prefill 只 1,021 → 1,063（+4%）。裸 vLLM 上同样是 prefill 1,226 → 1,312（+7%）。
2. 裸 vLLM compiled 的 **1,312 tok/s** 正好落在用户说的 1300–1600 区间 ⇒ 硬件没问题。
3. kvmem 相对裸引擎在 prefill 上仍有 **约 19% 开销**（1,063 vs 1,312）：不在 `prefill_chunk_tokens`
   （128→1024 无变化）、也不在 `mean_k_backend`（torch_reference vs qw3_cuda 无差别），
   而在 prefill 期间的 KVMem 侧工作（bounded-target 刷新/装配/统计），需要 profile 逐步耗时才能定位。
4. MTP 目前收益有限（`draft_tokens` 4 只 +11%），根因是 `FLASHINFER` 不支持融合多步 draft、
   每步重建 attention metadata、接受率仅 2.07–2.32/4。

**当前 dev 定稿**：`execution_mode=compiled_graph`、`gpu_mem=0.84`（池 104,372）、窗口 4,096+2,048、
`sink/recent=32/32`（1,024/1,024 token）、chunk 1,024、draft 1、CTX 32,768、`mean_k_backend=torch_reference`。
compiled 首次启动 290s（编译），命中缓存后 **57s**。

## 26. decode 提速：先找 kvmem 侧解法（无），再从 fork 找到 SM75 全图配方

**kvmem-vllm 侧没有专门解法**，但 fork 里已经埋了 SM75 的 spec-decode 全图分支
（`flashinfer.py::_sm75_spec_prefill_graph_query_len`）。之前它没生效，是因为我们没给
`cudagraph_capture_sizes`，于是 `get_cudagraph_support()` 返回 `UNIFORM_SINGLE_TOKEN_DECODE`
（只支持 query_len==1 的图），vLLM 把 `CUDAGraphMode.FULL` 降级为 `FULL_DECODE_ONLY`，
`enable_cuda_graph=False`，FlashInfer 的融合多步 draft 也就永远关闭。

**配方（全部条件都必须满足，来自源码）**
| 条件 | 我们的值 |
|---|---|
| SM75（CC 7.5） | ✅ 2080Ti |
| `dtype == torch.float16` | ✅ |
| `decode_context_parallel_size == 1` | ✅ |
| `use_non_causal` 关 | ✅ |
| `VLLM_BATCH_INVARIANT` 未设 | ✅ |
| `1 ≤ draft_tokens ≤ 7` | 4 |
| `cudagraph_mode.decode_mode() == FULL` | `execution_mode=full_graph`（`{'mode':0,'cudagraph_mode':'FULL'}`）|
| **`cudagraph_capture_sizes == {query_len × batch}`（batch=1..max_num_seqs）** | `max_num_seqs=1`、draft 4 ⇒ query_len 5 ⇒ **`[5]`** |
| `max_cudagraph_capture_size == query_len × max_num_seqs` | 5 |

生效证据（日志）：出现 `SM75 speculative FA2 prefill uses stable metadata buffers for FULL CUDA Graph replay`，
且 `CUDAGraphMode.FULL is not supported…`（降级）与 `Fused multi-step draft decode is not supported…`
（退回重建 metadata）两条告警**都为 0**。

**decode 进展（600 token 生成，同一台机器）**
| 配置 | decode | 备注 |
|---|---|---|
| eager，MTP1 | 23.7 tok/s | 基线 |
| compiled_graph，MTP1 | 42.0 tok/s | **最大单项提升**（图捕获）|
| compiled_graph，MTP4 | 43.1 tok/s | 融合 draft 未生效（仍重建 metadata）|
| **full_graph，MTP4，capture_sizes=[5]** | **44.6 tok/s** | 全图 + 融合 draft 都生效，**比 eager +88%** |
| （参照）旧项目 compiled+MTP4 | ~85 tok/s | 目标 |

**仍差 2× 的原因（下一步）**：实测每步约 50 ms，而 27B FP8 权重分到每卡 ~16 GB，
按 ~600 GB/s 带宽理论下限约 26 ms/step ⇒ 我们只到理论上限的 ~50%。
接受长度也只有 2.03–2.33 / 5（约 45%）。候选方向：
①fp8 KV 在 SM75 上的软件反量化开销；②TP=2 每步 all-reduce；③MTP draft 前向的额外步。
prefill 在这套配置下是 971 tok/s（比 compiled 的 1,063 略低，因为捕获了 5 宽 batch）。

## 27. decode 50ms/step 的逐 kernel 分解（torch profiler，无 ncu 权限）

**方法**：profile 里加 `profiler_config={'profiler':'torch','torch_profiler_dir':...}`（launcher 已支持），
启动后 `POST /start_profile` → 发请求 → `POST /stop_profile`，worker 会落
`rank0/rank1.*.pt.trace.json.gz`（注意要等文件大小稳定，否则 gzip 截断）。

**64 token 请求，rank0 GPU 时间 1,679 ms**
| 占比 | kernel | 判读 |
|---|---|---|
| **51.9%** | `marlin::Marlin`（FP8 权重 GEMM）| 权重带宽下限，≈30 ms/step；`w8a16`，MarlinFP8ScaledMM |
| **18.2%** | `gemvx::kernel<int,int,__half,…>` | MTP draft 头 GEMV，≈10 ms/step |
| 5.0% | `cutlass_75_wmma_tensorop_f16` | 小矩阵（部分层回退）|
| 3.3% | `direct_copy_kernel`（fp16 拷贝）| KV 装配/拷贝 |
| 2.0% | `cross_device_reduce_1stage` | **TP all-reduce 只占 2%** |
| 2.0% | `fused_sigmoid_gating_delta_rule_update` | GDN 状态更新 |
| 1.9% | `reduce_kernel MeanOps` | kvmem 打分的 mean-K |
| ~7% | 各种 elementwise/copy/rsqrt/pow | 归一化与逐元素 |

**两个假设的判决实验（裸 vLLM，full_graph + MTP4 + capture[5]，同一切）**
| 变量 | decode |
|---|---|
| `kv-cache-dtype float16` | 47.8 tok/s |
| `kv-cache-dtype fp8_e4m3` | **48.4 tok/s** |

⇒ ①**fp8 反量化没有可测成本**（48.4 ≈ 47.8）；②**TP all-reduce 不是瓶颈**（2%）；
③kvmem 相对裸引擎只慢 **8%**（44.6 vs 48.4）。

**MTP 草稿数扫描（kvmem，full_graph，capture=[K+1]）**
| K | decode | 接受长度 |
|---|---|---|
| 1 | 42.0（compiled）| — |
| 2 | **46.4** | 2.05–2.10 / 3（68%）|
| 4 | 44.6 | 2.45 / 5（49%）|

⇒ 定稿 `draft_tokens=2`、`cudagraph_capture_sizes=[3]`。

**结论**：本机 decode 的现实上限 ≈ **48 tok/s**（裸引擎），主要由 **Marlin w8a16 GEMM（52%）+ MTP draft GEMV（18%）**
构成，两者都是权重带宽问题；旧项目的 85 tok/s 对应 ≈26 ms/step（等于满带宽），
本配置只到 ≈51 ms/step。要继续追，需要动 fork 的 Marlin/SM75 kernel 选择（`linear_backend_per_quant`
只覆盖 `fp8_w8a8`/`nvfp4_w4a4`，我们的 `w8a16` 没有可选后端），属于独立内核优化项目。

## 28. 性能对齐 vllm-2080ti：把 fork 的 fast 档开关逐个验证（结论：已全部接上，天花板 ≈50）

策略定为「**机制用 kvmem-vllm，性能用 vllm-2080ti**」。旧项目的 85 tok/s 配置来自 fork 的
`launcher.sh --mode fast` + `VLLM_TURBOQUANT_*` 体系，逐个验证结果：

| fork 的性能开关 | 作用 | 接入后 decode |
|---|---|---|
| `execution_mode=full_graph` + `cudagraph_capture_sizes=[K+1]` | SM75 全图 + 融合多步 draft | **42.0 → 46.4**（最大单项） |
| `VLLM_SM75_SPEC_SYNC_MODE=nosync`（fast/aggressive 档默认）| 去掉投机图内额外流同步 | 无变化（46.4）|
| `VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1`（fast 档默认）| GDN/Mamba 保留 FULL 图 | 无变化；本配置 `min_cg_support` 已是 UNIFORM_BATCH，未触发该分支 |
| `draft_tokens` 2 vs 4 | 草稿数 | 46.4（接受 2.05–2.10/3）vs 44.6（2.45/5）|
| `--kv-cache-dtype turboquant_k8v4` + `VLLM_TURBOQUANT_*` 全家桶（**裸 vLLM**）| TQK8V4 KV 路线 | 50.7 tok/s（对照 fp8 48.4，仅 +5%）|
| `--kv-cache-dtype float16`（裸 vLLM）| 排除反量化成本 | 47.8 tok/s |

**结论：fork 的 fast 档性能开关已全部接上，本机 decode 上限 ≈ 47–51 tok/s（裸引擎 50.7，kvmem 46.4）。**
85 tok/s 需要的每步 ~26 ms 与实测 ~51 ms（Marlin w8a16 GEMM 30 ms + MTP draft GEMV 10 ms + 拷贝/逐元素）
之间是**内核效率**差距，不是配置开关差距；且 kvmem 只接受 `fp8_e4m3/float16/bfloat16`，
`turboquant_k8v4` 这条 fork 最快的 KV 路线与 KVMem 的 KV 格式约束不兼容（除非 kvmem 支持该格式）。

**当前生效（launcher 已默认导出）**：`VLLM_SM75_SPEC_SYNC_MODE=nosync`、
`VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1`、`execution_mode=full_graph`、`draft_tokens=2`、
`cudagraph_capture_sizes=[3]`。

## 29. 原版 vLLM-2080Ti vs 我们的补丁版本：速度 + 逐 kernel 对照

同一台机器、同一 checkpoint、**完全相同的引擎参数**（TP2 / fp16 / fp8_e4m3 KV / max_model_len 32768 /
max_num_seqs 1 / max_num_batched_tokens 2051 / no-prefix-caching / MTP2 / full_graph / capture [3] /
`VLLM_SM75_SPEC_SYNC_MODE=nosync` / `VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1`）。
唯一区别：我们版本多挂 kvmem 插件（`--scheduler-cls KVMemScheduler` + KVMemConnector）。

**速度**
| | 冷 prefill 4.1K | 600-token decode |
|---|---|---|
| 原版 vLLM-2080Ti | **1,174 tok/s** | **47.3 tok/s** |
| 我们（kvmem） | 1,021–1,063 tok/s | 46.4 tok/s |

**逐 kernel（同样 64 token 请求，rank0 GPU 总时间）**
| kernel | 原版 | kvmem | 差 |
|---|---|---|---|
| `marlin::Marlin`（主体 FP8 GEMM）| 787.2 ms (54.9%) × 8,517 | 871.7 ms (52.0%) × 9,399 | **+84 ms / +882 次** |
| **`gemvx`（lm_head，fp16 GEMV）** | **145.5 ms (10.1%) × 67** | **306.5 ms (18.3%) × 141** | **+161 ms / +74 次（2.1×）** |
| `cross_device_reduce_1stage`（all-reduce）| 96.6 ms (6.7%) × 4,455 | 34.3 ms (2.0%) × 4,935 | **−62 ms** |
| `cutlass_75_wmma`（fp16 小 GEMM）| 78.8 ms (5.5%) | 84.5 ms (5.0%) | +6 ms |
| `direct_copy` | 48.4 ms (3.4%) | 54.8 ms (3.3%) | +6 ms |
| `reduce_kernel MeanOps`（kvmem 打分）| 25.0 ms (1.7%) | 31.4 ms (1.9%) | +6 ms |
| **GPU 总时间** | **1,434 ms** | **1,676 ms** | **+242 ms（+17%）** |

**结论（可执行）**
1. 原版每 token 只算 **1.05 次 lm_head**（67 次 / 64 token）；我们算 **2.2 次**（141 / 64）
   ⇒ kvmem 让 lm_head 的调用次数翻倍，仅此一项就多 **161 ms / 64 token ≈ 2.5 ms/token**。
2. 主体 Marlin 多 882 次调用（+84 ms），来自 kvmem 的窗口内额外前向（选片/刷新）。
3. all-reduce 我们反而更少（帧更小）。
4. 因此 kvmem 的 decode 开销 ≈ 总 +17% GPU 时间，其中 **lm_head 重复计算占 2/3**。

**下一步（针对性）**：查 kvmem 为什么每 token 算 2.2 次 lm_head——
最可能是 `KVMEM_BOUNDED_MTP=1` / `request_planning` 在每步为 draft 单独取一次 logits。
若能把 draft 的 logits 复用 verify 的结果，decode 可望回到原版的 47+ 并进一步下探。

## 30. decode 达标：MTP4 + 代码类生成 → 80+ tok/s（kvmem 与原版持平）

之前我用**散文类 prompt** 测（"解释 bounded KV window…"），接受率只有 2.1–2.3/5，
所以怎么调都是 45–48 tok/s。换成**代码生成**后接受率升到 3.8–4.4/5，速度直接翻倍。

**同 prompt（写一个带 pytest 的 bounded KV cache 模块，max_tokens 900）**
| 配置 | decode | 接受长度 |
|---|---|---|
| 原版 vLLM-2080Ti，MTP2 | 66.4 / 66.0 tok/s | 2.79–2.87 / 3 |
| **kvmem，MTP2** | 63.5 tok/s | 2.78–2.88 / 3 |
| **原版 vLLM-2080Ti，MTP4** | **80.8 / 87.2 tok/s** | 3.81–4.19 / 5 |
| **kvmem，MTP4** | **79.6 / 86.5 tok/s** | 3.82–4.37 / 5 |

**对照散文类 prompt**
| 配置 | decode | 接受长度 |
|---|---|---|
| 原版 MTP2 | 47.3 | 2.05–2.10 / 3 |
| 原版 MTP4 | 45.1 / 43.7 | 2.11–2.27 / 5 |
| kvmem MTP2 | 46.4 | 2.05–2.10 / 3 |
| kvmem MTP4 | 44.6 | 2.45 / 5 |

**结论**
1. **性能达标**：代码类生成 **80–87 tok/s**（kvmem 与原版差 1–2%，在噪声内）。
2. 决定速度的是**接受率**，而接受率由 workload 决定：代码 ≫ 散文（3.8–4.4 vs 2.1–2.3）。
   MTP4 只有在高接受率 workload 上才划算；散文类 MTP4 反而略低于 MTP2。
3. 因此 dev/prod 默认改为 **`draft_tokens=4` + `cudagraph_capture_sizes=[5]` + `max_num_batched_tokens=4096`**。
4. 之前算的"FP8 理论上限 ~80"是对的：80–87 正好贴着 Marlin 权重带宽下限，说明已经压干净。

## 31. 修复：超过窗口的会话续写不命中 —— 根因是会话模式选错

**诊断过程**：在 `session_control.lookup` 的 10 个 miss 分支全部插探针（`_session_miss_probe`，
写 `/tmp/kvmem_session_miss.jsonl`，码 `miss01..miss10`），一次 7.2K 的 A1/A2 就把原因钉死：

```
{"code":"miss04","prompt":7280,"history":7168,"window":4352,"phase":"idle","prompt_history":0}
```

`miss04` = `needs_p` 为真但没有可用 P checkpoint 的分支。而 `prompt_history` 来自
`prompt_checkpoints._capture_install()` 里保留到 `pool.prompts[namespace]` 的旧快照，
那条保留只在 `preserve` 为真时发生：

```python
preserve = self.retain_prompt and terminal and previous is not None and previous.kind == 'P'
retain_prompt = (session_checkpoint_policy == 'pm')     # session_worker.py:111
```

我们原来是 `sealed` + `freeze_prompt`（frozen-P 会话）⇒ `retain_prompt=False` ⇒ `prompts[]` 恒空
⇒ `prompt_history` 恒 0 ⇒ **凡是需要 P 重放的续写（即真正被压缩过的会话）必然 miss**。
`sealed/frozen-P` 模式本来就只支持"同前缀的独立提问"，不支持多轮续写。

**修复（纯配置，无代码改动）**
```json
"session_checkpoint_policy": "pm",     // 同时保留 P 与 M checkpoint
"session_capture_terminal": true,      // pm 模式要求
"session_freeze_prompt": false         // pm 与 frozen-P 互斥
```

**实测（修前 → 修后）**
| 场景 | 修前 | 修后 |
|---|---|---|
| 7,254 token 会话续写（>窗口） | 10.45 s（全量重 prefill）| **0.40 s（26×）** |
| 同上第三轮 | — | **0.40 s** |
| 第二个会话（7,254）续写 | 10.3 s | **0.40 s** |
| 2,212 token 会话续写（<窗口） | 0.59–0.64 s | **0.25–0.26 s（9.5×）** |
| A/B 会话交替（小 prompt） | 0.64 / 0.59 / 0.63 s | **0.26 / 0.25 / 0.25 s** |

`sessions.hits=6, entries=3`，`session_restored_tokens=26,259`。
回归：答案全部正确（`ZX-4417-QJ7` / `QQ-8882-ZZ9`）。

**副作用/代价**：pm 模式会额外保留终端（M）checkpoint ⇒ 每会话 payload 更大
（实测 3 个会话 236 MB）；容量仍由 `session_capacity=8` × `session_checkpoint_bytes` 约束，
不够时调大后者或减小窗口。

## 32. 并发实测：池子按 2×(B+R) 给到了，但请求被**串行调度**

**配置（按用户口径：并发2 = 2 倍窗口）**
| 项 | 值 |
|---|---|
| 窗口 B+R | 4,096 + 2,048 = **6,144** |
| 并发倍数 | `max_num_seqs=3`（给退役中的请求留 1 个名额）|
| `cudagraph_capture_sizes` | `[5, 10, 15]`（query_len 5 × batch 1..3）|
| 池 | `gpu_mem=0.82` ⇒ **60,074 token**（≈3 组 × 2 并发 × 6,144 = 36.9K 之上）|

**实测结果**
| 场景 | 结果 |
|---|---|
| 两个极小 prompt（~20 token）并发 | **0.30 s 两个都完成** ⇒ 批量调度本身是通的 |
| 两个 2,214 token（不同会话）并发 | 墙钟 **4.34 s**（单个约 2.2 s）⇒ 完全串行 |
| 两个 7,254 token（超窗口）并发 | 墙钟 **21.0 s**（单个约 10.5 s）⇒ 完全串行 |
| 准入计数 | `authority_limit=3`、`rejections=0`、`waits=0` ⇒ **不是容量/准入限制** |
| 每步并发请求数 | 27 步 1 个请求、**仅 2 步有 2 个** ⇒ 调度器基本一次只跑一个 |

**排除过的原因**
- 池容量不足：不是（把池从 30K 加到 60K，`rejections` 归零后依然串行）
- 退役名额：不是（`max_num_seqs` 3 后 `retirement_or_active_limit=0`）
- MTP 草稿契约：不是唯一原因（`draft_tokens=1` 时 2 并发步数从 1 增到 4，墙钟不变）
- `request_planning`：是 bounded target 的前置，关掉直接启动失败
  （`Bounded target batch budget must fit an aligned block and admitted G headroom`）

**结论**：并发瓶颈在 **kvmem 的有界窗口执行路径**——一次只为一个请求建立选片帧/查询捕获流，
与上游 `docs/design-native-control-plane.md` 的说明一致（capacity reservation、bounded execution、
MTP draft contract、archive/window 路径都要求"同步单批"执行）。**用户的容量模型（2×B+R）是对的**，
但当前插件还没把它用于并行，只用于排队。

**要在本机拿到真正并发**，需要插件层改造（不属于配置）：
1. 每请求一个选片帧/查询捕获流（现在共享单流）；
2. 容量记账按"活跃帧数"而不是"单活跃请求"；
3. 归档/恢复的 staged import 支持多请求并行。
这是架构级改动，需先与你确认再做。
