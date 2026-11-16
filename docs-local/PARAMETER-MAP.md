# 参数对照：旧 kvmem 适配器（v0.5） → 现在这个 fork（上游 kvmem-vllm）

旧配置来自 `kvmem-vllm-adapter@v0.5`（`docs/PARAMETERS.md`），新配置来自上游
`kvmem/kvmem-vllm`（vLLM 插件，`kv_connector_extra_config.kvmem`）。

## 一、预算类（最关键，单位变了）

| 旧名字 | 旧值 | 新名字 | 单位 | 现在的值 | 说明 |
|---|---|---|---|---|---|
| `B` | 36,864 | **`select_budget`** | **token** | 2,048 | 每步"被选中的历史 token 预算"。旧 B 是完整常驻预算，新的是**只按 Query 选中的部分** |
| `R` | 16,384 | **`gen_budget`** | token | 2,048 | 生成上限 + 增长预留；必须能被 `block_tokens` 整除 |
| `CTX` | 524,288 | **`max_model_len`**（vLLM 参数） | token | 131,072 | 逻辑上下文上限；kvmem 不再单独管，交给 vLLM |
| `SINK` | 8,192 | **`sink_blocks`** | **块**（×`block_tokens`） | 1 → **32 token** | ⚠️ **单位陷阱**：旧的是 token，新的是块。8,192 token = `8192/32 = 256` 块 |
| `RECENT` | 2,048 | **`recent_blocks`** | **块** | 1 → **32 token** | 同上：2,048 token = `2048/32 = 64` 块 |
| `SELECT_BLOCK_TOKENS` | 64 | **`block_tokens`** | token | 32 | 选片/统计网格大小，也是 sink/recent 的单位 |
| `N`（并发） | 1 | **`max_num_seqs`** | 请求数 | 1 | 工作窗口 ≈ `N × (select + gen)` |

**约束（`core/config.py:25` 实测）**
- `select_budget`、`gen_budget` 必须能被 `block_tokens` 整除
- `sink_blocks + recent_blocks` 不能超过 `select_budget / block_tokens`
- `repository_blocks × block_tokens ≥ select_budget + gen_budget`
- `None`（不填）会由 `_native.resolve_keep_bands` 自动分配。实测：

| block_tokens | select_budget | sink/recent | 结果 |
|---|---|---|---|
| 32 | 2,048 | 1 / 1 | 32 / 32 token |
| 32 | 16,384 | 默认 | 1,024 / 4,096 token |
| 32 | 16,384 | 256 / 64 | **8,192 / 2,048 token**（= 旧 SINK/RECENT）|
| 32 | 65,536 | 默认 | 1,024 / 5,248 token |

## 二、内存/存储类

| 旧 | 旧值 | 新 | 现在的值 | 说明 |
|---|---|---|---|---|
| `HOST_MIB`（H 层 KV 预算） | 49,152 MiB | **`cpu_read_cache_bytes`** + **`repository_blocks`** | 4 GiB / 16,384 块 | 旧的一个池，现在拆成"读缓存"和"历史块数"两个上限；`repository_blocks × block_tokens` 就是能存多少历史 token（16,384×32=524,288） |
| `SESSIONS`（H 池份额） | 4 | **`session_capacity`** + **`session_checkpoint_bytes`** | 8 / 512 MiB | 会话复用：条目数 × 单 checkpoint 字节上限 |
| — | — | `archive_backend` | `cpu` | 上游把历史放 CPU archive（旧设计是 H 层）|
| — | — | `cpu_history_cache_bytes_per_request` | 0 | 每请求历史缓存（≤ `cpu_read_cache_bytes`）|

## 三、执行/引擎类

| 旧 | 旧值 | 新 | 现在的值 |
|---|---|---|---|
| `GPU_UTIL` | 0.972 | `gpu_memory_utilization` | **0.88**（22G×2，池 198,817 token）|
| `MAX_BATCHED_TOKENS` | 4,096 | `max_num_batched_tokens` | 1,024（启用 request planning 后实际 2,052）|
| `SPECULATIVE_TOKENS` | 4 | `draft_tokens`（`draft_method=mtp`） | 1 |
| `KVMEM_MAX_MODEL_LEN` | 200,000 | `max_model_len` | 131,072 |
| `CAPTURE_TOKENS` | 512 | 无对应（上游自动按步捕获） | — |
| `DISABLE_PREFIX_CACHING` | 0 | 上游固定 `--enable-prefix-caching False` | 恒关 |
| `KVMEM_COMPACT_REBUILD` | 1 | 无开关（上游内部总是选片+重建/装配） | 恒开 |
| `KVMEM_ENTRY_*`（入口策略） | retrieval | 无（上游不在入口裁剪） | — |
| `KVMEM_UI_DIR` | — | `vllm-kvmem ui` 子命令 | — |

## 四、旧配置等效 profile

`local/serve-2080ti-old-equiv.json` 把旧的 B/R/SINK/RECENT 原样搬过来：
`select_budget=36,864`、`gen_budget=16,384`、`sink_blocks=256`、`recent_blocks=64`、
`repository_blocks=16,384`、`max_model_len=131,072`、`gpu_memory_utilization=0.88`。
工作窗口变为 `36,864 + 16,384 = 53,248` token（旧的 B+R 同上），GPU 池 198,817 足够。
