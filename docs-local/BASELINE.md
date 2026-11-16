# KVMem vLLM 2080Ti 基线（冻结版）

> 本文件记录**当前冻结的可用基线**。之后的改进（真正并发、prefix image 多任务等）
> 留待上游推进后再做，不要在此基线上随意改动。

## 1. 组成

| 层 | 内容 | 位置 |
|---|---|---|
| 引擎 | `vLLM-2080Ti-Definitive` v0.2.2.post3 + **1 个 15 行补丁**（`KVMEM_SKIP_KV_CHECK`）| `/home/srj/vLLM-2080Ti-Definitive` @ `01506a5471` |
| 机制 | 上游 `kvmem/kvmem-vllm` + SM75 兼容补丁 + auto-session | 本仓库 `src/` @ `d4e27d1` |
| 补丁集 | `patches/0001-engine-skip-kv-check.patch`、`patches/0002-kvmem-sm75-compat.patch` | `patches/` |
| 启动 | `deploy/run-kvmem-vllm.sh start`（内部 `scripts/serve_watch.py`，信号轮询不硬等）| `deploy/` |
| 服务 | `deploy/kvmem-vllm.service`（Type=oneshot + RemainAfterExit）| `deploy/` |
| 配置 | `local/serve-2080ti-baseline.json` | `local/` |

## 2. 冻结配置要点

| 项 | 值 | 说明 |
|---|---|---|
| 模型 | `/home/srj/models/Coder390/FP8-fp8mtp`（FP8 w8a16，MTP 权重 fp8，`lm_head` 保持 BF16）| SM75 无 bf16 ⇒ `dtype=float16` |
| TP | 2 | 单卡装不下 27B |
| KV | `fp8_e4m3` | kvmem 上游要求的显式拼写（与 `fp8` 等价）|
| `max_model_len`（CTX）| 32,768（dev）/ 262,144（长上下文）| 逻辑上限，与池解耦（skip-check）|
| B + R | `select_budget=4096` + `gen_budget=2048` = **6,144** | 单请求工作窗口 |
| sink / recent | 32 / 32 块 = **1,024 / 1,024 token** | |
| `block_tokens` | 32 | 选片网格 |
| `repository_blocks` | 16,384（× 32 = 524,288 token 名义）| 历史块数上限 |
| `cpu_read_cache_bytes` | 4 GiB | **RAM 实际约束**（17,408 B/token ⇒ ~241K token）|
| 会话 | `session_capacity=8`、`session_checkpoint_bytes=512 MiB`、`policy=pm`、`capture_terminal=true`、`freeze_prompt=false` | pm 才能多轮续写命中 |
| 执行 | `execution_mode=full_graph`、`draft_tokens=4`、`cudagraph_capture_sizes=[5,10,15]`、`max_num_seqs=3` | SM75 全图 + 融合多步 draft |
| 引擎开关 | `VLLM_SM75_SPEC_SYNC_MODE=nosync`、`VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1` | 从 fork fast 档移植 |
| 显存 | `gpu_memory_utilization=0.82`（池 **60,074** token）| 0.96 可到 30 万 token，dev 用小值省时间 |

## 3. 已验证（真机）

| 能力 | 结果 |
|---|---|
| 生成速度（代码类）| **79.6 / 86.5 tok/s**（原版同配置 80.8 / 87.2）|
| 生成速度（散文类）| 44.6–46.4 tok/s（MTP 接受率低所致）|
| prefill | 1,021–1,063 tok/s（原版 1,174）|
| 超池 prompt | 218,220 token 答对；重复请求 1.36 s（**365×**）|
| 会话续写（>窗口）| 10.4 s → **0.40 s**（pm 模式修复后）|
| 会话续写（<窗口）| 2.4 s → **0.25 s** |
| 多会话交替 | A/B 各自命中，0.25–0.40 s |
| 自动会话 | 客户端不传 `session_id` 也能命中（auto-id = 会话开头 hash）|

## 4. 已知边界（未做，等上游）

1. **并发是串行的**：`max_num_seqs=3`、池够 2 倍窗口，但窗口化请求仍被逐条调度
   （每步只有 1 个请求）。需要插件层"每请求选片帧/查询捕获流"改造。
2. **prefix image（多任务共享前缀并行）**：上游限制 TP=1，本机 TP=2 不可用。
3. **256K+ 上下文**：受主机 RAM（`cpu_read_cache_bytes`）与位置表（262,144）约束。
4. 服务化：systemd 单元已写但未安装（`systemctl --user`/system 需你确认后安装）。
