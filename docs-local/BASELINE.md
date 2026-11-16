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

## 5. 容量口径（实测 + 源码核对）

### 5.1 每 token 的归档字节

`adapters/cache_manifest.py::archive_bytes_per_token` 只统计 **full-attention 层**：

```
每层每 token（每 rank）= (num_kv_heads / TP) × head_dim × 2      # K+V 各 fp8 1 字节
本模型：(4/2) × 256 × 2 = 1,024 B；full_attention 16 层 + MTP 1 层 ⇒ 17,408 B/token/rank
```
与 `cache-geometry.json` 的 `cpu_archive_bytes_per_token = 17,408` 完全一致。
**linear_attention（GDN，48 层）不存按 token 的 KV**（其循环状态是"每请求一份"），
这是与旧 ninfer 口径的关键差异（旧口径按 64 层算 = 65,536 B/token）。

### 5.2 CTX → 主机内存（TP=2 有两个 archive）

| CTX | 每 rank | 主机合计（TP2）| 建议 `cpu_read_cache_bytes` | 主机规划 |
|---|---|---|---|---|
| 32,768 | 0.53 GiB | 1.06 GiB | 1 GiB | ~4 GiB |
| 131,072 | 2.12 GiB | 4.25 GiB | 3 GiB | ~8 GiB |
| **262,144** | **4.25 GiB** | **8.50 GiB** | **5 GiB** | **13–16 GiB** |
| 524,288 | 8.50 GiB | 17.0 GiB | 9 GiB | ≥24 GiB |

### 5.3 多会话（`session_capacity` / 旧 RetainedSessions）**共享**一个 RAM 历史池

- 历史池是**进程级共享 arena**（`core/storage.py::make_shared_archive`：
  "One process-wide arena; each admitted authority has a lifetime partition"）。
- 名义容量 = `repository_blocks × (max_requests + session_capacity)`，
  再被**字节预算**夹住：`capacity = min(nominal, cpu_read_cache_bytes ÷ record_bytes)`。
- ⇒ **会话不会自动获得独立的 CTX 预算**；N 个会话各自保留 H token 的历史，
  就共同消耗 1 个池的 N×H。8 会话各 256K ⇒ 需要 8 × 4.25 GiB/rank（超出会在准入时被拒）。
- 会话 checkpoint 是**另一个池**：`session_checkpoint_bytes` 是**所有会话合计**预算
  （`PromptCheckpointPool.payload_bytes` 累计），实测 3 个会话（2.2K+2.2K+7.2K）= 236 MB。

**配置公式**
```
历史 token 上限 = min( CTX,
                       cpu_read_cache_bytes / 17,408 ,        # 每 rank，且被所有会话共享
                       repository_blocks × 32 )
会话 checkpoint 总量 ≤ session_checkpoint_bytes（所有会话合计）
```

### 5.4 场景：CTX=256K、`session_capacity=4` 会怎样

| 层面 | 结果 |
|---|---|
| 逻辑上限 | 每个会话都可用到 **262,144 token**（`max_model_len`；fork 位置表上限正好 262,144）|
| GPU | 窗口仍是 `B+R=6,144`，与 CTX 无关；池只需要能装下窗口（靠 `KVMEM_SKIP_KV_CHECK` 解耦）|
| 单请求耗时 | 256K 冷 prefill ≈ **4–5 分钟**（实测 ~1,000 tok/s）；命中后 ≈ 1–2 s |
| **历史 RAM（共享池）** | 4 个会话**共用**一个 arena：若都跑到 256K ⇒ 1,048,576 token ⇒ **8.5 GiB/rank、17 GiB 主机** ⇒ 需 `cpu_read_cache_bytes ≥ 9 GiB/rank` |
| 若只给 5 GiB/rank | 共享池 ≈ **308K token 总量**：例如「1 个 256K + 1 个 50K」；**再多会撞归档上限（硬失败，不是跳过）** |
| **会话 checkpoint（另一个池）** | 实测每会话 **60–80 MB**（窗口快照+模型状态为主，与上下文长度弱相关）⇒ 4 会话 ≈ 240–320 MB，`session_checkpoint_bytes=512 MiB` **够** |
| checkpoint 超预算时 | **优雅跳过**（`_capture_not_admitted`，`checkpoint_deferrals+1`），本次照常出结果，只是下轮不命中 |
| 归档超预算时 | **不是优雅降级**：`BufferError: Authority capacity exhausted`（实测会让引擎 core 报错）⇒ 归档池必须按"最大单会话"给足 |

**结论**：CTX=256K + 4 会话 **可以配，但要分开算两笔账**：
```
历史归档 = 4 × 每会话实际历史   （共享池，必须 ≥ 最大单会话，否则硬失败）
checkpoint = 4 × 60–80 MB      （共 512 MiB 够用；超了只是不命中）
```
想「4 个会话都能各放 256K」⇒ `cpu_read_cache_bytes ≥ 9 GiB/rank`、主机 ~17–20 GiB；
想「4 个会话共享 ~256K 总量」⇒ `cpu_read_cache_bytes = 5 GiB`、主机 ~13–16 GiB。

## 6. 默认配置（对外，保守）与变体

**默认 profile**：`local/serve-2080ti-default.json`（`deploy/run-kvmem-vllm.sh` 与 systemd 单元默认指向它）

| 项 | 值 | 为什么保守 |
|---|---|---|
| `max_model_len`（CTX）| **32,768** | 位置表上限 262,144，但默认要能在 22 GB 卡上安稳启动 |
| `gpu_memory_utilization` | **0.80** | 池 45,511 token；实际占用 17.7/22.5 GB，留 ~4.7 GB 余量 |
| B + R | 4,096 + 2,048 = 6,144 | 窗口小 ⇒ 单步快、显存确定 |
| `cpu_read_cache_bytes` | **4 GiB/rank** | 主机 62 GB 的机器上留给其余服务；可容纳 246K token 历史 |
| `session_capacity` | **4** | **4 × 32,768 = 131,072 token ⇒ 2.12 GiB/rank**，预算 4 GiB ⇒ **1.88× 余量** |
| `session_checkpoint_bytes` | 512 MiB（合计）| 实测每会话 60–80 MB ⇒ 4 会话 ≈ 240–320 MB |
| 执行 | `full_graph` + `draft_tokens=4` + `capture=[5,10,15]` + `max_num_seqs=3` | 代码类生成 80–87 tok/s |
| 会话 | `pm` + `capture_terminal=true` | 多轮续写能命中（`sealed/frozen-P` 不行）|
| 自动会话 | `auto_session=true` | 客户端不需要传 `session_id` |

**一致性规则（改配置时必查）**
```
会话数 × CTX ≤ cpu_read_cache_bytes / 17,408          # 每 rank，否则新会话会一直排队
会话数 × 80 MB ≤ session_checkpoint_bytes             # 否则 checkpoint 静默跳过（下轮不命中）
```

**变体**
| 文件 | CTX | 会话 | `cpu_read_cache_bytes` | 每 rank 需要 | 主机 RAM 建议 |
|---|---|---|---|---|---|
| `serve-2080ti-default.json` | 32,768 | 4 | 4 GiB | 2.12 GiB | ≥ 12 GiB |
| `serve-2080ti-128k.json` | 131,072 | 1 | 5 GiB | 2.12 GiB | ≥ 14 GiB |
| `serve-2080ti-256k.json` | 262,144 | 1 | 6 GiB | 4.25 GiB | ≥ 16 GiB |

**注意**：默认假设 2×22 GB（TP=2）。单卡用户需要改 `tensor_parallel_size` 并降 `gpu_memory_utilization`
（27B FP8 权重约 32 GB，单卡 22 GB 装不下）。

## 7. 默认配置：B=32K / R=8K / CTX=256K / 0.96（实测可行域）

**用户目标**：B=32K、R=16K、`session_capacity=1`、CTX=256K、`gpu_memory_utilization=0.96`。
实测 **R=16K 在这张 22 GB 卡上放不下**，原因已定位：

**GPU 内存账（0.96 ⇒ 20.63 GiB 目标，实测 free 21.12 GiB）**
| 配置 | 权重+非 torch | 峰值激活 | 剩给 KV | 池 token | 结果 |
|---|---|---|---|---|---|
| B=32K, **R=16K** | **18.42 GiB** | 2.06 GiB | **0.15 GiB** | 7,363 | ❌ 需求 147,456（3×48K）|
| B=32K, **R=8K** | 16.60 GiB | 1.05 GiB | 2.95 GiB | **166,417** | ✅ 需求 122,880（1.35×）|
| B=24K, R=8K | 16.60 GiB | 1.12 GiB | — | 166,417 | ✅ |
| B=16K, R=8K | 16.60 GiB | 1.12 GiB | — | 163,471 | ✅ |

**结论：额外占显存的是 R，不是 B** —— B 从 16K 增到 32K 占用不变（16.60 GiB），
R 从 8K 增到 16K 多花 **1.82 GiB**（≈227 KiB / R-token，MTP 草稿+生成预留缓冲）。
同时"最坏请求"按 `3 × (B+R)` 预留（3 个 attention 组），所以 R 越大两头都吃亏。

**R 不限制输出长度**：实测 `max_tokens=12000` 的长输出请求跑完 **~11K token @ 84 tok/s**。
`gen_budget` 只是窗口里的生长预留；vLLM 侧仍按 `max_model_len` 校验 prompt+output
（`http_sessions.py:137` 的 "Single-turn output must remain within G" 只在带
`kvmem_request_plan` 的显式规划路径生效）。

**最终默认（`local/serve-2080ti-default.json`）**
```
B=32,768  R=8,192 窗口=40,960      CTX=262,144      gpu_memory_utilization=0.96
session_capacity=1   cpu_read_cache_bytes=6 GiB   session_checkpoint_bytes=1 GiB
sink/recent = 4,096/2,048 token    max_num_seqs=1   capture=[5]   draft=4   full_graph
```

**验证（默认配置）**
| 项 | 结果 |
|---|---|
| 启动 | 57 s，池 166,417 token，占用 20.5/22.5 GB |
| 4,888 needle | `ZX-4417-QJ7` ✅ |
| 代码生成 | **86.3 tok/s** ✅ |
| 会话续写（7.2K，>窗口）| 7.57 s → **0.34 s / 0.28 s** ✅ |
| 长输出 | `max_tokens=12000` 生成 ~11K token @ **84 tok/s** ✅ |

**要 R=16K 的前提**：换 24 GB+ 的卡（+2.4 GiB 可用），或把 `gpu_memory_utilization` 提到 >1（不可行）。
