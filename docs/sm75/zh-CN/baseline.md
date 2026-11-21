# KVMem vLLM 2080Ti 基线（冻结版）

> 本文件记录**当前冻结的可用基线**。之后的改进（真正并发、prefix image 多任务等）
> 留待上游推进后再做，不要在此基线上随意改动。

## 1. 组成

| 层 | 内容 | 位置 |
|---|---|---|
| 引擎 | vLLM `0.2.2.post3` SM75 checkout + **1 个 15 行补丁**（`KVMEM_SKIP_KV_CHECK`）| `$KVMEM_ENGINE_DIR` |
| 机制 | 上游 `kvmem/kvmem-vllm` + SM75 兼容补丁 + auto-session | 本仓库 `src/` @ `d4e27d1` |
| 补丁集 | `patches/0001-engine-skip-kv-check.patch`、`patches/0002-kvmem-sm75-compat.patch` | `patches/` |
| 启动 | `deploy/run-kvmem-vllm.sh start`（内部 `scripts/serve_watch.py`，信号轮询不硬等）| `deploy/` |
| 服务 | `deploy/kvmem-vllm.service`（Type=oneshot + RemainAfterExit）| `deploy/` |
| 配置 | `local/serve-2080ti-baseline.json` | `local/` |

## 2. 冻结配置要点

| 项 | 值 | 说明 |
|---|---|---|
| 模型 | `$KVMEM_MODEL_DIR`（FP8 w8a16，MTP 权重 fp8，`lm_head` 保持 BF16）| SM75 无 bf16 ⇒ `dtype=float16` |
| TP | 2 | 单卡装不下 27B |
| KV | `fp8_e4m3` | kvmem 上游要求的显式拼写（与 `fp8` 等价）|
| `max_model_len`（CTX）| 32,768（dev）/ 262,144（长上下文）| 逻辑上限，与池解耦（skip-check）|
| B + R | `select_budget=4096` + `gen_budget=2048` = **6,144** | 单请求工作窗口 |
| sink / recent | 32 / 32 块 = **1,024 / 1,024 token** | |
| `block_tokens` | 32 | 选片网格 |
| `repository_blocks` | 16,384（× 32 = 524,288 token 名义）| 历史块数上限 |
| `cpu_read_cache_bytes` | 4 GiB | **RAM 实际约束**（17,408 B/token ⇒ ~241K token）|
| 会话 | `session_capacity=8`、`session_checkpoint_bytes=512 MiB`、`policy=pm`、`capture_terminal=true`、`freeze_prompt=false` | pm 才能多轮续写命中 |
| 执行 | `execution_mode=full_graph`、`draft_tokens=4`、`max_num_seqs=3` | SM75 全图 + 融合多步 draft；`cudagraph_capture_sizes` 不填，按 `(draft_tokens+1) × 1..max_num_seqs` 自动推导（此处即 `[5,10,15]`）|
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

## 7. 默认配置（最终）：B=32K / R=16K / CTX=256K / 0.96

**用户口径的两点已核实**

1. **`gpu_memory_utilization` 的语义**：它是"向驱动申请多少显存"的比例，vLLM 在这额度内分配
   权重 + 峰值激活 + CUDA graph + KV 池，**剩下不申请的部分是留给同卡其他进程的冗余**。
   所以越大越好（只要不 OOM）；0.96 时实测占用 21,074 MiB / 22,528 MiB ⇒ 余量 1.45 GiB，安全。
2. **"KV 看起来是我们的 3.5 倍"的根因：`query_replay_budget` 默认等于 `gen_budget`（R）**。
   旧 ninfer 的对应项是固定小值（`CAPTURE_TOKENS=512`），所以它的内存只随 B+R 增长；
   我们 R=16K 时 replay 也被拉到 16,384，多占约 3 GiB 缓冲区，把池挤到 7,363 token。

**实测三组（B=32K、CTX=256K、gpu=0.96）**
| `gen_budget` R | `query_replay_budget` | 权重+非torch | 池 token | 结果 |
|---|---|---|---|---|
| 16,384 | **16,384（默认=R）** | 18.42 GiB | 7,363 | ❌ 需要 147,456 |
| 16,384 | **2,048** | 15.45 GiB | **270,980** | ✅ 1.84× |
| 16,384 | **4,096（默认值）** | 15.77 GiB | **240,053** | ✅ 1.63× |

成本经验值：**replay 每 token 约 180 KiB**（ranged 150–190）。所以 replay 应显式给一个
"够用的固定值"，不要跟着 R 走。

**最终默认（`local/serve-2080ti-default.json`）**
```
B=32,768  R=16,384 窗口=49,152     CTX=262,144     gpu_memory_utilization=0.96
query_replay_budget=4,096          session_capacity=1
cpu_read_cache_bytes=6 GiB         session_checkpoint_bytes=1 GiB
sink/recent=4,096/2,048            max_num_seqs=1  capture=[5]  draft=4  full_graph
```

**验证（最终默认）**
| 项 | 结果 |
|---|---|
| 启动 | 56 s，池 **240,053** token，占用 21,074/22,528 MiB |
| 代码生成 | **79.5–79.8 tok/s** ✅ |
| 会话续写（7.2K > 窗口）| 7.51 s → **0.33 s** ✅ |
| 4,888 needle | `ZX-4417-QJ7` ✅ |
| 长输出 | `max_tokens=12000` ⇒ ~11K token @ 84 tok/s ✅ |
| 归档预算 | 1 会话 × 256K = **4.25 GiB/rank**，预算 6 GiB（1.41×）|

**replay 的语义边界**：`http_sessions.py:124` 的检查是
`plan.replay_end - plan.replay_begin > query_replay_budget ⇒ 拒绝`，只对**显式带
`kvmem_request_plan` 的客户端**生效；auto-session 路径（我们的默认）不传 plan，
所以 replay 只当"缓冲区大小"，设 4,096 不会限制提问长度。

## 8. 输出到达 R（窗口边界）之后怎么处理

**触发条件**（`serving/request_plan.py::action`）
```python
boundary = capacity - block_tokens          # capacity = B + R
if history - offset >= boundary:
    return 'pressure' if rolling_decode or prefill_window != 'keep_selected' else 'exhausted'
```
即生成把本地窗口填满时（`B+R-block_tokens`），计划返回 `pressure`。

**处理动作：bounded refresh（有界刷新）**，与旧设计的"R 满了 → 追加 RAM → 重新检索 → 继续"一致：
1. 把新生成的 token 追加进历史归档；
2. 从**全量历史**重新选 `B` 个 token（sink 头 + recent 尾 + 中间按 Query 分数）；
3. **只重建增量**（`rebuilt_tokens`），不是重算全量；
4. 保持同一个 request / sampler / 工具帧，流式响应不中断。

**实测（B=4,096、R=2,048，窗口 6,144，让模型连续数到 9000）**
```
请求: prompt 42 token，max_tokens 9,000
结果: 200 OK，completion 9,000，wall 89.54 s ⇒ 100.5 tok/s，无卡顿（GPU busy 205/208 采样）
bounded_refresh.jsonl: 2 条
  ① history_begin=6112  selected_tokens=4096  query_tokens=5  rebuilt_tokens=2848
  ② history_begin=8128  selected_tokens=4096  query_tokens=5  rebuilt_tokens=3072
```
⇒ 9,000 token 输出只触发 **2 次刷新**，每次重建约 2.8–3.1K token
（按 ~1,000 tok/s 的 prefill 速度 ≈ 3 s/次 ⇒ 整体开销 **约 7%**）。

**两种模式**
| 模式 | 行为 |
|---|---|
| 默认（`rolling_active_window=false`）| 到 `B+R-block` 才刷新一次，重建增量；`max_tokens > R` 在**显式 plan** 路径会被拒（auto-session 路径不受限）|
| `rolling_active_window=true` | 把 G 当"纯物理余量"（文档原话：*G is physical decode headroom in rolling mode, not a response limit*），边界更频繁、每次更小 |

**刷新节奏（重要，纠正一个误解）**：`B` 一旦选中就是**满的**，所以"48K 以内不刷新"只对**极短 prompt**成立。
实测（B=4,096、R=2,048，窗口 6,144）：

| 情形 | 刷新点（history_begin）| 结论 |
|---|---|---|
| 短 prompt（42 token）| 6,112 → 8,128 | 首次 ~`B+R`，之后每 **~R**（+2,016）|
| 长 prompt（4,856 token > B）| 4,096 / 4,352 / 4,608（prefill 期）→ **6,624**（decode 期）| prefill 期按块边界刷新；decode 期每 **~R**（+2,016）|

⇒ **decode 阶段的刷新间隔 ≈ R（`gen_budget`）**，不是 B+R。
默认配置 R=16,384 ⇒ 每生成约 **1.6 万 token 刷新一次**（之前"4.8 万以内不刷新"的说法不成立，
那只适用于 prompt ≪ B 的第一段生成）。
每次刷新重建量实测 2,592–3,072 token（≈1.3–1.5×R），按 ~1,000 tok/s ≈ 2.6–3.1 s。

**失败边界**：若刷新无法准入（容量耗尽），计划返回 `exhausted`（会中断而非静默降级）。
本机测试未出现。

## 9. rolling 模式（B）实测对照：可用但**不兼容 chat**，速度也无差异

**开启条件（全部需要）**：`rolling_active_window=true` + `async_refresh=async_restore=true` +
`cpu_history_coordination=true` + `maintenance_max_steps>0` + `session_capture_terminal=true`（pm 已满足）。

**关键限制（实测）**：rolling 模式**只接受 token-ID 形式的 completion**：
```
POST /v1/chat/completions → 400 "Rolling active window requires a token-ID completion prompt"
POST /v1/completions {"prompt": [token ids], ...} → ✅
```
（`http_sessions.py:115`：`prompt` 必须是 int 列表）⇒ **与我们的 chat / auto-session 默认路径不兼容**。

**同负载对照（B=4,096 / R=2,048，窗口 6,144，9,000 token 计数输出）**
| 模式 | 接口 | wall | 吞吐 | GPU 忙 | 最长空转 | 刷新 |
|---|---|---|---|---|---|---|
| **A（默认，同步）** | chat | 89.54 s | **100.5 tok/s** | 205/208 | — | 2 次（`rebuilt=2848/3072`）|
| **B（rolling，异步）** | completions + token ids | 89.48 s | **100.6 tok/s** | 206/208 | **0.8 s** | 2 次（`asynchronous=True`，`rebuilt=None`）|

⇒ 吞吐**没有差异**（100.6 vs 100.5）。原因是这个窗口下同步刷新本身只占 ~3 s/89 s（约 3%），
而异步机制也有自己的开销（`maintenance_busy_ms=2.0` 每轮的合作预算）。
B 的收益要在"刷新占比更高"的场景（窗口远小于输出、或 R 很大）才可能显现。

**结论**：默认保持 **A**（chat 兼容、实现简单、速度相同）；
B 留给"纯 token 级 API + 长输出"的专门场景，且需要额外约 3.6 GiB 主机内存
（一个完整活跃视图 + 一个 checkpoint 的在飞预算）。

## 10. 更正：上下文上限不是 262,144

之前写的"硬顶 262,144"是错的。实测（`max_model_len=524288`、B=R=8,192、`gpu_memory_utilization=0.90`、
`cpu_read_cache_bytes=12 GiB`）：

| prompt | 结果 |
|---|---|
| 269,654 token（needle 在中间）| ✅ 答对 `ZX-4417-QJ7`，686 s，393 tok/s prefill |
| 299,280 token | ✅ `FINISHED_STOPPED`，无 OOM |

约束的真实来源：
1. 启动器对 MTP 允许 `max_model_len` 到 **10,000,000**，只要求 `request_planning` 打开且 `B+R ≤ 262,144`；
2. 真正的墙是**显存余量**：窗口 49,152 + `gpu_memory_utilization` 0.96/0.88 时同一个 300K prompt
   在 prefill 阶段 CUDA OOM；把窗口缩到 16,384 就通过；
3. 模型声明 262,144 个位置且 `rope_type=default`，超出属 RoPE 外推：能答对 needle，但长距离质量未验证。

## 11. 默认 `gpu_memory_utilization` 改为 0.90

0.96 时显存占用 21,074/22,528（余量 1.45 GiB），接近 CTX 上限的长 prefill 会 CUDA OOM；
0.90 时占用 19,760/22,528（**余量 2.77 GiB**），默认窗口 49,152 下池仍有 **167,889 token**
（需求 3×49,152 = 147,456，1.14×）。带超长上下文的成功案例（269K / 299K）本来就在 0.90 下跑通。

## 12. 更正：窗口的显存需求不是 "3 × (B+R)"

之前写"池需求 ≈ 3 组 × (B+R)"是错的。真实几何（`cache-geometry.json`）：

| 组 | 内容 | block_tokens | 是否随窗口增长 |
|---|---|---|---|
| 0/1/2 | 48 个 GDN 层（3 组 × 16）| = max_model_len | ❌ 每请求**固定** `1+4` 页 × 3 组 = **15 页 ≈ 0.39 GiB** |
| 3 | 16 个 full-attention + MTP（17 层）| **1,616 token/页**（26.8 MB/页）| ✅ `ceil(W/1,616)` 页 |

⇒ 窗口 W 的显存需求 ≈ **17,408 B × W + 0.39 GiB**：
`W=49,152` 只有 **31 页 ≈ 0.81 GiB**（+0.39 GiB 固定）= **1.21 GiB**。

实测印证：`gpu_memory_utilization=0.82` 时池只有 70,690 token（≈1.25 GiB）仍能起，
窗口 49,152 完全够；0.90 的池 2.99 GiB 是**富余**，不是需求。

"3.5 倍"的来源有两个误解叠加：①把 3 个 GDN 组乘到窗口上；②把 `query_replay_budget`（默认=R，
约 180 KiB/token）当成窗口开销。

## 13. 默认改为窗口 131,072（依据第 12 节的真实几何）

既然窗口只需要 `17,408 B × W + 0.39 GiB`，0.90 的 114 页池（2.99 GiB）就能放下 131K 窗口：

| 项 | 49,152 窗口 | **131,072 窗口（新默认）** |
|---|---|---|
| KV 需求 | 1.19 GiB | **2.52 GiB**（池 2.99 GiB，1.19×）|
| 代码生成 decode | 79.6–86.5 tok/s | **80.1 tok/s** |
| 258,743 token 冷 prefill | ✅ 787 s / 329 tok/s | ✅ **1,017 s / 254 tok/s**，无 OOM |
| 会话续写 | 0.25–0.34 s | **0.28 s**（7.67 s 冷）|

同时新增 `local/serve-2080ti-lean.json`（窗口 49,152、util 0.82、池约 1.25 GiB），
留给需要把显存让给其他进程的部署。

## 14. 与旧 vllm-2080ti（0.972 → 池 ≈201K）的对照

| `gpu_memory_utilization` | 池 | 可用窗口 | 代码生成 decode | 258K 冷 prefill |
|---|---|---|---|---|
| **0.90（默认）** | 167,889 token（103 页）| 142K | 80.1 tok/s @131K 窗口 | ✅ 1,017 s |
| 0.92 | 191,453 token（118 页）| 166K | — | — |
| **0.96** | 240,053 token（148 页）| **215K** | **80.0 tok/s @200,704 窗口** | ❌ CUDA OOM（引擎退出）|

结论：**池容量与旧 fork 同级甚至更好**（0.96 的 240K > 旧 0.972 的 201K），200K 窗口能跑、
decode 速度不变；但 `0.96` 下**最长 prompt 的 prefill 余量不够**（会 OOM）。
按 workload 选：长输入 → 0.90 + 131K 窗口；短输入多会话的大窗口 → 0.96。

## 15. 最终默认：util 0.92 / 窗口 147,456 / chunk 1,024

| 项 | 值 |
|---|---|
| `gpu_memory_utilization` | **0.92**（占用 20,188/22,528 MiB，余量 2.34 GiB）|
| 窗口 | **147,456**（B=131,072 / R=16,384，sink/recent = 8,192/4,096）|
| 池 | 191,453 token = 118 页 ≈ 3.09 GiB；需求 107 页 ⇒ **1.10×** |
| `max_num_batched_tokens` / `prefill_chunk_tokens` | **1,024**（logits 缓冲 0.49 GB，4,096 时 1.94 GB 会让最长 prefill OOM）|
| CTX / 会话 / 归档 | 262,144 / capacity 1 / 6 GiB |

实测：代码生成 **80.1 tok/s**；7,252 token 会话续写 7.51 s → **0.33 s**。
新增 `local/serve-2080ti-200k.json`（util 0.96 + 窗口 200,704 + chunk 1,024，启动已验证）。

## 16. 最终默认：B=36K / R=4K（窗口 40,960），chunk 2,048，util 0.92

> 2026-11-21 更新：仓库默认已改为 `R=8,192`（窗口 45,056）、`sink=8,192`；本节保留当时的 R=4K 记录。

窗口大小是**产品选择**，不是容量上限（0.92 的池 118 页 ≈3.09 GiB，40,960 窗口只用 41 页 ≈1.07 GiB）。

| 项 | 值 |
|---|---|
| `select_budget` (B) | **36,864** |
| `gen_budget` (R) | **4,096**（窗口 40,960）|
| sink / recent | 4,096 / 2,048 token |
| `prefill_chunk_tokens` / `max_num_batched_tokens` | **2,048**（logits ≈1.0 GB）|
| `gpu_memory_utilization` | 0.92（占用 20,188/22,528 MiB）|
| CTX / session / 归档 | 262,144 / 1 / 6 GiB |

实测：代码生成 **80.1 tok/s**；7,252 token 会话 7.52 s → **0.33 s**。
注意刷新节奏随 R 变小：`R=4,096` ⇒ decode 每生成约 4K token 触发一次 bounded refresh（每次重建约 1.3–1.5×R）。
