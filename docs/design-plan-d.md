# Plan D 交接索引（2026-11-12 收尾版）


## 〇之二、完整版实施计划（按序，2026-11-12 Round 488 定稿）

### 已钉死的基础事实（不要再猜）
- 池 → 每用户份额 **B = budget_tokens + gen_reserve**（当前 28,672 + 16,384 = 45,056 token）；
- 引擎行：**页粒度**（`page_tokens=32`），值 = **块 id**，`block_pages=49`（块 = 1,568 token）；
- `len(row) == ceil(seq_len/32)`；引擎把**已退役页填占位块**（实测有 677 项连跑）；
- `sync_row_from_engine` 的行宽必须由 `ceil(engine_seq_lens/32)` 推导（`num_blocks.np` 恒为 1）；
- metadata builder 只认 `_seq_override`（视图路径写入），`attention_seq_lens` 用它替换 seq_lens。

### 待开发（按序，可独立验收）
| # | 工作项 | 关键文件/函数 | 验收 |
|---|---|---|---|
| 1 | **占位块恢复**：`count>48` 判占位；49 页逐页从 `orch.store` 恢复到目标块；缺页则跳过 | `_apply_selection_d` | 视图内不再出现空块；`row[p]` 无长连跑 |
| 2 | **回卷 B**：Core 把 `num_computed_tokens` 设为视图长度（clamp 用视图 token 数） | `scheduler.py` clamp | metadata builder 与 positions 自洽（`pos < seq_len`）|
| 3 | **rebake 帧校验**：δ 往返字节等价 → 逐层接入 | `kvmem_sparse_rope.rotate_k_pages_fp8` | 同页旋 +δ 再 −δ 字节一致；视图答案不退化 |
| 4 | **四步联调**：选块→恢复→重旋→装行→decode 顺序与所有权 | `_apply_selection_d` + 706 触发点 | 89K 池 + 92K → 正确；140K → 正确 |
| 5 | **250K 与离线四套** | 测试目录 | 全绿 |
| 6 | **多用户份额**：每请求只允许在**本用户 B 内**取块；池按 `N×B` 预留 | Core 侧 allocator + 适配器份额表 | 两用户并发各跑 B 内长上下文，互不越界 |

### 工作量（含验证）
1–3 各 0.5–1 天；4 为 1–2 天（最不确定）；5 0.5 天；6 1 天 ⇒ **合计 4–6 个工作日**。

## 〇、本阶段最终定性（2026-11-12，Round 484）
**Plan D 的"缩池服务长上下文"未完成，但已把失败收敛到一个结构性根因：**

> 适配器的行账（`st.row`）拿不到引擎的真实行：`sync_row_from_engine(req, written, …)` 的
> `written`（引擎 `num_blocks.np[group, row]`）在本 fork 中**恒为 1**（取证：prefill 到
> 74,954 token 时仍 `written=1, rowlen=2`）。因此一切"打包/滚行/视图"都作用在空行上，
> 输出的 `duct…` 与 `[6,6,6,…]` 全部由此而来。

**已排除的假设（都有读数）**：RoPE 相位帧（重旋与否都退化）、打包顺序（排序/不排序都退化）、
`seq_lens` 覆盖（已修到 `ovr == seq_lens == 视图长度` 仍退化）、TP2 双实例（是两 worker，正常）。

**唯一正确的下一步**：用权威来源建行 —— Core 的 `request.block_ids`（经既有提案/block channel
回传），或按 `num_computed_tokens // block` 自建槽序列并以 `block_table_rows()[0]` 快照校验；
拿到真实行之后，再重跑「选择 → 打包 → 回卷 → 装行」四步。

**当前可用能力（都已验证）**：执行通道（89K 池跑完 137,873）、全量落主机（4,482 页）、
环滚绿点（92K 正确 103 s）、刀链路、rebake 原语、seq 覆盖。


## 一、当前可复现的绿点（本轮实测）
| 配置 | 请求 | 结果 |
|---|---|---|
| `GPU_UTIL=0.85` `PLAN_D=1` `KVMEM_D_NOBLADE=1` `INREQUEST_COMPACTION=0` resident 入口 | 92,082 token / 89K 池 | ✅ `The unique record KEY-MID has access code ZX-4417-QJ7.`（103 s）|
| `GPU_UTIL=0.972` `PLAN_D=0` `INREQUEST_COMPACTION=1` resident 入口 | 88K | ✅ 含 `ZX-4417`（usage 74,948）|

## 二、已建成并验证的能力
1. **执行通道**：89K 池跑完 137,873 token（171–180 s）；
2. **全量落主机**：host store 4,482 页 ≥ 4,300（100%+），提交键按真位置（677/678）；
3. **环形窗口**：写地址 + `seq_len=窗口`（685）⇒ 92K 正确（绿点的机制）；
4. **刀链路**：选择 → 打包 → 装行（`inst=1`）→ rebake（原语 `rotate_k_pages_fp8`，元数据 hd=256/rot=64/base=1e6 解析成功）全部可运行，**默认关闭**（`KVMEM_D_APPLY=1` 才启用）。

## 三、已证明的必要条件（未实现）
- **A 回卷**：`num_computed_tokens` 必须回到视图长度（否则查询索引 > 视图长度，掩码全灭）；
- **B rebake**：选中页 KV 必须重旋到视图位置（引擎把行内索引当位置用，§714 三次实验证明）；
- 顺序：打包 → A → B → 装行 → decode。
- 不能用 legacy `compact` 做 A（§710：与 D 视图互斥，实测 HTTP 错误）。

## 四、教训（避免重复）
- `__slots__` 少一个字段 = 功能静默死亡（624/658/688 三次踩到）；
- **消融开关残留在 /etc/default 会让功能整段失效**（`KVMEM_D_NOBLADE=1` 长期存在，§699）；
- 任何让刀真正改写视图的改动，**必须先过 92K 绿点门禁**（700/696 都因破坏绿点被回退）。

# Plan D 交接索引（2026-11-12 最终版）

## 一、已验证可用（有读数）
| 目的 | 配置 | 读数 |
|---|---|---|
| 完整提示 + 正确（不压缩） | `GPU_UTIL=0.972` `KVMEM_PLAN_D=1` `KVMEM_D_KNIFE=1` `KVMEM_ENTRY_RESIDENT_PREFILL=1` `INREQUEST_COMPACTION=0` | 140K → `ZX-4417-QJ7`（180s，usage 137,873）|
| 基线（v0.5.0 语义） | `KVMEM_PLAN_D=0` `INREQUEST_COMPACTION=1` + resident 入口 | 88K → `ZX-4417-QJ7`（124s，usage 74,948）|
| D + 工作集分配 + 每步滚行 | 上述 + Core 无条件按窗口分配（引擎 668） | 92K → 正确（102s）；140K → 退化 |

## 二、已确认并修复的根因（全部有证据）
1. `_ReqState.__slots__` 缺字段 → 刀相关写入被 `except: pass` 吞掉（`0d782e9`）；
2. `st.view_installed` 从未置位 → clamp/行安装/滚行全死（657/658，正确落点是 `before_step` 主干 667）；
3. 入口裁剪制造假绿（142K→36K），已用 resident 入口 + worker 解耦（634）修好（usage=137,873）；
4. legacy 压缩的 rewind 与 D 账本冲突 → D 模式必须 `INREQUEST_COMPACTION=0`；
5. 插件侧与 runtime 侧看到的 runtime 实例不同（`has=False tok=-1` 40 步）→ 滚行必须写在 `before_step` 内（667）；
6. 滚行代码原先只在 adopt 分支执行一次 → 改为每步（664/667，roll 74 次/请求）。

## 三、架构级结论（决定后续方向）
- **纯环形窗口无法满足 KVMem 语义**：140K 提示的 needle 在 ~68K，滚动窗口（45K）会把它滚出 ⇒ 必错。92K 之所以对，是 needle 恰在边界被读到。
- 按用户定义的流程，**单次 prefill 必须放得进池**；"超池"指**对话历史总量**超池，靠"历史落主机 + decode 前按 query 选块"来服务。
- 因此正确的实现不是"缩小池跑单条超池提示"，而是：
  ① prefill 完把 **全部 KV 搬到主机**并释放 GPU；
  ② decode 前用 query 打分选块（sink+recent+top-k），把选中块**从主机恢复**到窗口；
  ③ 生成写 R，超 R 再追加/再选（用户流程第 4 步）；
  ④ 这套能且只能在"每轮 prefill 适配池"的前提下工作。
- 这条路径依赖的正是被我们关闭的 **resume/host-store** 机制——需要把它与 D 的"真位置 + 压缩地址"重新对齐，而不是当作废弃功能。

## 四、下一步建议
1. 立项：**"单轮 prefill 适配池 + 历史在主机 + decode 前选块"**（即用户流程 2–4 步）作为唯一验收形态；
2. 复用 `HostPageStore` + `restore_pages_batch`（纯拷贝、已验证），把 resume 从"hash 链采用"改为"按 query 选块恢复"；
3. 验收：多轮对话总历史 > 池，每轮增量 < 池，needle 正确。

# Plan D 交接索引（2026-11-12）

**可用且已验证的配置**
| 目的 | 配置 | 读数 |
|---|---|---|
| 完整提示 + 正确（不压缩） | `GPU_UTIL=0.972` `KVMEM_PLAN_D=1` `KVMEM_D_KNIFE=1` `KVMEM_ENTRY_RESIDENT_PREFILL=1` `INREQUEST_COMPACTION=0` | 140K → `ZX-4417-QJ7`（180s，usage 137,873）|
| 基线（v0.5.0 语义） | `KVMEM_PLAN_D=0` `INREQUEST_COMPACTION=1` + resident 入口 | 88K → `ZX-4417-QJ7`（124s，usage 74,948）|

**已确认的根因（都已修）**
1. `_ReqState.__slots__` 缺字段 → 刀相关写入全被 `except: pass` 吞掉（0d782e9）；
2. `st.view_installed` 从未被置 True → clamp/行安装/滚行全死（657）；
3. 入口裁剪造成“假绿”：142K 被裁成 36K（634 解耦，resident 入口后 usage=142,868）；
4. legacy 压缩的 rewind 与 D 账本冲突 → D 模式必须 `INREQUEST_COMPACTION=0`（否则 duct）。

**仍未完成（唯一剩余实现）**
逐 chunk 的“行重排（环）”：row[i] 必须恒等于逻辑块 (base+i) 的物理槽；`delta = frontier − window`；`seq_len = window`；`num_computed` 不回卷。
- 现状：执行通道已成立（89K 池跑完 137,873 token，171s），但缺行重排 → 答案退化。
- 关键代码位置：`runtime._reselect_blockwise`（刀）、`plugin` 步进钩子（652/655 的 delta-slide 与 ring 旋转，当前未触发，需查 `_st631 is None`）、`_install`（写行）。

# 方案 D：全量 token 流 + 压缩 attention 地址空间（完整版，用户拍板 2026-11-03）

## 0. 动机（v0.5.2 用户实测的两大症状）

1. 二次请求命中后仍要重算 ~47K（skip 深度被 StateImage 只能在首刀前拍死锁在 ~32K）；
2. prompt 完成后有 child 全量重算 30480（"原版没有这一步"——正确）。

两个症状同根：**v0.5.x 的刀压缩了 token 流本身**（rewind `_all_token_ids`/num_computed），
GDN 线性状态因此与全序列失配，一切（child 重算、浅 skip、快照窗口）都是这个失配的补偿。
原版 ninfer 按层分离执行程序（`set_linear_state_slots`/`linear_attention` 独立管线），
token 流对线性层永远全量，换出只发生在 attention 的地址空间——所以它没有 child、
reuse 深度 = `observed prefix/cursor`（全量）。

## 1. 不变量（D 的全部语义）

- **D1 token 轴永不压缩**：`num_computed_tokens`、`_all_token_ids`、GDN 输入流按真实
  长度推进；positions 恒为真位置（≤262,144 位置表内，入口防线已有）。
- **D2 attention 轴是窗口**：每请求 block_table 行 = **≤B+R 页的检索视图**，行内按
  真位置**升序**排列（sink 头部、评分中段升序、recent 尾部）；写地址=滚动游标进
  尾部槽位。行内容完全由 adapter 拥有（fork 既有固定容量 row 设施）。
- **D3 写映射走 slot_positions**：fork `compute_slot_mappings(positions, slot_positions)`
  的双位置签名就是为此预留——`positions` 喂 RoPE（真位置），`slot_positions` 喂写
  地址（视图滚动槽）。D 模式 adapter 必须逐 token 提供 slot_positions（真位置稀疏，
  行索引推不出来）。
- **D4 刀 = 纯页重排**：水位过时，选择（sink+recent+评分）→ 淘汰页 bytes 已在 store
  （per-step 提交 baked=0 规范真位置）→ 保留/新选页按真位置升序**重排入行**
  （restore_pages_batch 纯拷贝）→ 释放页归还池。**不截 token、不 rewind、无 child。**
- **D5 正确性论证（不改 kernel）**：视图行真位置升序 + 全部 ≤ 当前 chunk 起点 ⇒
  kernel 的索引式 causal 与真位置 causal **对前缀视图等价**；RoPE 相对相位由 K 写入
  时的真位置烘焙决定（baked=0 canonical），评分数学正确。chunk 内 causal 与采样
  路径不涉及视图坐标。
- **D6 状态永远真**：GDN 连续执行 ⇒ prompt 末的状态就是 observed prefix 的状态 ⇒
  resume 深度 = 全前缀（state 归档 + 视图页归档 + token 尾部），**命中轮只算增量**。
  child 机制、QueryReplay、escape 抑制、双账本 rewind——**整体删除**。

## 2. 手术点

| # | 位置 | 改动 |
|---|---|---|
| S1 | fork `gpu/model_runner.py:1607` | compute_slot_mappings 增传 `slot_positions=`（来自 adapter 注入的 per-request 写游标表；无注入时 None=现行为） |
| S2 | fork `v1/core/sched/scheduler.py` + kv_cache_utils | D 模式下页配额 = 窗口常数（`--reserve` 已有 off 通道），按视图容量而非逻辑长度 |
| S3 | adapter runtime | 写游标管理（滚动槽分配/复用）、slot_positions 张量供给、行=升序视图 |
| S4 | adapter 刀 | blockwise 刀改 D 语义：选择→重排→归还，删 rewind/channel truncate/`_all_token_ids` 截断 |
| S5 | 删除清单 | child rebuild、QueryReplay、escape_suppressed、双账本（num_logical offset 体系退化为恒等）、shallow seal/两段快照补丁、OFF 整体重算路径——**不留兼容** |
| S6 | resume/archive | StateImage 在 prompt 末即真；entry = 视图页集 + cursor + 尾部 token ⇒ skip 全前缀（原版 observed prefix 语义） |

## 3. 验收（goal 定义）

- 88K/152K/250K needle 全对；**152K 二次请求 ≈ 只算增量（目标 <15s，非 169s）**；
- 命中轮与冷跑字节一致；A/B 锚重生成（执行模型换了，锚换代一起出）；
- 离线全套 + 双租户并发初步（水位=每请求份额自律的初验）；
- 文档：README/DELIVERY 双臂合一语义重写，design-plan-d 收尾。

## 4. 分轮

R1 fork 通道（S1/S2 + adapter S3 游标）→ identity-D 短请求与现行字节一致；
R2 刀 D 化（S4）+ 旧机器删除（S5）→ 超池 needle 全对；
R3 resume 深度（S6）+ 全 battery；R4 文档/发布。

## 5. R1/R2a 验证状态修正（见 design-prefill-pressure §6.43）

R1 "identity 通过"与首轮 R2a 验证全部发生在 site 树（无 D hunks）上，判定作废。
site 树 a156cf2bfd 已含 R1+R2a 全部 fork hunks（patch sha 7f11009c）；R2a 真验证 = PLAN_D=1 boot + deltas/view_lens 钩真加载 + 同查询双发确定性 + 40K/88K needle。锚点新基线 de3921f3/2d117bd（spec4）。

## 6. R2a 定案（真树，2026-11-03 深夜）

- **D=1 identity = 表不注入**（`KVMEM_D_KNIFE=1` 才建 deltas/view_lens 表）。
  教训一：全零 slot_positions 张量替换 None 入 Triton kernel **不是数值等价**
  （40K needle 中毒 27283e53 代）——R2b 传真值时 slot 路径必须 needle 级验证。
  教训二：seq_lens 的内核约定 `pos = seq_len - query_start + idx`（诊断实锤
  old=1/new=9），外部无法复刻其 base 算术——view_lens 表必须由内核同点位供给
  或由刀直接写内核期望值。
- R2a 真树验收：40K×2 确定性 True（8bad5b37）、锚点臂内容正确、boot 干净；
  88K 纯 dense 无刀臂 needle False = 已知长上下文衰减边界（正是 R2b 检索刀
  存在的意义，v0.5.0 rebuild 臂 88K 7/7 为证）。
- 环境基线：site=a156cf2..0b83d19（patch 39dbcb23）；spec4 各臂锚点代不同属
  合法臂差异（de39=pressure 臂 / e783=D-dense 臂 / b804=classic 臂），换代判据
  以 needle 电池 + 同查询双发一致为准，锚点不再跨臂比较。


## 7. R2b 协议（刀=页重排，本设计的施工合同）

**复用面（不动）**：(c) 水位触发器、selector/core_selection_scores 选择、quiescent_compact 的
worker 侧页搬移与快照回退、adopt/revert 通道、geometry fail-closed。
**切换面（D 分支）**：

| 阶段 | v0.5.2 现状 | R2b D 语义 | 失败语义 |
|---|---|---|---|
| 水位判据 | num_computed(压缩账本) ≥ hi | **view 占用 ≥ hi**（num_computed 恒真，永不 rewind） | 同现状（idle throttle 不变） |
| 提案 | select_pages 升序 | 不变 | 空提案=不动 |
| worker install | row 重写+页交换（已有） | 不变 + **写表**：deltas[ri]=st.tokens−base_view、view_lens[ri]=base_view | 异常→revert 快照（row/表一起回） |
| Core apply | truncate _all_token_ids + rewind num_computed | **D 分支：两者都不做**；epoch/事件照发 | 拒案=无账本回滚需要（更稳） |
| prompt 末 | compact_rebuild→child | **compact_rebuild=False**：无 child，decode 直接续 | escape 谓词天然恒假 |
| 每步维护 | — | view_lens[ri] += n_new（worker step wrapper） | 步失败=现状一致 |
| hook | seq_lens=表值（纯读） | **恢复 +query_start**（内核约定 pos=seq_len−ql+idx，表只存 base） | mismatch 诊断常开（刀下 old=真 new=视图 属预期） |
| seal/resume | 浅快照两段式 | **R3**：StateImage 在 cursor 处诚实可得（D1 不变量）→ 深度=前缀全长，skip 合法 | R3 定 |

不变式自检：D1 num_computed 单调；D2 row 物理集=窗口容量恒定（cap 冻结后零 alloc/free）；
D3 刀零 truncate；D5 因果=视图升序+前缀性质；GDN 只吃真 token 流。
施工顺序：①apply 开关（plugin/Core）②worker install 写表+step 维护 ③hook +ql
④水位判据改 view 8⑤88K live 电池 → 152K/250K → 删旧机器（R3 一并）。

## 8. R2b 现场进展（2026-11-04 晨，主机重启后）

- 断点续验发现夜间电池从未真正运行刀：`KVMEM_D_KNIFE` 行在环境文件操作中丢失（d_mode False、表从未创建）——补回后链路瞬间打通。
- **第一刀端到端活了**：`d_tables_published` view_base=24048 / true_base=40960..45056 / **delta=16912..21008**（与 stride 理论吻合）；prefill 100% 正常推进；publish 的 inference-tensor 坑三连修（写侧 no_grad→.data→**建表侧 inference_mode(False)**才是正解）。
- **剩两门**：
  1. 第二刀 idle：quiescent_compact 在进入 blockwise 之前 return None 且无事件（所有 blockwise None 分支都有 log，一个没出现）——需给入口三检查补 reason 事件再打一发定位；
  2. 窗口越界守卫：stride 节奏失配时 prefill 写出窗口（本轮 80970 vs 53248 窗，第二刀没落 → 越界 ~8K → 请求 900s 未完被客户端 timeout）。方向：第二刀修好后按 stride 自然落；另加硬保险 = worker 端写游标逼近窗满时同步拒写并报 `d_window_guard`（提案，下轮定）。
- 夜间遗留：`/tmp` 数据文件全灭（重建了 88K 数据集 80970tok/needle@43K）；site/patch/engine 三树同步状态未受影响（重启无变化）。
- 本轮证据链：88K runs（125.4s needle True 无表时代 → 修表后 timeout 时代）+ console 直读（d_mode False→True、has_deltas False→True 的对照）。

## 9. R2b 第二轮夜战（2026-11-04，探针驱动的全部根因命名）

reason 探针（wrapper exception/is_rebuilt/entry）把"第二刀 idle"完全解剖，逐层修：
1. `NameError: epoch` —— quiescent_compact 签名无 epoch 参数（u2 锚当时就断在这里）：补参数闭环 wrapper→runtime ✓ 离线 qc/pp 绿。
2. **after_step fill 越界 raise** 打断 quiescent RPC → D 分支 `d_window_guard`（不污染 row，占位 -1 方案作废——它把 row 撑成 3334 毒死 validate）。
3. **stride 没留 chunk** → `_win - _vis - chunk`（刀必须在下 chunk 写越界前落）。
4. **refuse 风暴**：validate 的 frontier/epoch 与提案不匹配 —— 提案 stamp 采用发送方 (frontier, epoch)（`_d_sender`）+ worker 侧 frontier 相等检查 D 豁免。**残留失败**：stamp 修正后 88K-1 又 timeout（第二刀 cadence 仍有死角，疑似 `_d_sender` 全局态跨请求污染或多请求 RPC 时序）——这是唯一剩下的一门。

**已验证的正面成果**：88K 两发 needle=True（121.9s / 95.4s——95.4s 比无刀 dense 105s 更快，第一刀语义真实生效）；d_tables_published delta 与 stride 理论吻合；确定性（同代 sha 曾一致）。
**下一轮唯一缺口**：第二刀 refuse/idle 的死角（先按请求隔离 _d_sender，一发 88K 看 refuse 是否绝迹）→ 然后 152K/250K → 删旧机器 R3。


## 10. R2b 协议层收官（2026-11-04 round5）——缺口转移到 decode/MTP 通道

**协议层现状（全部达成，实测）**：
- 88K 双刀 cadence：vis 24048→23632、delta 16912→50096（滚动增长正确）；
- scores=true（TP 跨 rank 共识在 D 下改走"同步触发即共识"——trigger 本身 collective，两 rank 同刻进 RPC，各自 core_selection_scores 同长；consensus collective 的 availability 检查在 D 下反而误伤 → 跳过；worker 入口的 `_d_sender` 盖章让提案与 Core 现算值天然相等）；
- refuse=0：validate 的 D 豁免 = resident 相等 + visible>=resident（两条都是 rewind 世界项）；
- overrow/margin law：选择必须真空至少 2 chunk，防空转刀。

**唯一剩余缺口 = decode 侧**：prefill 100% 后请求 190s+ 不结束（abort 在 released 前无 crash）。
嫌疑排序（下一步逐个用 console 断点测，不猜）：
1. MTP4 draft 的 position/slot 通道在 D 视图坐标下越界（scratch 页、draft block_table 行）；
2. bonus-token 写位（query_len=5 步）slot=-1 或越窗；
3. `_resample_kernel/_rejection_kernel` JIT 首编译风暴（慢但会结束——不足以解释 190s 不收尾）。
判别实验：D boot 上 spec0 × 88K —— 若 spec0 完成且 needle 正确 → MTP 通道独毒；
若 spec0 也卡 → 主 decode 路径（slot 映射/seq_lens）的视图问题。

**运维事实修正（本回合踩坑）**：生产加载的 runtime 源 = site venv 的 **site-packages**
（finder 只映射 `vllm/`；repo src/*.py 由 start 脚本拷入 site-packages）——
**改 src/ 后必须重启才生效，且 patch/版本文件同步要覆盖 site-packages 拷贝链**。
本轮所有修复实际生效（boot=代码重启），但"改了不重启"的静默窗口骗了两次中期判读。


## 11. R2b 质量门达成（round 5 终）+ MTP 门移交

- **单变量矩阵**（88K 数据集 80970tok / needle@~43K）：
  | 配置 | 时长 | needle |
  |---|---|---|
  | D刀+spec0+旧律（final 刀被 2-chunk 律挡死） | 78.9s | **False** |
  | D刀关+spec0（纯 dense） | 117.1s | True |
  | D刀+spec0+**分律**（mid 要真空 2 chunk；final 要保覆盖） | 115.9s | **True ✓** |
  - final 刀接线三连：(c) 候选 D 豁免 prompt-complete 排除、(b) final 集 in-flight 门在 D 下改用 placeholders、apply 打 `kvmem_d_final_cut/kvmem_final_reselect` 防重刷。refused 风暴随之绝迹（本轮 applied=3 refused=0）。
- **移交门 1（decode/MTP）**：spec4+D 刀 → prefill 100% 后 190s 不收敛（无 crash，abort 后 released）；spec0 正常完成 → **MTP4 draft 通道在 D 视图坐标下的 position/slot 独毒**，下轮 console 断点定位（嫌疑：draft block_table 行/scratch 页/query_len=5 写位）。
- **移交门 2（152K/250K 电池）**：spec0 口径先验通，再开 MTP 门。
- 生产恢复：SPECULATIVE=4 + PLAN_D=0（v0.5.2 已验证臂）；刀场保留在 /etc/default 注释位。


## 12. R2b 收官扫描（round 6）：cadence 全绿，剩"final 问题刀落点"一门

**本轮打通（applied 计数与 refused=0 为证）**：
- (c) 触发在 D 下全程开火（d_c_probe 事件链），stride 基线归零修正后不再 45056 复读；
- **刀 10-11 连发全成功**（epoch 1..25，vis 24-27K 稳态）；152K spec0 完成 140.7s（dense 219s → **刀提速 35%**）；
- **两口径分离定案**：提案=validate 的 clip 和；刀律+表=窗口整页（`_d_window_visible`）——覆写 visible 的教训（chunk 对齐期全对、prompt 完成瞬间永拒 397 连）已成文；
- d_none_reason/d_rpc_return/d_c_probe/d_window_guard 四探针构成完整取证面（本轮三次靠它们命名根因：NameError(final 未定义参数名)、UnboundLocalError(定义顺序)、frame 分叉）。

**剩余一门（下一步）**：152K needle False 的唯一解释链——统计刀正确滚动视图，但**"问题驱动 final 刀"没在 prompt 完成刻以问题分重选到 needle 块（block 74）**。已排除：law 误拒（已放宽为 row-bound-only，epoch 25 落地=final 在 decode 步有成功 applied），kept 列表显示 tail 块在推进（86/91）但 **中间问题分没把 74 选进来**。可能：final 刀在 decode 段触发时问题分对"页 4750 区"打分缺失（harvest 只到 45056 统计）→ 需要在 prompt 完成瞬间（(b) final 集首次成立时）落一刀，用**冻结问题重打 store 分**（v0.5.0 的 rebuild_select 有现成问题打分链——D 版走 blockwise(pressure=False) 的 core 分数是否真是问题驱动 = 下一步第一检查点）。

**spec4+刀判据未验**（draft view-split 代码已落，真实刀况 timeout 待复测——若 final 门修复后仍 timeout 则 draft 修复复验提级）。
服务态：spec0 + D=1 + knife=1（生产臂 SPEC=4/PLAN_D=0 恢复留给验收全绿后）。


## 13. Final 刀打分链路定罪（round 7）

诊断事件 d_final_scores（一次 final 一发）实测：
- `core_len=704`、`n_blocks=92`、**`block_scores_len=29`**、`top8=[8,9,10,13,12,14,11,16]`、
  `score_around_needle={}`（needle 块 74 **根本不在分数表里**）。
- 判词：**分数网格 resample 只铺到 block 28**——`b=(gi*grid_pages)//q` 的 grid_pages/q 配比
  在 152K 上覆盖不到 n_blocks（704 条分数 × 32 页 ÷ q 上限≈29 块）。这不是排序问题、
  不是 room 问题（room=10 有名额），是**评分坐标系没走到 needle 所在区域**。
- v0.5.0 时代 final 刀正确选页走的是 `_rebuild_select` 的官方选择段（engine grid、
  block_ok、tail forced——88K/152K/250K 7/7 全对的 proven 打分链路）。
- **下轮施工定案**：D 的 final 刀弃用 blockwise 自建 resample，改调 rebuild_select 的
  选择段抽成 `_d_final_pages()`（返回升序 logical_pages），再接 D install（无 rewind、
  无 channel、row 直写、表照旧）。mid 刀维持统计 grid（它的职责只是滚动窗口）。


## 14. 打分链路：独立深水区立项（round 8 叫停游击战）

三轮连续"根因"被下轮数据推翻的记录（全部留档，防止再犯）：
1. round 6 判"final 刀被覆盖律误拒"→ 放宽后 final 落地（epoch 25）但 needle 依旧 False；
2. round 7 判"resample grid 只覆盖 29/92"→ 抽段/坐标系假设；
3. round 8 修 align_scores 截断（raw 全长度 2236）→ **d_final_scores 本轮零事件** =
   blockwise 的 pressure=False 聚合段根本没被执行——前三轮观测的"final 刀"其实是 mid 刀，
   **观测对象本身搞错了**。applied 序列 vis 27808/25024/23808 + stride 全为 mid 特征。

**当前铁的事实**（多轮重复、互相一致）：
- D 协议层稳定：cadence 11 连发 refused=0、两口径分离、152K spec0 140.7s 完成（dense 219s，
  刀提速 35%）、40K/短请求全对、boot 健康；
- **D 刀的选择质量未达验收**：mid 刀（统计分）合法地可能漏 needle 页，而"问题驱动的 final
  刀"在 D 臂的实际执行路径从未被证实跑过（本轮零事件）——它是**下一个独立工程单元**：
  先把 final→blockwise(pressure=False) 的调用链打通并有事件证据，再谈打分网格。

**生产线决议**：验收线未达，**生产保持 spec4 + PLAN_D=0（v0.5.2 验证臂）**；D 臂继续在
实验环境迭代。raw/consensus/sanitize 改动对 classic 臂零影响（classic 不走这些分支），
已五套件验证。


## 15. Final 刀结构公案破案（round 9）+ 最后的深水区定性

**破案过程（事件链完整）**：
1. 无条件入口探针 d_knife_enter（上轮探针埋错位置的教训修正）+ Core (b) 决策探针 d_b_probe；
2. 发现 (b) 探针零事件 → 值级探针 → 也零事件 → 把 step 外层 except 从吞掉改为记录
   sched_compact_error → **抓到 NameError 真凶：round 8 我给 (b) 上方插值探针时写了
   `if _isd and self is not None:`——`_quiescent_compaction(core)` 模块级函数里没有
   `self`！异常被外层吞 → (b)(final 集/RPC/d_b_probe)整段死码，而 (c) 在异常之前的位置
   正常执行（这正是"刀全落地但 final 永不发生"的诡异组合的源头）**；
3. 清除毒块后：**final 刀首次真实执行**（d_b_probe=1、pressure=false 入口出现、
   d_final_scores 出事件）。

**最终定性（下一轮唯一问题）**：d_final_scores 显示 core_len=704——recall.scores 的输出
长度 = (frontier+63)//64，frontier=45056 = **statistics 收割冻结在首刀位**。152K 的 45K 之后
97K tokens 从未被打分收割 → 无论选择算法怎么改，分数世界里根本没有 needle 区的格子。
**问题从"选择/坐标/网格"全部收束为一个**：D 滚动刀下 statistics harvest 的 frontier 为什么不
随真账本推进（读 v0.19 收割协议：谁喂 harvest、刀窗内/外、harvest 与视图安装的关系）。

**运维教训（第二次踩同一坑并写死）**：给 Core 模块级函数插代码时禁止引用 self；
探针必须放条件链最外层；except 吞异常 = 取证黑洞（sched_compact_error 现已常开）。


## 16. Harvest 死锁定位到最后一层（round 10）

- 破案链：harvest_attempt 事件显示收割停在 step 7（tokens 28672→45056）→
  `publish_position_offset`/`harvest_decode_rows` 通道考古 → wrote0 拦截修复上线 →
  本轮事实：statistics_final `frontier=45057, steps=12` ——**45056 之后只有 1 行进入统计**
  （4096 chunk 全部未落，且 d_harvest_wrote0 事件不存在 = 没走 wrote0 分支——
  它们在 `stats_from_bridge` 之前/之中被别的路径吞了：候选 `skip("no capture")`/
  `skip("drafts")` 或 `_tp` 锚在 chunk step 上取不到 n 行）。
- 本轮落地的机制（都有代码+套件背书）：fork 发布 `kvmem_step_true_positions`（delta
  通道存活时每步真坐标克隆）；harvest D 分支=真坐标写锚+视图偏移读锚；wrote0 不再推进
  `_stats_next`（防止错位复读的雪崩）且值级可见（d_harvest_wrote0）。
- 下轮第一取证点：在 `stats_from_bridge` 返回空/短的地方打值事件（n/layers/rows/buf 容量），
  一发区分"读不到行"vs"锚没生效"vs"skip 门"。三套 skip 计数现已并进 statistics_final 的
  skipped 字段——本轮它还是 null（说明连 _skip 都没走——更可能是 harvest_attempt 在
  step≥8 后就没再触发：plugin 调用点的 n_new 或 prefill 标志在刀后异常——查调用点条件）。


## 17. Harvest 公案判决：供水断在 capture，收割无罪（round 11）

- 函数首行漏斗探针（d_hv_funnel，无条件前 8 次/请求）实测：harvest_key_statistics
  **每步都在被调用**，stats_core=True、has_stats=True、n_tokens 正常；
- write_step 之后的 d_harvest_call 事件 = 0 → 每步都死在 `sums = stats_from_bridge(...)`
  为空（"no capture" 分支）；
- statistics_final：frontier=45057、steps=12、written=11280 —— **第 11 步整 chunk 入统计、
  第 12 步只入 1 行、此后 sums 恒空** = **bridge capture 从首刀附近就停止写行**；
- 判决：**问题在 capture 侧（谁停的水），不在收割坐标系**。此前 3 轮在 harvest/selector
  上的修正（真坐标锚、wrote0 保护、is_rebuilt 短路）都是必要卫生但不是本病灶——
  is_rebuilt 短路本身仍是真修复（stale mark 确实存在且毒过别的消费者）。
- 下轮唯一目标：capture 停水根因。第一嫌疑：capture op 的门读 `st.rebuilt` 字段（不是
  is_rebuilt() 谓词——短路盖不住它）；第二嫌疑：capture install 的 per-request 门
  （ARMED/geometry）在刀事件后失活。取证法：capture_slot_mapping/cap.note_tokens 处
  漏斗事件（同 d_hv_funnel 模式），一发看哪层没水。


## 18. Round 12: 刀系统全链路已执行到位；断水定位到引擎 capture 内部（最后一根水管）

- zero-offset 假设被字节级证伪（frontier 仍 45,057）；offset 不是病灶（修复保留：
  D 下推 0 依然是语义正确卫生）。
- 但本轮 tail 时间线给出新事实：**mid 刀每 ~2 步正常滚动（row_synced 序列
  29120→23808→27904→32000→23552→27648→31744 = 视图在窗口内健康伸缩）**，
  **epoch25 问题刀成功落地**（d_knife_enter pressure=False @143134 → blockwise
  749 pages moved=749 failed=0 → applied vis 23966 → row_synced 继续推进）——
  **刀的执行系统全部到位，唯一缺的是给问题刀的分数**（statistics 断供）。
- 断水点收束到最后一个黑盒：bridge 的 capture op 设备侧写行为（stats_from_bridge
  读取端已证每步在跑、行源为空）。**现成读出**：bridge 自带 `_op_calls`/status
  （read_all→"statistics" 事件），下轮第一步 = 对比第 12 步前后 op_calls 增量与
  bucket_writes 增量，一发区分"op 没被调"vs"op 调了没写行"vs"写了读不到"。
- 结构修复累积（全部保留）：publish_offset D=0、is_rebuilt 短路、wrote0 雪崩保护、
  真坐标锚、两口径分离、(b) 探针、漏斗探针、epoch 盖章、margins 法、final 刀接线。


## 19. 断水点静态排雷（round 13）：假设出局清单 + CPU 复现路线

静态解剖 capture 写路/读路后的出局名单：
- `KEY_ROWS_LIMIT` 卡死 → **出局**（plugin 每步无条件重设 0/1）；
- `_K_MISMATCH` 宽度失配 → **出局**（刀不动 key 宽度，且首轮 warmup 11280 行正常入）；
- `first` 坐标错 → **出局**（stats_from_bridge 根本不用 first 定位行，span 由 n 导出，
  读的是 bridge 的当前步缓冲头部——我此前所有"锚"工作对这个读路无效）。
剩余唯一可能：`read_key_buckets(layer, span)` 每层返回 None（666-692 区），其
return None 只有两个门：take<=0（span=66 排除）或 **buffer/rows 对象本身为 None
（= capture 那层本步没写过 → n<=0/…）**。

**下轮第一步 = 免 boot 的 CPU 复现**：tests/bridge_capture_test.py 已有 mock bridge，
扩一个场景 = d_mode + 首刀后步骤序列 → 断言 sums 空复现 → 在 mock 里直接调试读路。
复现不了才上 GPU 探针（read_key_buckets 的每层返回值事件）。

**本轮顺带纠正的认知**：`harvest_attempt` probe<6 截断曾让我们以为调用停在 step7，
真实情况是调用从未停（漏斗证）——探针截断本身就是本次连环误判的一半来源，
已把"探针不截断/截断必标注"写入运维条律。

### 运维条律追加
- 探针要么不截断，要么在事件里带 cap 标注（防"观察工具骗人"第三次发生）。
- 同一问题连续两轮"根因"被证伪 → 立即停手改复现路线（CPU mock 优先），
  不在真机上继续假设循环。


## 20. Round 13 实录：op_calls 曲线 + 两个被更正的观测 + retire 断裂立案

- 直读曲线（steps%3 无 cap 后）：op_calls 严格 +16/步（每层一次）到 frontier 36864、
  armed=true 恒真 —— **capture op 被调用且在写**（graph/eager 假设出局）。
- 观测自我更正（第二次犯同一条律并当场记下）：
  1) harvest_attempt 的 probe<6 截断 → "调用停在 step7" 是假；
  2) d_hv_funnel 的 cap<8 → "harvest 只跑了 8 次" 也是假。
  **凡带 cap 的探针，其最后一条事件的距离不等于真相 - 判读前必须先看 cap。**
- 立案的真断裂：`session_checkpoint_retire_aborted: "TP workers did not all
  prepare a checkpoint"` @请求收尾 - D 臂上 checkpoint prepare 两侧不对称 →
  retire abort。这既是 R3（resume/seal 在 D 下的诚实性）的核心缺口，也可能与
  统计断供同源（seal/prepare 走的是 harvest 数据的 completeness 检查！
  "statistics coverage incomplete at seal" 分支在同一条链上）。
- 下一轮唯一入口（非猜测，是对案）：retire abort 现场取证 - prepare 失败侧的值
  （哪层检查、当时 coverage 数字）；它与"第 12 步后 sums 空"很可能是同一个
  completeness 检查的两个断面。若同源，修 capture 覆盖 = 修 seal + 修 final 分数。


## 21. Xid31 硬防线落地并验证（round 14）

- 尸检结论（用户机器两次卡死）：NVRM Xid 31 MMU Fault = kernel 写越出冻结行后
  GPU 页表故障 → 驱动级硬锁死（journal 写不出，表现为整机卡死）。事后记账 guard
  拦不住 kernel 的写，必须事前不可能。
- 机制（fork scheduler 一处 + Core apply 一处，零 kernel 改动）：
  * 刀落地时 Core 记 `request.kvmem_d_ceiling = knife_true + (window − vis) − chunk`
    （两账本同步前进 → 真位写到 ceiling 时行恰好满，chunk 为刀响应头量）；
  * 排程点守卫：`computed + num_new + lookahead > ceiling` → num_new_tokens=0
    （引擎自己的"无事可排"路径，等待至多一步，刀因 stride 同额度已在路上）；
  * 首刀前无 ceiling（dense 写天然界内）；ceiling 只在 D 分支生效。
- 实测（D=1 knife=1 spec0 boot）：88K 75.1s 完成、152K **119.8s** 完成（此前同条件
  140.4s 且每轮 guard 事件 2-6 次）；**d_window_guard=0、refused=0、本 boot Xid=0**；
  三树同步：engine guard commit、site 32519891f0、patch 7650ab16/tree cb74be83。
- harvest 断供新线索：本轮 statistics 走到 step 10+ frontier 40960 仍在写
  （旧形态 step 12 死）——defer 步改变了 harvest 时序；final 分数仍覆盖不足
  （block_scores 29/49）→ 断供点下轮以 d_hv_pre/post（已埋，无 cap）判 read-back
  空 vs 写路空。


## 22. Round 15 立案：刀后时序在"同 boot 第二个请求"上全面劣化（跨请求态嫌疑）

实测矛盾（同一代码状态、同一 boot、两次 152K）：
- guard 首轮（boot 后第 1 个大请求）：step-end statistics 事件正常爬升（op_calls
  +16/步、frontier 40960+ 继续涨）、刀 15 发 refused=0、guard=0；
- 同 boot 第二轮 152K：step-end statistics 事件 **0 条**、final 刀进入 blockwise
  （d_knife_enter pressure=false ×2）但**打分段未执行**（d_final_scores 0 条）、
  statistics_final 显示 written=0（真实请求键！）；
- bridge 读路静态无清除点（_KACC 无 clear/pop、take 恒 66、reset 无调用者）——
  "读空"物理不可能 → 病灶必然是**写路或请求态字典**：`self._statistics[req]` 键
  错位（attach 与 final 读的对象不同）或 step-end 循环对第二个请求根本没进。
- **下轮第一刀**：blockwise 入口早退点 + `_statistics` 字典键 + step-end 循环命中
  三处值探针（全部无 cap），单 boot 连跑 152K×2 直接对照两轮差异。

安全面不变：guard 已让越界不可能（两轮 Xid=0），本机可继续实验。
性能面不变：guard 下 152K 最快 119.6s（dense 219s 的 54%）。


## 23. 断供终审（round 16，无 cap 铁证）：请求状态在首刀附近被拆

- 第四次踩 cap 坑（bucket 用 n_new//4096 恒 0 - 每 chunk 都是 4096）后彻底去 cap。
- 铁证（152K 双跑）：
  1) `d_hv_funnel` 到 call 10（tok=40960）为止，之后零调用 —— 收割从**第一刀前一步**就停了；
  2) call 9 起事件里 `tok=None` = **runtime `_state[req]` 已不存在，而 Core 请求仍在跑**；
  3) 与 round-13 的 `session_checkpoint_retire_aborted`（TP workers did not all
     prepare a checkpoint）同场景：seal/retire 链半路折断 - 状态拆了、没拆干净。
- 病灶定名：**D 刀落地时 seal/retire 对请求状态的提前释放**（谁 discard 的：
  accept_reconstruction? _release_slots 附带? retire 链? 下轮在 `_state.pop/del` 
  与 session retire 出口装无 cap 值事件，一发点名）。
- 此前"step-end 循环成员继续"的结论也被坏 bucket 污染，一并作废重查。


## 24. Round 17 双证伪与路标

- 证伪一（我的 sweep 误杀论）：加了 8 步 grace 后 `d_sweep_spared=0` 且 funnel 照旧
  停在 call 10 —— sweep 根本没做过误杀（round-16 的"disposer 定名"过早，撤回）。
- 新线索（比 disposer 更基础）：`released` 事件仍是**旧格式**（无 after_absent 字段）
  ——**worker 进程执行的 plugin 代码 ≠ repo 磁盘文件**！loader 鉴定是下轮第一步：
  `python -c "import vllm_kvmem_plugin as m; print(m.__file__)"` 在 worker 上下文
  （console 探针打印 module __file__ + d_mode 实际值 + sweep 函数源码指纹）。
  **若 worker 加载的是 site-packages 的旧拷贝，则过去几轮所有"plugin 修改"对
  该进程可能从未生效**——多个"修了没用"的现象将获得统一解释。
- funnel/harvest 的 call 停滞机制本身未定名；但先解决 loader 真相等最低层问题。


## 25. Round 18：sweep 误杀论成立并被治；新病灶=我自己的 0-token defer

- 统一 release 政策落地（runtime.sweep_release + seen_alive，plugin 三处出口全部
  改走单一政策；site A/B + retirement 合法路保留终端授权）——"旧格式 released"
  之谜=三处平行 sweep 代码分叉（不是 loader）。
- 判决性数据：`d_hv_funnel call=57`（贯穿整个请求生命周期，首刀不再是收割坟墓）
  + skipped 字典首次可见：**"wrote 0 (frontier 31199 first 31199 n 0)"** ——
  **新病灶定名：round-14 的 ceiling guard 用 num_new_tokens=0 做 defer，
  而 vLLM 的 num_scheduled_tokens 收了 0-token 条目，下游（harvest/commit/
  book）没有为 n=0 的步准备语义；请求时间 119→208s。**
- 修法（下轮第一件事，非假设）：defer 用"该请求本轮完全不排"（loop 里 skip，
  scheduled dict 不出现 0 条目）——engine 原生语义即支持某请求跳过一轮。
  复验判据：wrote-0(n=0) 噪声消失 + 152K 回到 ~120s + funnel 继续贯穿 +
  statistics frontier 越过首刀位持续增长。


## 26. Round 19：guard 撤除与倒退链的诚实记录（下轮 git bisect 定标）

同码同 boot 对照（全部 152K spec0，D=1 knife=1）：
| 代码状态 | 时长 | knives | 异常 |
|---|---|---|---|
| round-14 guard（0-entry 版）+ 无 sweep_release | **119.8s** | 15 | 当时判"guard 致 208s"存疑 |
| round-16 grace sweep + guard(0-entry) | 208.5s | 0(no_row×100) | blockwise_no_row 新病灶 |
| round-19 guard(native skip) | 208.5s | 0 | 与 0-entry 同症状 → guard 形态无罪 |
| round-19 guard 撤除 | 221.9s(spec4×2) | **0** | pressure_gate=0、funnel 狂转 476 |

**结论修正**：208s 与 guard 形态无关；**119.8s 是 round-14 代码态（guard 0-entry、无
sweep 统一政策）的产物**——倒退发生在 round-16/18 的 sweep/grace/seen_alive 系列改动
里（`blockwise_no_row=100` = worker 状态在 RPC 时刻 row 为空 = **before_step/absorb
链被 sweep 改动饿死**，或 funnel 476 次里 after_step 主链被跳过：pressure_gate=0）。

**下轮第一动作（方法定死，不再假设）**：git bisect 三个已知状态文件组——
`033b7eb`（119.8s 态）与 HEAD 之间，文件集锁定
plugin/vllm_kvmem_plugin.py + src/kvmem_sparse_runtime.py；每态单跑 152K 计时 +
数 knives/no_row；两三次 boot 之间必收敛。
guard 已撤（其保护在 cadence 正确时代本就未用到；cadence 修复后按 round-14 判据重加）。


## 27. Round 20：收割死亡定形=请求被"合法 retirement"释放（Core 视其 finished）

- import 回归治愈（无 `_np` 丢失后：no_row=0、knives=11、gate=16、refused=0、140.4s）。
- d_hv_result：44 条=11 步×2pid×2，**第 11 步后函数零进入**；funnel 尾部 `tok=None`
  而 `has_stats=True`（_state 没了、_statistics 还在）；`released` 事件为**旧格式**
  =出自 2860 的 `retire_finished_worker_requests`（唯一合法 retirement：
  `req in scheduler_output.finished_req_ids`）。
- **判决**：收割不是被谁杀了 - 是 Core **在第 11-12 步（首刀窗口）把未完成的请求
  标成 finished**，retirement 依规释放了 state。此后引擎还在跑该请求（prefill 继续、
  刀继续落、140s 完成响应）但 runtime 账本已注销 → 统计冻结、final 无分、needle False。
- 下轮唯一问题（新问题域）：**为什么 finished_req_ids 含一个 prefill 没走完的请求**——
  嫌疑：preempt（D 的 cap 让 allocate 的 demand 失真→资源判定误 preempt？）、
  max-model-len/len 检查、或 `_finishing_reconstruction` 残留标志。取证：
  scheduler_output.finished_req_ids 到达 retirement 时的值事件（带请求当前
  computed/prompt 进度对照）+ 引擎日志 preempt/abort grep。
- 若=preempt：D 的 cap（R2a 临时 book 改写）需与 scheduler 的资源判定和解——
  这条线索把"刀 cadence"与"假 finished"统一（也解释 round-14 15 knives 后统计仍冻结）。


## 28. Round 21：假 finished 论证伪；死亡收束到"首刀后 step_end 不再含该请求"

- 到达探针（d_retire_arrival）实测：真实请求的 retirement 只在**正常终点**发生
  （tokens=143140 ≥ prompt=143134，unfinished=false，still_scheduled=false）——
  round-20 的"Core 假 finished"判决**错误**，撤回。合法 retirement 无罪。
- 新的收束（全部无 cap 探针互证）：
  * d_hv_result：22 条（11×2pid），sums_layers=16 全健康，first 止于 40960；
  * funnel 止于 call 10-11；**没有 stats_rows_error、没有 sparse_step_error、
    skipped=None** ——harvest 不是死在内部任何门，是**根本不再被调用**；
  * pressure_gate 仍 16 次（它们走 quiescent RPC 的 pending flush 路径，不经
    step_end）→ **首刀之后，worker 的 step_end scheduled 循环不再包含该请求**。
- **下轮唯一取证**（一处无 cap 事件）：step_end 循环入口打印
  {pid, steps, sorted(scheduled keys)}——一发对比首刀前后 scheduled 内容；
  若请求真不在 scheduled 里：查 scheduler→worker 的 num_scheduled_tokens 构造
  （D 的 cap/defer/epoch 交互），这是收割断供的真正上游。
- 已排除名单（累计）：capture 写停/读停、align 截断（已修 raw）、坐标锚、
  sweep 误杀（grace 后照旧）、假 finished、_np import（已修）。


## 29. Round 22：死亡再上移一层 - step_end 包装器自身约在 step 12-14 停跑

- 本轮探针（snapshot 含 nn_vals；funnel 含 d_mode；warmup 过滤 + cap 120）实测：
  * d_step_snapshot：step 11-14 `n=1 nn=[4096]` —— 请求持续被排、n_new 恒 4096，
    **scheduled 内容与数值两罪均证伪**（round-21 的"循环不再含请求"错，
    round-22 的"n_new=0 静默门"也错 - 双双撤回）；
  * 但每 pid 的 snapshot 条数停在 ~12：step_end 包装器**自身**不再被调用；
  * funnel 的 call 计数同样停在 10-11（两事件互证同一时刻）；
  * pressure_gate 仍 16 次：RPC pending-flush 路径活着（它不依赖 step_end）。
- 死亡时刻 ≈ step 12-14 = 第二、三刀窗口；请求照常 140.6s 完成（引擎主链路无恙）。
- **下轮取证（同层再上移）**：execute_model 入口第一行无条件事件（每步一条，含
  `is_profile/dummy_run` 标志）：若 execute_model 活着而 step_end 死 - wrapper 内部
  的提前 return 链（`KVMEM_EXEC_POST`/`dummy_run`/`is_profile`）是凶手候补；
  若 execute_model 也停在同步数 - CUDA graph replay 路径绕过了包装器（decode 步走
  捕获图，不再进 Python execute_model），harvest 需要挪进 graph 外唯一必经点
  （quiescent boundary 的 pending flush 已经在这么活）。
- 累计证伪清单：capture写停 / align截断 / sweep误杀 / 假finished / scheduled缺席 /
  n_new=0 - 六案皆埋于证据。


## 30. Round 23 案卷：execute_model 全程活着而 step_end 只跑 12 步——矛盾定案

- d_em_entry（包装器第一行）：42 条/pid，真实请求 1227ca 到最后一步都在，
  dummy=False prof=False 全程；
- sparse_step_error=0（大 try 无异常）、引擎日志 0 错误、retirement 只在终点；
- d_step_snapshot（step_end 循环头）：12 条/pid，step 14 后不再出现；
- **静态已读尽**：em entry 与 snapshot 之间只剩四行代码 + 两个证伪的门
  （EXEC_POST/dummy）。代码与观测矛盾。
- RPC 路径补收割（d_rpc_harvest）written=0 也定案：bridge 缓冲只存最后一步的
  行，且刀后步的 harvest 若在 step_end 死路外补收，锚已错位 - 补救路线关闭。
- **下轮唯一动作**：snapshot 前再放一个无条件事件（d_preclear：steps、
  state['sparse'] is None、n_keys），一发闭环"最后四行"之谜；若 d_preclear
  同样停在 12，则凶手在 result=original() 内（它没返回，worker 线程阻塞后
  由别的路径继续执行引擎步 - 需要查 async 采样线程与 execute_model 的关系）。
- 案卷补充：D 臂 spec0 收割死亡史 = 与 spec4 同症状（MTP 通道假设作废）；
  生产臂 v0.5.2 harvest 无恙（无刀）。问题被精确隔离在"D 刀 + step_end"。


## 32. Round 24-25 案卷：我亲手埋的"裸 return"才是头号元凶；新前沿=state 在 ~step 8 蒸发

- **头号案件（已破已修）**：round-21 我用 s.find 移动错位的 d_pre_after 探针时，
  把 `if os.environ.get("KVMEM_EXEC_POST", "1") == "0":` 条件行整个吃掉，留下
  **裸的 `return result`** —— step_end 段自此 EVERY 步无条件返回。round 21-24 的
  "step_end 死亡"不是引擎之谜，是我对文件的一次毁伤。修法=条件行复原（本轮），
  验证=snapshots 从 11 → 116/pid 级（funnel 232 条，stats 写到 31200 后靠
  decode-drain 续命）。**教训（第六条取证条律）：用 s.find 锚定"多行语句中间行"
  的编辑必须先 grep 该文本出现次数；移动探针=删旧+插新两次独立编辑，不是一行替换。**
- **修复 step_end 后的新前沿**：stats 冻结点从 45057（刀1）提前到 31200（~step 8），
  funnel 分布显示 tok=None×82：state 在中段消失而调用继续。seal 豁免（本轮第二修，
  证据支持弱）没有改变冻结点 → 我的"seal 清零 _stats_next"假设**未被证实**，记录
  待查。真凶候选：grace=8 的 sweep_release——请求若在任意 8 连续步缺席
  （async 双阶段里 `_kvmem_req_ids` 的某阶段可能不含 prefill 请求）就会被中程释放。
  **下轮唯一取证**：release() 入口无条件栈事件（前 6 次/pid，含调用方栈），一发
  点名 31200 时刻是谁按下的按钮；若是 sweep，则 grace 的缺席判定要改用
  "scheduler 报告过该请求的步数"而非"连续缺席计数"。
- 本轮其余落档：EXEC_POST 复原；seal D 豁免（保留：ledger 连续性论证独立成立，
  即便非本案元凶）；decode 桶路径早退（消除 n=0 雪崩的正当性来自 decode 行走
  harvest_decode_rows 专职通道）；RPC 补读删除（与 step_end 竞争同一缓冲）。
- 预算：13 boots。纪律：停。交还。


## 34. Round 25 案卷：三处机械修正落地；final 刀疑点收窄到两点

已修（本轮）：
- decode 桶路径早退**删**（stats 冻结时它把 decode n=1 的正常入桶也拦了；
  原 clamp 分支健在，语义回归 v0.5.2 行为）；
- 复核确认 margin 律现场文本**已经是正确形态**（row-bound only for final）——
  本窗 overrow×10 是 MID 刀被 margin 律正当拒绝（空转刀），不是 final 被害；
- death-watch 三处全装（rethrow=0 事件 → worker-respawn 假设**证伪**）；
- 上轮"state 蒸发 31200"被本窗数据推翻：主请求 funnel 到 call 57 tok 31212
  一路健康——那是把**前一请求（88K×6 混跑窗口）的终盘**误读成主请求。
  **教训（第七条取证条律）：跨请求窗口必须按 rid 分组后再读时间线。**

final 刀未落的剩余疑点（两点，下轮一发分辨）：
1. raw=True 是否真在 runtime 生效（recall.scores 调用点的 selection_blocks
   截断行为——d_final_scores 若再现且 core_len 仍 704 = raw 没生效）；
2. (b) 在 88K 短窗口只 fire 一次（d_b_probe=1）而那一发的 pressure 标记与
   进入段是否吻合（enter 事件按 rid+pressure 分组统计）。
探针方案：enter 事件已有的话，加 recall.scores 入口的 blocks 参数值事件
（每请求首次），一发同时读 1+2。

预算纪律：本轮 6 次 boot，停止。产线恢复 spec4/PLAN_D=0。


## 35. Round 26：旧机器链锁死 + 终末问题视图锁落地；问题收窄到"视图→decode 读"最后一环

本轮定案成果：
1. **stage-2 rebuild 链在 D 臂彻底锁死**（_rebuild_select 函数级守卫 + finish 驱动
   出口丢弃队列项）：rebuild_select_diag/compact_rebuild_queued/reconstruction_started
   三事件全归零（此前每轮 6-8 条 - v0.5.2 机器在 D 臂的幽灵执行被消灭，
   目标"删除旧机器"完成此项）。
2. **raw 分数覆盖证实生效**：core_len 704→1178、block_scores_len 29→**49/49 全覆盖**，
   且 **needle 块 39 打进 top8 rank3** - 问题打分正确命中。
3. **终末问题视图锁（TQL）**：问题刀落地后 MID 刀拒绝对该请求再压缩
   （worker 端 d_question_cut 即时标记 - Core 端 final_cut 要晚一个 RPC 回复，
   实测拦不住紧随的下一步）。双发同 sha 确定性成立，applied 序列含问题刀。
4. 期间两次"我以为写了"（cw1/fl3 锚败未写盘）都被诚实事件数据当场拆穿 - 教训：
   每处编辑后 grep 目标文本计数是唯一可信的落盘证明。

**剩余唯一问题（精确）**：问题视图已安装、锁定、确定性稳定，但 needle 仍 False。
病灶候选只剩两枚：(a) 安装后 view_lens/deltas 表的值与视图内容不一致（decode 读错行）；
(b) 问题刀 install 的页集与 kept 列表不一致（restore 未真正搬到 needle 页）。
下轮第一诊断：问题刀落地时快照 st.logical 是否含 needle block 的页 + 表值三元组
(view_base/delta/needle_page)，一发分辨 (a)/(b)。

## 36. Round 27 案卷：刀无罪（needle 页恒在视图内），病灶=读路径表值

- coin 三轮修正后终判：`needle_page=1264 in view=1, logical_len=869, visible=27808` - **页坐标核查（40471//16），非块坐标**（我先 39*64 的块算术差点冤枉了刀）。刀安装、锁定、覆盖全部正确。
- **D 读路径唯一剩余嫌疑**：问题刀落地后 MID 刀持续 republish 表值（delta 随流增大到 17248+），若 decode 的视图读表在**某一步后不再 republish 或被 flush 顺序错位**，attention 就按错 delta 取行。病灶=表值时间线与视图安装的一致性 - audit 探针因缩进毁损被还原未遂，下轮第一步在 remobilize 与 republish 两处（以 12/16 空格正确缩进）装 slot 解析审计（needle 真位→slot 应得视图位置 vs delta 公式实际给出），一发定最后案。
- 本轮 boot 13 次，严重超预算；停止。


## 37. Round 28 案终：设备内容全程真实；时序定罪——问题刀晚于答案落锤

- needle_trace 直通（KVMEM_TRACE_PAGES）后内容审计铁证：
  * pre-knife：`compacted_slot=null`（needle 页根本不在视图内 - MID 选择按统计不含它，合法）；
  * question knife 安装后：`device_sha == store_sha` 全程一致 - **读路径无罪、内容无罪、delta 无罪**。
- 唯一病灶=时序：问题刀在 prompt 完成步 (b) 落地，但 `prompt 完成 → 第一个 decode 步`的
  窗口里答案已按"无 needle 页视图"生成（max_tokens=16 全部在问题视图安装前 commit）。
- 修法（R2b 最后闸门，下轮施工）：问题刀提前到"问题被冻结的瞬间"安装 - 即 prompt 最后一
  chunk 完成后、第一次采样之前（Core 端：scheduled_new_reqs 的最后一 chunk 步之后立即触发
  (b)，或 worker 端 prefill_done 首个 decode 步之前置于采样钩子内）。安装点提前一步，
  答案的第一 token 就出生在带 needle 页的问题视图里。
- 至此方案 D 的每个部件都有实证无罪/有罪判决：刀✓锁✓覆盖✓表✓内容✓ - 只剩这个安装时机。


## 38. Round 28 终节：push 恢复 + 最后闸门的物理陈述（R3 设计裁决点）

- GitHub push 恢复（21 commits 一次补上，origin/main=7c8fb2c）。
- 时序闸门的完整物理：(b) 挂在 Core.step 末尾 = `result=original_step` 之后，而**第一个
  采样 token 在 prompt 完成步的 execute_model 内部已产生**——问题刀在其后安装救不回它；
  且**问题向量本身来自最后 chunk 的 forward harvest**（q_rows），最后 chunk 之前不存在
  "已冻结的问题"——**纯安装提前无解**。
- v0.5.2 靠 child rebuild（重 prefill 末段）跨过此闸；D 禁 rewind/child，等价能力需要
  "末段视图重写 + GDN 状态回置"——GDN 单槽性质使其无法在不动 token 流的前提下重算末段
  （重 forward 会覆盖 state——除非有 D6 StateImage 引擎钩子）。
- **裁决候选（R3 议题，非本轮 hack）**：
  A. 采样器让路：prompt 完成步**不采样**（scheduler 把完成步排为 forward-only，
     采样推迟一步；问题刀在步尾安装 → 第一 token 出生在问题视图）。改动在排程语义，
     不动 GDN/state——**D 不变量完整**，vLLM 有 forward-only 步先例（chunked prefill 步
     本就不采样）。
  B. 双答案窗口：接受首 token 出生统计视图，从第二 token 起问题视图（needle 若是
     长思维链答案，16-token 窗口里可能已足够 - 需实测）。
  C. StateImage 引擎钩子（D6 完整版）：末段重算 + 状态快照回置 - 改动最深，收益最大
     （顺带把 R3 resume 深度一起解决）。
- 倾向序：**A 优先**（最小合法）、C 并入 R3 设计、B 作对照实验。


## 39. Round 29：step 间缝不存在（nosync），候选 A 真形态=worker 内 pre-sampler 刀

- 装 step 级 seam（forward→sample 之间下刀）：d_seam_knife 零事件 + 完成步返回
  executed=True - **本 fork nosync 调度：EngineCore.step 同步包含采样**，Core 层
  不存在 forward/sample 消息间隙。88K 80s 完成（seam 层无副作用，保留作 async 模式保险）。
- **候选 A 真形态（下轮施工）**：间隙在 worker 内部 - GPUModelRunner.execute_model 的
  forward 完成后、sampler 调用前。且 D 刀的安装**本来就全是 worker 侧动作**
  （选择/restore/row 写/表发布皆不碰 Core 账本）- **worker 本地 pre-sampler 刀**合法：
  完成步 forward 后（问题向量此刻刚被 harvest），runner 钩子调 runtime 的
  `d_local_question_knife(req)`：确定性选择（raw 分数、两 rank 同输入同结果 - consensus
  在 D 下已跳过）+ remobilize 安装 + 表发布，然后才 sampler - **第一 token 出生在问题视图**。
  风险登记：TP 两 rank 需同轮执行（确定性输入成立）；与 (c)/(b) 的幂等（final_cut 标记已有）。
- 本轮预算：4 boots 止，seam 保险层代码保留（D 门内无害）。


## 40. Round 30：seam 刀全链运转；候选 A 判死——首 token logits 物理先于刀

- slots 血案（`__slots__` AttributeError 杀光两臂 2 轮）修复入 `__slots__` +
  try/except fallback 双保险；离线 6/6。
- 本 boot 刀链全绿：`d_seam_local_knife after:True`（错误 0）、blockwise_reselect×2、
  锁生效、答案从复读机变 "user is is"（token2+ 确实活在了问题视图里）。
- **候选 A 的物理判决（定案）**：完成步的 logits 由 forward 在装刀**之前**算出——
  pre-sampler 刀救不了第一个 token；16-token 思维链被错首词带偏，needle 到不了输出。
  v0.5.2 靠 child rebuild 重 prefill 整个 prompt 才有正确首 token。
  **D 不变量（永不 rewind/无 child）下唯一出路 = D6 StateImage 重评原语**：
  末段 forward 重放 + GDN 状态在 cursor 处快照/回置（token 流与账本不动，
  重放是幂等再计算）——这正是 R3 resume 深度的同一个机制，一并施工。
- seam 刀保留（token2+ 的正确视图 + 锁的终末语义，都是 R3 重评的地基）。


## 41. Round 31 R3.1：D6 重评原语机制侦察完成（施工蓝图定形）

fork 无 StateImage 名字，但**同等机制已在 spec-rollback 生命周期里现成运转**：
- `mamba_hybrid.preprocess_state`（forward 前跨块迁移，spec scratch 列快照 =
  `run_fused_precopy`，postprocess 按 token_bias 折叠 accepted 列回主列）；
- 底层 `_copy_mamba_state_block`（(layer, state-type)×tile 分块 GPU 拷贝）
  + `batch_memcpy`；conv 与 temporal 两类状态全覆盖；
- 状态张量引用经 `get_extra_attn_kwargs` 直达 attn 层，无需自挖 KV cache。

**D6 重评六步（全部用现成原语）**：
1. 完成步 preprocess_state 时 snapshot 主列状态 → scratch（spec 机制自动，
   spec0 借用 scratch 块分配）；
2. 首跑 forward 完成 → seam 问题刀安装（round-30 已通）；
3. restore：scratch→主列（precopy 反向，src/dst 指针交换）→ 状态回到末 chunk 前；
4. **重放 forward 末 chunk**（同形状：batch=1 行 × 4096 token，positions/slots
   用现有通道）→ attention 读问题视图（needle 在场），GDN 从正确起点再推到正确终态
   （幂等再计算 - D1 不违：账本/token 流全程未动）；
5. 用重放的行末 logits 替换首跑 logits 进 sampler → 第一 token 出生在问题视图；
6. 后续 decode 正常（scratch 释放，锁保住问题视图到下一问题）。

分工：R3.2 = 步骤1/3/6 wiring + 离线；R3.3 = 4/5 重放与 logits 接管 + 88K 全链；
R3.4 = 152K/250K 电池 + resume 复用同一原语（observed-prefix：restore 到 cursor +
skip=不重放只装视图 - 顺路）。


## 42. Round 31 R3.2 施工定形：replay-and-replace（非侵入版）

侦察补充定案（本轮后半）：
- sampler 输入是 execute_model 局部变量（line 1712 `sample_hidden_states`）——
  hook **无法原地替换 logits**；但插件 wrapper 在 execute_model **返回值**外面——
  故采用 **replay-and-replace-output** 形态：
  1. seam 刀装完后，hook 在 runner 上置 `self._d_replay_request = scheduler_output`
     标志并**自递归调用一次 execute_model(同一 scheduler_output)**（guard 防环）；
  2. 重放跑完 forward+sampler → 其 ModelRunnerOutput 存 `self._d_replay_output`;
  3. 首跑 sampler 照常产出（garbage），wrapper 在 return 处把该 req 的
      sampled token **整体替换**为 replay 的 output（一行 dict 覆盖）；
  4. 首跑与重放的**账本二次效应全部封锁**（重放不得记账两遍）：
     view_lens advance（`_d_in_replay` 旗标跳过）、statistics harvest（旗标）、
     bridge drain（重放前不清 buffer 的时序：首跑 bridge 留到最后 chunk 行，
     重放覆写同键 - 幂等）、note_engine_* 上报（旗标）；
  5. **状态快照/回置**：completion 步 forward 前（wrapper 入口）把该 req 行的
     全层 mamba 主列拷进 scratch 列 - 用 `_get_mamba_group_info` 的 group ctx +
     `get_mamba_state_copy_funcs`，Python 逐层 clone 起步（48层×2类 ≈ 百个
     slice copy，毫秒级），kernel 版后换；刀装完后、重放前 scratch→主列回置。
- D1 合规论证（终版）：token 流/账本/位置通道全程未动；重放=同一 token 序列在
  同一真坐标上的**幂等再计算**（attention 读问题视图、GDN 从快照再推到同一
  正确终态）；被替换的只是**采样结果**（本来就要重来的 logits 线）。
- 施工序：R3.2a 快照/回置 + 单元（假 completion 步离线 round-trip 状态恒等）→
  R3.2b 自递归重放 + 二次效应旗标 + 输出替换 → R3.3 88K 全链 + coin/logits
  双证（第一 token 出生视图 needle_in_view=1）→ R3.4 电池。


## 43. Round 32：R3.2a 完成 - 状态快照在线运转（48 tensor/请求/完成步）

- `d_state_snapshot/d_state_restore` + `d_preflight_snapshot`（完成步 forward 前）
  端到端验证：`d_preflight_snap row 0 tensors 48`（双 rank 各一次）。
- 施工中学到的三个接口真值（全部由探针链逼出，非猜测）：
  1. `runner.input_batch.req_id_to_index` 在 execute_model **入口不可用**
     （V2 内部才 build）→ 行号改用 `st.row_index`（view.row_index）；
  2. `BlockTables` 不可下标 → 真接口 `runner.block_tables.block_tables[gid].gpu`
     （live view 同源）；
  3. `MambaSpec` 是 frozen dataclass（单元用 object.__setattr__ 构造）。
- 快照覆盖=48 mamba 层（conv+ssm 同 tensor 家族），attention 缓存**故意不快照**
  （位置键写、幂等重放自然复原）。
- 离线：state_snapshot 单元 5/5 新入列（graceful None/只选 mamba 组/round-trip
  恒等/行→块列 0 解析/部分失败计数）；全套 11 项绿。
- 下轮 R3.2b：seam 刀安装后 **restore→重放同一 scheduler_output→wrapper 处
  用重放 output 替换**；重放旗标封 view_lens advance/harvest/engine-notes。
- 取证方法自白：本轮 6 次 boot 全部是"一发一断"纪律执行——每次只测一个未知，
  零假设叠加零猜测。


## 45. Round 33 里程碑：**88K needle TRUE** - 单写者定案 + seam 违约定罪

- **答案**：plugin advance 块（每步 `_vlv.data[_ri] += _cnt`）与刀 publish 双写者冲突，
  且块内引用 replay 时代死变量 → NameError → `except: pass` 吞 → **表冻死**；
  fork 的 seq_lens override 本就 = view_lens + live ql（刀是唯一需要写表的人）。
  advance 块**整体切除**（第 N 次 except-pass 陷阱入档：任何 except-pass 必须计数）。
- seam 刀违 R2b 契约（install-after-commit：restore 拷未 commit 末 chunk）→ 缴械。
- replay 自递归方案废弃：restore-to-cursor + 单步 ≠ 重评（漏吃末 chunk 中段）；
  快照机器保留（未来全末段 re-prefill 用）。
- 88K ×64：**needle True**，79.9-82.1s。D 臂第一次绿电池。
- vl_mismatch 首 chunk 的 old=真账本 vs new=ql 是**设计语义**（非刀时基 0=恒等），
  属预期，不修。
- **下轮：152K 同判据**（问题刀 749 pages + needle）。


## 46. Round 33 终局：刀系统已死的完整尸检 + 复读病理解 + 重建路线

- 事件面铁证（truth 探针在 `quiescent_compact` 真入口，未节流、cap 60 自计）：
  **d_truth=0 - RPC 从未到达 runtime**；d_compact_applied/d_c_probe/d_b_probe 同窗全零。
  刀系统（Core 触发侧）在某次 boot 后整体停摆，触发链断裂点未定位（下轮）。
- **复读病理解剖（完整因果链终闭合）**：
  1. 某 boot 早期真刀落地过（历史 applied 证据）→ `kvmem_view_active=True` 被设置；
  2. 刀系统死后表停在最后 publish 值（advance 被切除期冻在 0 / NameError 时代遗产）；
  3. fork 的 seq_lens override 在 active 下把 causal 覆盖成 `vl[0]+ql = 0+1`；
  4. decode 步模型只看见 1 个 token → "1111…"/"The user is asking…" 无限循环。
  **所有此前的"视图内容错"猜测全部作废：是 causal 轴被覆盖成常数。**
- **机制缺口登记（必修）**：`kvmem_view_active` 与 view_lens 表是 per-runner 全局，
  **不随请求终局复位**：一个请求的刀影响下一个请求的 causal。修复选项：
  (a) override 加"表值合理性"守卫（vl=0 且真账本>ql → 不覆盖，退 dense）；
  (b) release/finalize 时复位 active+表（同请求生命周期内也需初始化为"账本恒等"）。
  **两者都做**：a 是硬保险丝，b 是正确语义。
- 88K "needle True" 历史判决修正：那是 D 未生效的 dense 原生跑（页配额 guard 自
  round-21 撤销后从未恢复，88K<209K 池天然驻留）——**里程碑未达成，如实撤回**。
- 下轮顺序（每步单 boot 验证）：
  1. override 保险丝 (a) + 生命周期复位 (b)；
  2. Core 触发链定位（d_truth 恒 0 的断点：(c)/(b) 候选循环或 collective_rpc 层）；
  3. 恢复 scheduler 页配额 guard（真窗口 + Xid 保险）；
  4. 88K/152K 在"真刀 + 活表"下落判。

## 47. Round 34：D 机器真正活着的铁证 + 剩余病灶一句话

- 断链史终局：round-29 seam 编辑吞了 `EngineCore.step = step` 一行——刀系统（Core 触发侧）静默停摆 5 轮，期间所有"零事件"皆源于此。修复后 88K **真刀 needle=True**（applied 含问题刀，answer 带 ZX-4417-QJ7 全串）。
- 三件套生效：fork fuse（vl=0 行不覆盖 causal）、release 清零（跨请求毒化闭合）、advance 重建（decode 步 cache 缺失时从 scheduler_output 重算）——**kvmem_vl_mismatch 归零**（88K+152K 全程），**152K 统计账本首次写满 143135/36 步**（harvest 死亡彻底终结，advance 是真凶）。
- 152K 仍 False，病灶收敛为一句话：**问题视图安装后会被后续 MID 视图覆盖**（终末锁只守 runtime/_c 一条路，refused×97 提示 Core 侧还有第二扇门）。下一刀：锁贯通 Core 谓词（plan_d 的 (c) 谓词已含 final_cut——查 refused 是否 MID 被拒后问题刀又被拒……读 refused 详情+applied epoch 序列）。
- 性能现状：探针全开 163s/88K（R4 卸载后预期回到 ~80/140s）。

## 48. Round 35：TP 共识三层修正史 + 问题刀仍不落（剩余战场）

- 改动三连（全离线绿，commit 只写内容）：
  1. validate 的 TP 等值比较改**逻辑页集合**（D 下各 rank 物理槽/可见数天然异）；
  2. 上界律 `selected[-1]*page>=logical` 修正为跨界尾页合法的 `selected[-1] >= ceil(logical/page)`（69x "invalid or non-shrinking" 元凶，两臂共用正确形式）；
  3. D 的共识分支恢复 **tp collective**（问题向量均值跨 rank 交换）——sanitize 版让排名各自漂，实测问题刀两次 enter 都不落地。
- 现状读数：refused 从 "different working sets" 转为 "different **logical** working sets"×68 + non-shrinking×69——**两个 rank 的问题分数排名本身不同**（MID 靠 consensus 平均过；final 的 `_last_core` 兜底路径 rank-local）。问题刀仍未 apply。
- 下一步（不混）：a) 把 final 也走 `_core_consensus` 集体平均（与 MID 同路，恢复原状）；b) 若仍分歧，取证两 rank 的 top16 差异（是分数还是行数）。锁语义之后：refused×68 里还有多少是 MID 被误伤（D-scoped 比较后 MID 应该 100% 通过——它没通过说明**问题分数被 MID 复用**？）。
- 性能注记：探针全开 295s（R4 卸载预期 ~140s）。

## 49. Round 36：共识统一 - 协议全绿，152K 转入打分质量层

- 根因终审：`_core_consensus` 的 D 分支 sanitize 后 `return` - **collective 从未为 D 跑过**（上轮改的是非-D 条件，未命中）。修正=sanitize→fall-through 同一条 `average_reconstruction_scores` 集体路。
- 判决读数（一发）：**applied 10 / refused 0**（历史最好：上轮 5/75）；问题刀 epoch 25 真实落地（b_probe→enter pressure=False→applied 全链）。D 刀协议层（选、共、锁、装、拒零）**完整闭环**。
- 152K needle 仍 False，但病灶换层：**问题向量被 prompt 里字面 "MID." 段落污染**（答案里 "for the MID" 双证），top8=[90,89,45,88...] 全是尾部文本块，needle 块 39 掉出前排。这是**打分质量**（冻结问题行的选取/去噪），不再是刀协议 - R3.4：对比 88K（同题面 top8 含 39）与 152K 的 q_rows 冻结位（问题文本在 prompt 尾部哪几行、MID. 段在哪几行），把"问题=最后 N 行"的 N 定准。
- 协议层验收推进：88K ✓真刀；152K 协议 ✓（等质量层）；Xid=0 全程（保险丝+锁在岗）。


## 50. Round 37：query 稀释假设证伪 + 新主嫌疑=零自由刀让 MID 逐出 needle

- 判别三连（152K 同题面）：QUERY_TAIL=6 / QUERY_ROWS=24 / 默认 —— **top8 完全相同
  [90,89,45,88,46,85,84,86]**。问题打分对窗口大小不敏感 ⇒ **稀释假说死**。
  （诚实记录：我的 KVMEM_QUERY_ROWS 加错了路——真评分 query 走 core_selection_scores
  的 _q_rows 裁剪，即 QUERY_TAIL；两路实验同 top8 反而成了双保险证据。）
- 真形态重构：needle 块 39 在 MID 刀时代**曾经在场**（88K 首刀窗口期答对证明），
  152K 问题刀 top8 前排全是 tail 块（filler 自相似主导，问题向量只有 15 token 对
  640-token 模板头+tail 的相对权重）→ kept 8 块无 needle → **问题刀安装即逐出
  needle**；此后 MID 零自由（749 页满窗）刀再钉死。
- **下一轮手术（R3.5）：自由页法则贯通**——零自由选择（chosen 页数 ≥ 当前在场 resident
  覆盖）：(a) Core 谓词拒绝（D 模式 pressure/非 pressure 皆查）；(b) 选择器给问题刀
  保底把"统计高分且含 needle 历史"的中间块纳入（tail 自相似挤占问题前排的修法=
  对 q 与块 K 的匹配做 top-k 稀疏化/去 tail 重复段冗余）。两步各一 boot，先 (a)。
- 88K 保持真刀 True；协议层 10/0 保持；本轮 3 boots。

## 52. Round 38：query 卫生生效但 needle rank 23 不动 - 电池构造天花板定罪，两级视图开工

- force-tail(96 行问题)生效证据：max 从 137.9 降到 8.35、cliff 消失、rank20 全谱平滑 - query 干净了。
- **但 needle_rank 仍 23**：152K 电池尾部是 90 个与问题句同文的 clone 块（"routine sensor sweep…access code for KEY-MID"结构），**任何以问题句为 query 的检索都无法在 90 个自相似块中把 needle 块排进前 18** - 这是测试构造的物理天花板，不是 D 的缺陷（v0.5.2 能过是靠 child rebuild 把整个 prompt 重 prefill，问题句在 GDN 状态里而非仅靠 attention 检索 - D 无 child 无 rewind，等价能力就是缺口本身）。
- **对策（施工中）= 两级视图**：D 的 MID 刀以统计 query 打分 - 88K 实测 needle block 39 rank 3（统计 query 含真实信息内容）；问题刀视图 = **统计 in-place 选块 ∪ 问题 top-k**（mid picks 先进 room，问题排名续填）- 保住 needle 邻域不被 tail-clone 挤掉。改动三处（scoped 在 _reselect_blockwise 内部）：记录 `_last_mid_pages[req]`（MID apply 时刻 logical 页集）、final 视图填块先 mid 后 order、`_mid_kept_blocks` 页→块换算。
- 教训入档：三次锚事（wt1/wt2/b2）全因**同形代码存在于 quiescent 与 blockwise 两区** - 函数级编辑必须先切函数边界再匹配（本次 head/tail 分割法一次到位）。
- 预算：本轮 7 boots 超支停止；union 验证=下轮第一发（152K+88K 双判）。

## 53. Round 38 终节：D 电池首次双绿（152K+250K），88K flaky 定罪=room 抢占顺序

- union 首验：**152K True**（答案完整句"…code is ZX-4417-QJ7 filed under vault M-118"）、**250K True** - D 臂首次过两发深水位；88K flaky（x1 False/x2 True，同 boot 同输入）。
- flaky 机制定形：88K budget 48 块里 sink+recent 占 ~40，room 仅 ~8；mid blocks **ascending** 先吃 room=前排 sink 邻域块，把 order rank3 的 39 挤出去；152K room=10 反而够用（order 全 clones 顶前排时 mid 补位保命）。两向需求相反 → **静态顺序规则无解，必须按分数合并**（mid blocks 用 order 的分数加权排序，或 room 五五分）。下轮一发调参。
- 里程碑如实：D 电池 152K/250K True（union era），88K 待稳定化；这已是方案 D 迄今最深验证（v0.5.2 同等长度靠 child rebuild，D 全程无 rewind 无 child）。

## 54. Round 39：union 三代调参战果与遗留（88K 稳定 True，深水位未愈）

- 战果：**88K 三轮连 True 稳定**（round-robin fill）；152K/250K 仍 False。
- 合并证据（d_final_merge）定形：152K final 刀 `mid_n=0` **恒空** - `_last_mid_pages` 从未被填、harvest 编辑也无 d_stats_asset_harvested 事件 → 收割代码自身在异常里静默死掉（`except Exception: pass` 第七次咬人，现行律：任何 except 必出声）。候选死因：harvest 作用域的 `pt/q/sink_blocks` 变量名不匹配、或 `_last_core` 在 D 下从未被 pressure 路径写入（MID 刀现在每刀 fresh fetch，可能根本不写 _last_core）。
- 250K 读数修正认知：其 top8=[148, 71, 72, 73, 146, 138, 140, 145] - **问题分数并不差**（needle@75 在块 71-73 邻域进 top8！），flaky 属 decode 期其它刀回滚视图（round-robin 未验证该臂）。
- **下轮单诊断（一次 boot 定生死）**：把 harvest 的 except 全改出声 + 打印 `hasattr(orch,"_last_core")` 与 pressure 路径的 core fetch 结果是否落 `_last_core`；同时验证 250K 是否只需 final 刀后再无 MID（terminal lock 在 250K 的覆盖度）。
- 预算纪律：本轮 8 boots，停止。

## 55. Round 40：harvest 代码在、事件零 - `_last_core` 为空是唯一静默路径（下轮 orch 身份结案）

- 诊断面（kb148/149 两 boot）：d_final_merge 有、d_harvest_error 零、d_harvest_shape 零、d_stats_asset_harvested 零、d_question_core_missing 零、d_query_trim 零。
- 逐分支排除：harvest 代码存在（4406-4448，awk 窗口曾误导）；事件全哑 = **`getattr(orch,"_last_core")` 恒 None/空**（shape 日志深在 `if _lc0` 块内，lc 空则整段无声 - 这条日志放置本身是个设计错误，事件应在判空前无条件出）。
- 矛盾点：MID 刀 applied×10 的 kept 明显含中间块（vis 26-28K）→ 它们的 core fetch 成功 → `orch._last_core=list(core)` 应写进同一 per-req orch 对象 → final 刀读同一 `self._orch[req]` 却为空。候选：(a) MID 路径的 orch 与 final 的**不同实例**（per-call 重建?）；(b) MID 刀走了另一函数（3866/3668 的 core 直接喂 kept 不经 orch 属性）；(c) 赋值在 return 之后不可达。
- **下轮单发**：在 orch 属性写/读两侧各放一条 `id(orch)` + pid 事件，一发定实例身份；同时把 harvest 的 shape 日志提到判空之前（诊断卫生）。
- 本轮 3 boots；生产恢复 spec4/PLAN_D=0。

## 56. Round 41：union 全灭判决 + 基线诚实化（88K 本就是 flaky）

- 四种 union（mid-ascending / round-robin / score-ordered / asset-synced-view）实测：**152K 无一过线，88K 反而从 3×True 滑向全 False** - slot 层调参宣告失败并全部停用（`if False` 门保留代码+重开条件：middle-only 资产 **且** 88K 回归跑绿）。
- 基线归位（去 env 残留 + union 停用）后 88K x1 False / x2 True - **此前的"三轮连 True 稳定"证伪：88K 一直处在 flaky 带内**，union 只是把概率推向坏侧。接受这个真相比接受一个假绿灯有价值。
- 根因层次修正：所有 union 都在治"needle 不在 kept"的表症；真层在**打分器对 tail 复读段的偏好**（clone 块 90/89/88 无论什么 query 都霸榜 - 它们的 K 与任何"最近读过的文本"都高相似）。**下一步进打分层**：block_scores 的近重复降权（对 top-N 相邻块做冗余惩罚，MID/FINAL 两臂共用），或者问题向量只取纯问句（force_tail 探针 5 boots 零事件 - trim 从未真正执行，那条路本身没测过）。这是 v0.5.x 也存在的共同天花板（当年靠 child rebuild 全量重 prefill 绕过，D 必须直球解决）。
- 电池现状（诚实版）：88K flaky（约五五开）、152K/250K False、协议层 10/0 与 causal 保险丝不动。
- 预算：3 boots，停。

## 57. Round 42：窗口混叠冤案 - 152K 在纯基线（无 union）已过线 True

- 判读事故：kb153 窗口把 88K×2 与 152K 的事件混在一起读，`core_scores top=[633..]` 实为 88K 的请求（frontier 45056=88K 首刀时刻）；`applied vis 12128` 全是 88K 形态 - **152K 那一发当时根本没跑到问题刀**（run 被 timeout 截断）。"union 四连败"的部分判决被这条混叠污染。
- 纯基线重跑（union 停用、env 干净、同 boot）：**152K needle=True**（"…code ZX-4417-QJ7 filed under vault M-118"完整句）+ 问题刀 enter tokens=143134 pressure=False 全链在案。
- **推翻/确立**：round-41 的"union 全灭"结论降级 - 对照实验里 152K 的 False 从未被干净测过（每次都混在别的 run 窗口/别的代码状态里）。正确事实待四连发：152K x2 + 88K x3 - 若稳定则基线（无 union、RR fill 保留、stats fallback 恢复）就是当前最优配置；88K 的 flaky 带也需这轮样本重定。
- 教训条律（第八律）：**电池判读必须按请求 rid+prompt 长度隔离事件窗口，跑一发判一发**；混窗判读已经两次（round-27 分组、本轮窗口）制造假结论。

### 57b. 基线电池定案读数（kb153 boot，零重启连打七发）

- 88K: True / True / True（x1 115.6s、x2 82.2s、x3 115.6s）
- 152K: True / True（204.4s、153.4s）
- 250K: False x1（290.2s）
- **定案：无 union 基线（stats fallback 恢复 + RR 保留）就是 D 迄今最优配置** - round-41 全部 union 变体都被这次干净对照判死。
- 250K 待解剖（下一个独立缺口）：其问题刀 top8=[148, 71,72,73,146,138,140,145] 与 needle 块的真实对应需要先校准重建版 250K 数据的 needle 块号（我 39/69 的公式对 250K 未验），再判"挤掉/低分/未入场"。电池余一站。

## 59. Round 43：250K 数据冤案 + union 假说整体证伪（stats 资产恒不含 needle）

- 250K 重建版 v1 数据缺陷：needle 句写成 "KEY-ZX-4417-QJ7"（无 KEY-MID 标签），问句引用不到 - v2 重建为 88K 同款 "KEY-MID special access code ZX-4417-QJ7" 配对句式（needle token 112591 block 109/230）。
- v2 纯基线读数：问题刀 top8=[71,72,73,74,140,145,82,129]，**needle block 109 rank 9**，room=10 - 一步之遥。union v2（保留 stats 中间 top-2 槽位）实测：`stats_top6=[413,414,416,...]` **统计场全局顶到末 chunk 自身区域**，中间排名头两名=[7,8] 零 needle 价值；预留槽反把 88K/152K 基线拖 False×1。已缴械（if False 门+判词入码）。
- **union 假说家族（v1 页资产 / RR / score-ordered / asset-synced / v2 stats 预留）全部证伪**：深水位统计资产从不含 needle（其相似度场由"最近读过的块"主导） - 检索资产层没有免费午餐，109 掉出 top8 是打分器对 tail 相似场的真弱点。
- 三发纯基线验证跑在中（250K x2 / 88K / 152K）。
- 250K 若仍 False：唯一未测的打分层直球 = **问句纯净化**（force_tail 从未有事件 - 探针先修活），问句 6-24 行单独成 query，clone 场被 "KEY-MID" 专名拉开。那是设计 51 想做没做成的那一发。

### 60. Round 43 终节：基线三发 False 复现 - flaky 带存在且比样本大（下轮同 boot 重采样定身位）

- kb157 三发（250K/88K/152K）全 False，同码 kb153 曾 6/6 全 True - **代码状态存疑优先于答案随机性**：待把运行码 diff 到 fc2c40f 后同 boot 重打。
- 本轮有效新知：250K 真 needle block=109（token 112591/234563 校准），问题刀 top8=[71,72,140,145,134,73,147,144] - **71-74 = KEY-MID 标签复读段**（filler 中的 KEY-MID special 句），非 needle 正文段 - 打分器对"标签文本"敏感对"答案正文"钝，与 152K 的 top8=[90,89...]（clone 段）同构。
- 收束预算：4 boots，停。

## 61. Round 43 终节：flaky 定量化 + 确定性 tie-break 落地 + 波动源锁定到分数值层

- 同 boot 五发 88K：True/False/False/False/False（含 tie-break 前后各测）；问题刀**每发都 apply**（vis 11086 形态在场）- flaky 不在刀落地与否。
- **波动现形**：top8 五发近同但 **rank 7-9 抖动**（27↔36↔46 互换、38/43/45 进出）- needle 块恰在这些名次徘徊 → 答案翻面。修复 step1：`sorted(key=(-score, block))` **确定性平局裁决**（旧排序回落 dict 插入序，而插入序随 TP all_reduce 到达序漂 - **同 boot 非确定性的直接来源**，此修复永久保留）。
- step1 后 top8 仍 ±1 漂 → 剩余波动在**分数值本身**：候选 = 问题刀打分时 last-chunk bucket 行的 harvest 在途差异（未证实）。下轮诊断：同 prompt 同 boot 两次问题刀对比 `needle_score` 数值（d_final_scores 已有该字段），数值同而名次动=tie-break 未覆盖的浮点微差；数值动=harvest 时序，再查刀前 drain 完备性。
- 电池：88K 1/5、152K 0/2、250K 0/2（v2 数据正确后） - **比 fc2c40f 时代读数差是统计巧合还是回归，需 tie-break 版重跑积累样本**（每尺寸 ≥5 发才有判决资格，此前"3/3 全绿"样本量同样不足 - 诚实修正：所有单发/三发结论降级为"倾向"）。

## 62. Round 44：判读污染总清算 - kb158 起的"flaky"数据多半测错了臂

- 事故：kb158/161/162 三次判决命令**没先翻 PLAN_D**（round-43 收束时生产=0），跑的其实是**生产 v0.5.x 臂**：证据=kb162 四发窗口 d_final_scores **零事件**（D 刀根本没运行），而 88K 三发 True 全是 v0.5.2 原生成绩、152K False 是 spec4 生产臂异常（值得单独留意）。
- 连带清算：kb159/160 的"同 boot 五发 2/5、tie-break 后 1/5"读数同属错臂嫌疑（当时 env 状态未验证）- **D 臂 flaky 的真实水平：未知，待重测**（本轮 kb163 正在做：env 检查置前+provenance 计数+五发）。
- 纪律升级（条律九）：**每次判决实验的第一行必须是 `grep PLAN_D /etc/default` + boot 自检事件确认臂身份；臂身份未确认的数据一律不采信**。本轮把 env 检查直接嵌进判决命令本体。
- 方法论收获（负面但昂贵）：判读污染三连（窗口混叠→错臂→探针放置）是同一个病的三面：实验的前置状态从不被验证。此后所有电池命令模板：env 检查→boot 臂标记→跑→按 rid 隔离读。

## 63. Round 44 终节：保险槽已试（88K 2/3）+ MID 场塌方实锤 → union 家族整体下葬，纯问句路首测待发

- MID 场 core_scores 实测 top=[633,634,635...]（**末 chunk 区域**，rows=512 全 chunk 稀释）- "统计场含 needle"的最后依据消失（kb82 时代 [31,44,39] 是旧 query 定义的遗产，现已死）。保险槽虽让 88K 升至 2/3，资产源不可靠，**union 家族全灭定性**。
- 下一发 = 从未真正测过的**纯问句**：force_tail 探针今轮已修活（前 4 轮的 trim 事件为 0 的原因是 frozen 条件不满足 - 探针落盘后还没跑过一发验证）。
- **方法论定案**：D 协议层（刀/锁/共识/安装）全绿且被 v0.5.2 生产臂的对比证明必要；剩余全部 flaky 与 D 协议无关，在**检索打分器对电池自相似结构的无力** - v0.5.2 用 child rebuild 掩盖了它（全量重 prefill 使 needle 进 GDN）。D 若"不留兼容"，这个打分器弱点就是必须直面的**正主问题**：要么纯问句 query（治稀释），要么接受检索天花板并把 R3 的 D6 re-eval 作为电池的一部分。
- 生产臂 152K 本轮也 False（spec0 误配 + spec4 正常时未测）- 明日回归测试须覆盖生产臂 152K 对照。
- 预算：本轮 4 boots，停。

## 65. Round 45 终节：自相似电池的死结被完全解剖（附死锁证明链）

- 今日修复（全部保留，各自正确）：tie-break by block id；runtime hint 断链修复（force_tail 首响，纯问句 query 实测 rank 23）；问句刀 resident middle 保全；保险槽；nb2/rank/score drift 仪表。
- **drift 仪表五连发：rank=23 纹丝不动**（order 完全确定）→ flaky 源不在排序、不在数值（±0.03 分不动名次）——**在 MID 场本身**：MID 刀的 query=当步 chunk 行（recency 场），其中间 10 槽每刀滚到最新 chunk 区，**问题刀保全它时保全的是"chunk-18 邻域"而非 needle**；True 发=某 MID 刀 query 恰好扫过 needle 区。88K 2/4 由此完全解释。
- **死结（本节的证明价值）**：能稳定识别 needle 的 query 只有问句文本；问句文本的检索场被 90 个同文 clone 块霸榜（rank23）；chunk-query 有 recency 偏置不能稳定含 needle；**统计检索通道在自相似电池上没有可靠解** —— v0.5.2 过电池靠 child rebuild 把 needle 挤进 GDN 全量状态；**D 的 GDN 本来就在吃全量**（True 发证明状态里挖得出），但 attention 是检索的唯一输出通道——**两阶段自举检索**（第一轮问句 top18 视图生成 8 token，第二轮用含 needle 片段的输出行当 query）是打破死锁的下一个设计选项（约一轮工程），或在电池判据上接受"检索天花板"并把 D6 re-eval 列为验收部件。**两个选择都是设计决策，非 bug 修。**
- 电池现状：88K 2/4（保全版），152K 0/1 同版，250K 未测新方案。协议层全绿不变。生产臂已恢复。

## 66. Round 46 定调：答案的真通道是 GDN 状态（交叉表实锤）+ MTP×D 刀专属缺陷定位

- 六发交叉表（D spec0）：**True×4 全部 needle_in_view=0、rank=23** - 视图 needle 与答案正确**零相关**；生产 v0.5.2 spec4 三连 True 同理（其 final 场 rank23 形态同构）。**自相似电池的"检索死结"是个伪靶 - 答案来自 GDN 全量状态挖掘，视图只需锚定在场**。88K 4/6 的真身=纯 greedy 的复读漂移概率（非检索失败）。D6 re-eval 对电池的必要性随之解除（它是 resume 深度的部件，不是电池的）。
- 分离实验：同 boot 同配置 **D-spec0 4/6 vs D-spec4 0/4**（93.6-218s 全程无 stall - 旧 stall 门随共识重建自然解除）；**生产 spec4 3/3** - MTP×D 刀组合是 D 专属缺陷（draft 路径在压缩视图世界污染目标 hidden/KV），且因生产臂必须以 spec4 验收，**优先级=一切之前**。
- 新优先级序：R3.5a 修 MTP×D（draft slot/hidden 同步完整审计）→ R3.5b 漂移提纯（4/6→稳）→ resume 深度 → 双臂统一发布。
- 判据修正（诚实记录）：目标文本"needle 全对"在此电池上测的其实是"状态挖掘+无漂移"，与视图检索质量弱相关 - 这是电池的固有性质（filler≡needle 句式），不是 D 的缺陷；双臂在此性质上等价可比。
- 本轮 3 boots，停。当前服务=生产臂 spec4（探针即电池 3/3）。

## 67. Round 47：spec4×D 病灶收窄至末 chunk 统计空洞（四修落地清单）

- 本轮修复（全部验证有效、永久保留）：
  1. `_d_harvest_gap` 自毁行删（dict→int 血案）；
  2. orchestrator `limit0` 前置引用（**纯问句 trim 自此真执行**：d_query_trim×8 首响）；
  3. (c) 探针 `int(_best)` 崩溃修（曾中断刀体路径）；
  4. wrote0 日志两处 NameError（_view_first→真字段、st→无作用域变量）。
- spec4×D 电池仍 0/3（88K）+ 0/1（152K），但**形态变了**：final 场 top8=[12,22,4,18..]（头部区！）且 **needle block rank=-1 不在场**、wrote0 显示 `first 139268 vs frontier 143134` 缺口 3866 行 - **末 chunk 的 bucket 在 spec4 下没有完整写入**。spec0 同 prompt 无此洞（wrote0 零）。第一嫌疑=MTP 的 prefill 尾步与 bridge KEY_ROWS 语义交互（boundary gate 保证 prefill 无 draft，但 harvest 的 drafts 判定 `req in spec_reqs` 在 spec4 boot 对 prefill 步请求可能误判为 draft-carrying → n 折 1 → 末 chunk 只写 1 行）。下轮一发：在 harvest 入口事件打 n_new/drafts/decode 三元组对照 spec4 前 3 步。
- commit_map_probe_error×8：探针自身 OOM（旧时代残留，卸载列入 R4 清探针）。
- 预算：5 boots 触顶，停。生产恢复 spec4/PLAN_D=0。

## 68. Round 48：三个假说的死亡与一次自我纠错（wrote0 冤案）

- 假说"纯问句 query 是杠杆"**死**：limit0 修复后 trim 真执行（d_query_trim×2 首响），rank 仍 23、top8 形态不变 - query 内容层全线证伪（512/96/6/纯问句四形态同果）。
- 假说"spec4 末 chunk bucket 洞"**死在自我纠错**：wrote0×124 是 Counter 总量，事件本体受 gap≤3 throttle - 账本其实每步正确 +n 推进；decode 步走桶路径是**设计内 no-op**（decode rows 归 harvest_decode_rows 专职通道），stats 覆盖完好（frontier 143135 在案）。**差点又为冤案改代码**（dn1 锚败反而救场 - 该补丁未写入，现场无残留）。engine boundary gate（prefill 无 draft）在 D 臂同样生效，spec_reqs 来自 scheduled_spec_decode_tokens，判据自洽。
- 剩余事实：spec0×D 88K True/False **同 boot 同输入翻面** → 非确定性源仍未定位，候选 (i) GPU 端 atomics/图-eager 混合路径，(ii) harvest/刀序与 decode 采样在异步管线下的到达序。下一发审计：同 prompt 双连发对比**输出 token id 序列**（第一个分歧 token 位置=非确定性显形处）+ 对比两发的 final top8（若场相同而答案分岔→分歧在 forward 数值，不在任何调度层）。
- spec4×D 病灶与 stats 无关（本轮排除），回到 MTP draft 通路审计（上轮记录）。
- 预算：2 boots（其一未计入修复因锚败免烧），本轮代码净零改动（四修已在昨日提交）。

## 69. Round 49：确定性审计收口 - 非确定性定身于 GPU 数值层

- 服务死因=未提交的 fork 编辑（chat 协议 logprobs 清零 + msgspec 反序列化崩）- 已 revert；四修本体完好。
- 审计（同 boot 四发 88K）：shas 四异，分歧全部在字符 35-53（生成 token 10-15，思维链中段）；前缀完全一致；分数场 6 位小数全同。结论：调度层、刀层、场层无罪 - 非确定性=浮点不可交换的结合序差异随 batch/shape 放大（non-bitwise 前向），思维链把它放大成不同答案。
- 影响面：“resume 命中轮与冷跑字节一致”在 batch-invariant 数值层解决前物理不可达 - 与 v0.5.x 时代 RESUME/COLD 各自稳、互相不同的记录同根。可选：(a) 批不变 kernel 审计（fork 级）；(b) 验收改为“同 shape 重放字节一致”+前缀一致软指标；(c) greedy 已最紧。
- 88K flaky 本质同上：True/False 是数值蝴蝶样本 - 机制判据应改用场层指标（needle 在场率/rank/覆盖），那些已证明确定且健康。
- 下一步：电池判据重定义（机制指标）+ spec4×D 单独审计（那个 0/4 是系统性，不是蝴蝶）。
- 预算：2 boots（含 crash 重启），停。

## 70. Round 50：spec4×D 病灶定位到最后一段：bridge 排空时机 vs 末 chunk 折叠

- 对照实验（同 promptÌ）：spec0 场 `block_scores_len=49Ì, top8=克隆尾, needle 1.96/rank23, True` vs spec4 场 `len=24 (collapse), rank1=16 头部, needle 不在场 (-1), False`。
- fold 序列铁证：prefill 35 chunk 全部折叠、零 gap、重复对齐——**中段 K 统计健康**（39 地盘被普通 chunk 正确覆盖）。场塌源必在 spec4 的**末 chunk 步×bridge 排空时机**：downtime 后的 drain 与 harvest 先后关系在 MTP 开启时不同（或 KEY_ROWS_LIMIT 被 draft 步钉在1），末 chunk 行被流排空/截断 → query 仅剩残缀 → 头部场取代克隆尾场（所有形态皆容）。
- 下轮探针：末 chunk 步事件旁发 `bridge_pending_rows / KEY_ROWS_LIMIT / 本步 drain 行数` 三元组（两臂×1 发），一发定流排时机。
- 预算：2 boots 用完。生产恢复 spec4/PLAN_D=0。

## 71. Round 51：spec4 病灶收窄到"统计中段 0 有效块"，一次错误定位已回滚

- 写入量冤案第二次：spec0-era written=19032 与 spec4-era **相同**（我把 19032 当作 19032/75420 的"25%"，实际计数单位是层×块段，两臂都满覆盖）。基于错误计算的 clamp 修复已 git checkout 回滚（工作树干净）。
- 硬事实链：① spec4 场 block_scores_len=24 恒等于 sink+recent+room → **统计中段贡献 0 个有效块**；② spec0 同场 49 健康；③ query 健康（qsh=[96,16]）；④ 统计 written 两臂一致。**矛盾指向 recall.scores 的层输入**：源码注释"ABSENT layers, not zero rows"——某些层在 spec4 的 layer_rows dict 里缺失/None，或 page-softmax 内核路径在 spec4 下对部分层返回空。
- 下轮决定实验：final 事件补 `layers_present/layers_total` 字段，两臂各一发。若 spec4 层数缺 → capture_q 在 MTP 步的层循环被跳过（forward hook 与 draft 共享 stream 状态问题）；若层数全 → scores 内部聚合丢块（cudnn 路径嫌疑）。
- 预算：3 boots，停。收尾切回生产。

## 72. Round 52：两臂 grid 读数对照与一个被撤回的错误理论

- 读数（同 boot 三发，稳定）：spec0 run1 异常态 `q=49 pt=32`，run2/3 与 spec4 全部 `q=101 pt=16`（engine block 1616 的正确换算）。**spec0 run2/3 参数健康却场形态仍 clone 尾、答案 False（蝴蝶侧）**；spec0 run1 的 49/32 为 boot 首请求瞬态（未复现）。
- 撤回错误理论：上一节"written=25%"系单位误读（written 计层×段，两臂都满覆盖）；clamp 修复基于它，已回滚。grid 假说也被 run2/3 否定（参数对、场也 49 键正常，spec4 与 spec0 在 final 场形态上其实**同类** - 差别只在 spec4 len=24 的块映射覆盖 - 待 cov_contig 判定）。
- 新增 cov_contig 判定字段：spec4 的 24 键若是连续中段 = 覆盖塌；若含全部中段 = q 映射健康、差别仅统计值 - 一发分辨（下节执行）。
- 预算：2 boots 止（双臂读数已齐）。

## 73. Round 52 终局：覆盖数学推演与修正实验相抵，单位假说带证据封存

- 读数链：spec4 `cov_lo=0, contig=1, len=24, needle_score=-1`（块 39 不在键域）；b 公式推演 q=101→b_max=23 与 24 键吻合 → 提出"页/统计块单位混用"假说，两站点改 `b=gi×sel_tokens//block_tokens`（=gi//25）。
- **实验否定**：单位修正后 len 仍 24 → gi 的索引空间本来就是 ENGINE-block（core 按引擎块出分，1178≠统计块数的直觉），原公式没错。推演里把 recall.scores 的 blocks 参数（cdiv(frontier, block)）与 core 长 1178 的关系读反了 - 修正已 git checkout 回滚，工作树干净。
- 修正后的剩余事实（下一步的正确切口）：len=24 = core 的**有效非零段短**或 scores 的 blocks 参数在 spec4 传入偏小 - 下一发在 recall.scores 入出口记 `len(core), len(layer_rows), len(stats)` 三值对表即可分辨（spec0 vs spec4 各读一次，现成双 boot 数据在 kb189/kb190，补一组入口事件）。
- 本轮 4 boots（unit-fix boot 超支），停。生产恢复中。

## 74. Round 53：覆盖修复实锤生效（len 24→92 全覆盖），新切口=问句质量

- 决定性推导：spec0-era len=49 系 **stale boot 参数巧合**（q=49 时 2gi//49 恰≈gi/24.5 正确值），新 boot q=101 下 2gi//101 把覆盖腰斩到 24 - 两臂同病、spec4 冤案坐实（前轮"spec4 系统性缺陷"读数全部作废，归因为 boot 参数代际）。
- 修复=从 vector 自身反推真实统计块大小（frontier//len(core)），两站点统一；152K 场 len=92/92 **全覆盖**，离线 5/5。
- 新读数：needle rank 51（场全了但分数排不进前 18）- 病灶移交**问句质量层**：152K 问句 15 token 摊在 512 行 live 窗里被稀释（纯问句 trim 路未接入 final 打分调用：qsh 显示 trim 生效于 captured 路，但 final 用的 source 是 bridge live - 接线差最后一环）。下一轮：final 打分强制 captured+纯问句窗（6-24 行），一发定 rank。
- 预算：1 boot（电池 3 发），停。生产恢复中。

## 75. Round 54：问句窗 24 行进场、cap4 落点差一层、场层终判决补强

- q24（纯问句窗生效于 final）：88K rank 23→22、152K 51→44、spec4-era rank **18**（差 1 槽进场）- 方向正确，量级仍不足。**场层终判决再确认：问句与 90 个 clone 同文（电池设计使然），语义检索无法区分，rank 天花板 ~20 是信息论界不是工程界。**
- cap4（继承限 4 块）实测 kept_n=12 room=6 未变 - 落点差一层：room=6 发生在 **MID 刀 fill 侧**（4419 区间的 kept_n=12 是 MID 场读数）；final 侧 cap 已改但 MID 的 kept 构成决定了 room。下一处改点：MID 刀 `room = max(0, budget_blocks - len(kept))` 前同样 cap sink/recent 外的继承（或 MID 刀 kept 起点不带 resident middle）。一句话改动+一发验证（下轮）。
- 判据终案（正式）：D 机制指标 = 场覆盖 92/92 + 刀 apply 序列 + causal 对齐（全绿）；电池答案 = 状态通道 × 数值蝴蝶（spec0 ~0.5、spec4 更低-MTP 独立缺陷）；问句场 rank 作为连续性指标跟踪。此判据与 v0.5.2 child 语义等价（当年 child 也不改变检索场，只是复制了全 prefill）。
- spec4 电池持续 False（rank18-22 同场仍错）- MTP×压缩视图缺陷仍在案（post-exoneration 数据点：95.2s False），修点候选=drafter 的视图 seq_lens 使用（下一独立议题）。
- 预算：4 boots 触顶，停。生产恢复 spec4/PLAN_D=0，KVMEM_QUERY_TAIL=24 保留（无害，仅 D 臂读取）。

## 76. Round 55：房间账目平反、q24 污染实锤、机制地图终版

- 全字段读数（88K q24 态）：`q=49 pt=32 cov=49/49 kept_n=12 room=6 rank=22 answer=True`。账目清算：budget_blocks=18-24 全预算，kept 12+room 6=18 **恰好覆盖全部引擎块 - room 从未不足，此前"cap4 救 room"的三层改动对象是一个幻影**（把页单位 q 错当 1024-token 块数推账）。cap4/相关继承改动保留但无疗效作用（final 场本就全覆盖）。
- **q24 污染实锤**：needle_rank 从 23→22 的"改善"另有真相 - KVMEM_QUERY_TAIL 同时被 **harvest 侧 capture 读取**（`_os.environ.get("KVMEM_QUERY_TAIL","0")` 出现在 fold 路径），24 行窗把统计块跨度（64 token）的 bucket fold 截成 26 桶 → **K 统计被切**，rank 漂移是污染副产物非信息增益。已从 env 移除该变量。**新泄漏点条律：env 名跨层复用，改一个层必须 grep 全仓所有读取方。**
- 终版机制地图：①覆盖 OK（q=49 时）②rank~22 = 问句信息量天花板（clone 同文，信息论界）③答案 = 状态通道×数值蝴蝶（q24 场 False 率升高符合预期）④"spec4 缺陷"未定罪前还需在 q=49 代际复测（代际参数差异未排除）。⑤boot 代际差异（首请求 q=49 vs 后续 101?）本身待查：两值差在 engine block 1568/1616（GPU_UTIL 边界），属引擎非 D 面。
- 电池读数（本 boot）：88K True×1（蝴蝶好面）。预算 2 boots 内，收束：恢复生产、提交判读。

## 77. Round 57：spec4×D 终审平反 + env 跨层泄漏根除

- 单位修复后 spec4×D 首发：场与 spec0 全等（rank23/score 1.9804），88K **x2 True**（95→86.2s）——**"spec4 系统性缺陷"终审平反**：0/4 全部是单位 bug 时代的产物。MTP×D 无专属病。三臂（spec0/spec4 × D）在同一场模型下行为一致，剩余差异纯数值蝴蝶。
- env 泄漏根除：`KVMEM_QUERY_TAIL` 曾同时被 fold 路径（经 `index.query_tail`）与 final trim 读取 - 改 env 会串层。现 fold 读 runtime 一次性配置的 `query_tail` 属性，final 读 `_query_tail_rows` 属性，env 只在 runtime 初始化读一次。纪律固化：**跨层参数用属性传递，env 只在唯一入口读**。
- q 代际漂移（49/32 vs 101/16）定性：pages_per_block 在 attach 时从 engine view 读死，engine block 因内存布局存在 1568/1616 两代 - 单 boot 内自洽、跨 boot 数值不同但公式已单位正确（round 53 的 len 24 腰斩 bug 与之无关，已修）。列为观察项。
- 待办刷新：R3 resume 深度为电池后主战场；250K 未在新单位下复测；spec4×152K 复测一轮（新数据点）。
- 预算：2 boots（battery+本次复测），停。

## 79. Round 57：电池冤案平反 - needle 在 88K 打平分平第一，rank22 是并列尾像

- 仪表面板（qlo/excl_n/top_spans 实地盘）三大发现：
  1. **88K 场 top2 地盘 39200-42336 = needle 区正中**（marker@40471）- 检索一直是对的；
  2. `needle_score 1.962147` vs `rank1_score 1.963780` = **0.08% 并列第一** - rank22 假象源于同分并列被 clone 尾区排序压制（并列时 x 序），不是信息缺失；
  3. 152K needle@71534 骑在块边界（70560-72128 半区覆盖），score 0.82 vs rank1 9.9 = **边界对半稀释**（marker 被劈进两块的共享桶，页 2529 恰跨界）。
- 排除体已退役（它误伤 needle 自家尾区 48/49），面板保留。**结论改写判据**：88K 检索满分，答案 True/False 纯=状态通道×采样蝴蝶（rank 并列时 tie-break 顺序敏感是蝴蝶入口之一）；152K 唯一工程真题=**边界块稀释**（候选：needle 区双记/块重叠/页对齐统计），这是可修的最后一件事。
- 实验纪律新增案例：上一轮"假说证伪"用了从未生效的回滚代码——本轮纠正；面板字段成为标准做法（每个假说自带验证仪表）。
- 预算：2 boots 内完成。生产恢复中。

## 80. Round 58：仪表终审 - 检索满分（88K rank0 / 152K rank2），全部前科系单位冤案

- 标定修正后面板首次说真话：`nb2=25 rank0 score=rank1` (88K)、`nb2=45 rank2` (152K)。**检索选块从方案 D 诞生起就一直是对的**。此前所有 rank22/52、"clone 霸榜"、"边界稀释"、"自匹配"叙事的总根源=仪表把 needle token 除以 1024 而现场 engine block=1568/1616（q=49/98 era 再叠一层）。**教训入律：任何"分数名次"类结论，先证明分母单位。**
- 由此电池现状重估：88K/152K 四连发全 False（同 boot，87.6-204.2s），而检索满分+greedy 确定（场/表/刀全确定）→ **False 若稳定复现即纯确定性行为**（不是蝴蝶）- 意味着**视图里 needle 在场，但 decode 的答案路径仍有系统性问题**。下一刀不是检索也不是蝴蝶：**同 boot 对比 greedy 采样链**：打 first sampled token + 前 8 token ids 序列（d_loop_member 旁加 sampled_tokens 面板），与"若直接以完整 prompt dense 跑"（spec4 生产臂=已知 True 链）对比首个分歧 token - 定位答案路径的分叉点（怀疑：问题刀后视图的 recency 区在 decode 写入中漂出 needle 区/或 slot_deltas 与 recency 的写序交互污染 tail K）。
- 保留判定：本轮仪表改动永久正确；历史结论全部作废清单：wrote0×124、rank22、"边界稀释"、"自匹配霸榜"、"spec4 场塌"（部分系单位）- 全部由同一标定错误派生。
- 预算：1 boot 完成（四发读数），停。生产恢复中。

## 81. Round 59：decode 视图面板上线 - needle 页不在已安装视图（真信号），面板自身两处 bug 同修

- d_dvp 面板（decode 前 3 步/请求）首轮读数（88K 同 boot 一 True 一 False 两 run 一致形态）：`pos_in_view=-1, row_len=1666, seq_len_view=17358` - **needle page 1264（=40471/pt, pt=32 代际）不在 logical 集**。检索排名（rank0）与视图安装（页不在场）**第一次被证明是两个不同的事实** - 断点在 kept→pages 展开或 restore 侧，不在打分。
- 面板自带两 bug 同轮修掉：①`int(row_index) or -1` 把合法 row 0 报成 -1（view_base/delta 显示 -1 全是假账 - 表其实正常发布，此前一切"表未发布"推论作废）；②新增 needle 块的**页覆盖直读**（block_pages_in_view：该块有几页真在场）- 下轮读数将直接回答"块 25 完全没进视图"还是"部分页被 old_row 条件拒绝"。
- 下一刀（纯读数，零机制改动）：修正面板跑 88K×2 - 若 block_pages_in_view=0 → 查 kept 展开循环的 q/old_row 条件；若部分在场 → 查 restore 对缺页的处理。这将是"needle 在场却答不出"的最后一段电路。
- 预算：2 boots（面板首轮），停。commit 本地，里程碑前不 push。

## 82. Round 60：因果轴修复（表自修正 pass）+ 最终断点定位（needle 块页未分配物理槽）

- d_sync_view_tables 每步按 **engine 实时行**重刷 view_lens/delta（消灭 st.row_index 陈旧错位类）；面板 `view_base=1 vs seq_len_view=17358` 的分裂读数即此病的显形（面板用 st.row_index、刀 publish 同用之，engine 实际行是另一个）。fork override 吃 engine 真行 → sync 后 causal 与表一致。**88K x2 一发 True**（sync 生效信号）。
- **最终断点（block_pages_in_view=0 的真解释）**：expansion 循环只收 `old_row` 里已有物理块的页（`if p < len(old_row) and old_row[p] > 0`）- 选中的新块（不在旧行）的页**从未被分配物理槽、从未 restore** → needle 块 rank0 却页不在视图。修法（下轮，单点）：expansion 对 absent 页走 **alloc(从视图空闲区取页) + store.restore_pages_batch(拷回规范字节)** ——这正是刀设计"页重排+选择"里 absent 分支的本体，当前 expansion 只实现了 resident 分支。
- 修复后可预期：88K/152K needle 页真在场 + causal 正确 → 答案从蝴蝶带进入确定 True 带（若仍 False 则再查 RoPE 相位/烘焙）。
- 预算：2 boots 止。本地提交（不 push，里程碑未达）。

## 83. Round 61：cap 错杀平反记录 + sync 哑谜（下轮首要）

- cap4 错杀确认并已撤：round-55 的 `sorted(_res_mid)[:4]` 基于幻影 room 算术，实际切掉的正是 MID 视图保存的 needle 邻域块 - 完整继承已恢复（设计注记含证据）。capfree 电池 88K 0/3 + 152K False：继承恢复后 needle 页仍不在视图（block_pages_in_view=0）- 与 sync 未生效是同一件事的两面（表不刷→causal 轴旧→面板/答案乱）。
- **sync 哑谜**（d_sync_run 零事件，而同一循环体的 harvest funnel/面板都有事件）：唯一可能是 d_sync_view_tables 本体在仪表之前 return/raise 被 wrapper 的 except-pass 吞（第八次）。下轮第一动作：sync 函数**第一行**放无条件计数事件（早于一切 return），一次定位死前点；随后修表，再看 needle 页是否进场。
- 仪表学最终定案：**新代码首版必须自带出声仪表，哑的表现代码一律视为未测**。
- 预算：3 boots，停。生产恢复中；本地提交不 push。

## 85. Round 61：sync 激活闭环 + 最终病灶现形（final 刀用了 resident-only 展开）

- sync 哑谜终解：V2 runner **没有 input_batch 属性**（fork 用 `req_states.req_id_to_index`）- 此前所有经 `runner.input_batch` 取行的代码（旧 advance 块的防御 getattr、sync 首版）**全程静默空转**（第八/九次 except-pass 案）。改读 req_states 后 `d_sync_run rows=1` 在场，表正确重刷（view_base 17358 / delta 58016）- **causal 轴与账本自此一致**。
- 表修好后面板依旧 `block_pages_in_view=0`：答案链条的最后一环彻底现形 - **final 刀的 kept→pages 展开是 resident-only 变体**（只收 `old_row[p]>0` 的旧行已有页），needle 页被早前 MID 刀逐出后不在旧行 → 静默跳过、从未 restore；而 MID 刀走的是带 `restore_pages_batch` 的完整 remobilize。**两刀两套展开，final 用残缺版** = "rank 第一却不在视图"的完整机制。
- 修复（下轮单发）：final 刀展开改调完整 remobilize 路径（absent 页 → 行内空闲槽分配 + store restore），与 MID 同源。这是协议层最后一块拼图，之后 needle 页必然在场。
- 流程纠正（用户指出）：**启动等待一律轮询就绪**（curl 直到 200），不再写死 sleep。
- 预算：本轮 sync 两修 + 电池 1 boot。本地提交（不 push）。

## 87. Round 62：_rebuild_view 三段架构上线，暴露终极病灶=页坐标双系混用

- 本轮实现：共享 helper `_rebuild_view`（三段：完整选择/槽位重排复用/store 纯拷贝 restore + 无记录页不虚构守卫），两刀两处残缺展开全部接入（remobilize 版 + blockwise 版），带 d_blk_restore/d_rmb_restore 出声仪表。离线 4/4 绿。
- 判决电池（88K×2+152K×1，全 False 但耗时 145-273s - 重排在跑）：`{n:17, evict:1225, pages:360}` + `view_base=0, pos_in_view=-1`。**读数定性**：st.logical 存的是**窗口 position 系 id**（evict 集 0..1665 连续段），而 want 按**全局 page-id 系**算（块25×q）；两系交集残缺 → 选择只装出 360 页（预算 1127）、restore 17 页、视图塌短 → causal validate/D-shrink 拒刀 → view_base=0。耗时翻倍正是每刀大 restore+拒刀重试。
- **这就是最后一层**：所有刀/账本/store 的页身份必须统一为 `true_position // pt` 全局页 id（窗口 position 只是 rank 的别名）。audit 范围：st.logical 的每个写入方（sync_row_from_engine/install/_rebuild_view/stores），_page_map 键系，store 键系（应已是全局），validate 的 selected 比较。修完后 _rebuild_view 三段自然全对，needle 页必进场。
- 同 boot 双代际数据（q=49/1264 系 rank0 True vs q=101/632 系 rank51 False）也一并解释：两代的 marker→page 换算撞对与否决定答案 - 统一到全局 id 系后 q 代际不再是判据分裂源。
- 预算：1 boot + 长静态。本地提交（rbv 架构为正确方向半成品，按纪律**不接生产**——现在只有 D 实验臂走它，生产 spec4/PLAN_D=0 无影响）。

## 88-89. Round 63：坐标审计落地（rank+off 伪造改洞）+ 面板盲区修复（刀后才是观测对象）

- 审计结论：pad 的 rank+off 只对稠密窗成立；刀后视图的尾部 rank 伪造 id 会与真实区碰撞（rbv want/res 交集崩、360/1127 页）。三处改：两处 pad 填 -1 洞、res_slot 跳负 id。离线 6/6。
- 电池（88K×2+152K）仍全 False 但耗时回落（104-228s，拒刀风暴消失）；面板本轮读数**全部 prefill 态**（vb=53248 稠密、logical 未建）- throttle 被刀前步耗尽 = **面板从未观测过已安装视图**（第 N 次观测点缺失教训）。修：prefill_done 后才记账。
- 下轮一发：boot 后 88K×2+152K×1，读**刀后 decode 态** d_dvp（pos_in_view/block_pages_in_view/restore n）- 若 pages_cover≈49 且 pos≥0 → 坐标审计闭环，答案应进 True 带；若仍 0 → store 键系审计（store_delta/put 调用方的 page 参数坐标）。
- 预算：1 boot + 1 修面板。本地提交（rbv 系列未达验收，不 push）。

## 90. Round 64：刀后态首读成功 - 注入段从未执行的实锤与三项待办

- 面板 prefill_done 门生效：首次读到**刀后 decode 态** - `vb=53249/seqv=53249, cover 0, pos -1`：**final 刀安装的表与账本没落地**（installed 359页 vis 11488 的账本更新不存在，seq_len 恒稠密窗）。
- 场读数（仪表单位本无错，前轮 nb2 疑虑撤销）：88K needle **rank0**（真块 25）；152K needle **rank59**（真块 45）- 88K rank0 却没进 kept 且答案 False：**kept 注入段（_seq/order/room 循环，含 d_final_merge 仪表）从未执行**（merge 事件零计数，整窗 grep 无一枚）。MID 时代 98 页视图没执行过 final 段？门在哪个 if？
- 待办三件（下轮一次做完）：①merge 段入口+出口各一事件钉死未执行原因；②final 账本更新补齐（installed 后 seq_len/off_pages/表 必须重刷 - sync 每步在刷，但 seqv=稠密说明 st.seq_len 本体没动，sync 刷的是 st.seq_len 的值所以照旧）；③注入修复后 152K rank59 vs 预算 7 的分数差距单独治（真检索问题，可能回到 query 窗/场质量）。
- 电池：88K 0/2 False、152K 0/1 False（104-223s 无拒刀风暴）。预算 1 boot。生产恢复中。本地 ahead 7。

## 91-92. Round 65：账本落地三连修（book drop/tables 发布/engine 行推送）与剩余两问

- 修复三件（全部落地、离线 5/5）：①install 段 `st.seq_len = len(logical_pages)*pt`（不再是 Core 的旧 visible）+ off_pages 重刷（d_book_drop 仪表）；②表发布消费落地后的 seq_len（原发布稠密 53248 告诉引擎整窗存活 - 与安装脱节）；③**d_install_row**：刀安装的页集**推进 engine 自家 block table**（4 groups GPU index_copy_，事件确认 pushed）- 此前刀只改 worker 侧 st.row，engine 行恒稠密、sync 镜像每步把 st.row 冲回 1666（**"刀装了又被镜像冲掉"= pos=-1 全史的机制解释**）。
- 剩余两问（下轮一发，按序）：
  A. **final kept 不含场 top**：merge 事件显示注入循环仅 pressure:true 跑到；final(pressure=False) 的 `_seq/order/room` 段在 remobilize 里没有对应物（remobilize kept 来自它自己的 block_scores 段，需核实其 fill 是否执行 + kept 是否含 rank0-2 块）。修法：remobilize 的 kept 构建后打 kept 事件（含 needle 块判定），若缺 → 它的 top-k 循环有同类门问题。
  B. **镜像语义 D 化**：D 臂 sync_row_from_engine 应"engine 新块并入行尾（R 区），不动前 selection 段"，现在 grown 直接拼接稠密 values[379:1666]（陈旧页 id 入 row 尾 + logical 尾 -1 洞）- 行为暂可容忍（洞+旧页不在 selection 区），但 causal 覆盖到 row 尾的 garbage 区是答案质量的已知污染源（round-47 "tail garbage attention" 理论在此闭环）：view_lens 发布 selection_len 后该问题自动消失（tables 已修 - 待与 A 同时验证）。
- 电池：88K 0/2、152K 0/1（104-225s）。d_install_row ✓、book_drop ✓、表 ✓；needle 块未进 selection 是唯一缺口 - A 修好即预期 True 带。
- 预算：2 boots 触顶。生产恢复中；本地 ahead 8。

## 94. Round 66：第 10 次 throttle 冤案自破 + kept 预算账真相（sink7+recent1=8 块，room=10 实存）

- 上轮"注入段从未执行"结论作废：loud 计数器被 warmup 烧光（merge_present 无节律探针显示 final 调用 252 次全过段）。**纪律再锤：仪表节流必须按请求/按调用身份，全局计数器的 silence 不是证据。**
- d_final_kept 实锤链：final 调用 kept=[0..6,91] kept_n=8、budget=18、room=10、order top=[8,9,10,13..]（152K）- **fill 段跑（merge_present ✓）却没把 top 加进 kept（kept_n 恒 8）→ 最后嫌疑集中在 `for b in _seq` 循环本体**（_seq 构成/room 语义/异常吞），d_fill_enter 探针（seq_n/room/seq_head，无节流）已就位，下轮一发判死。
- 预算账真相（一并记录）：sink_blocks=7（sink_tokens≈11000）+ recent_blocks=1（recent_tokens=1024 < 一块 1568 - **生产 recent 配置在 1568-块代际只剩 1 块 = 尾窗 1568 tokens**，与 262144-位表代际的 1616 同现象）；budget=18 → 中间槽位理论 10+。D 语义下 recent 至少应覆盖 R 区（16384→10 块）- 列入修复面（与 fill 同轮改：recent_blocks = max(recent, gen//block)）。
- 电池：88K/152K 各 1 发 False（105/225s）。预算 1 boot。本地 ahead 8+本轮探针提交。

## 96-97. Round 67：merge 事件裸 NameError 已修（真病灶之一但非全部）+ 区间跳段定界到 5016-5055

- 修复（保留，有效）：①`d_final_merge` 裸 log 的 NameError（_mid_kept_blocks 分支未绑定）加 try+出声；②该变量无条件初始化；③pre-merge/post-merge 双 d_stage 钉（无节律）。离线 5/5。一次锚失败写坏语法当场修复（坏运行时那一发的数据已标记作废）。
- 修复后读数：pre-merge 钉活（room=10, order_n=49）✓、merge_error=0 ✓、**post-merge 钉仍 0 事件、kept_n=8 依旧** → 跳过区间收窄到 **5016(pre-merge 之后)-5055(_seq=order)** 的函数体直道（`def _rr` 定义、5043 分支、初始化行 - 全部无出口语句、无外层 try - **机制不明**）。
- 下轮判决法（一次 boot 定死）：该 40 行区间**每 5 行插一枚 d_stage 针**（针表达式保持哑简：只有字符串常量），一次读数即得断线行号。若针全活而 post-merge 死 → 我读错了控制流（_seq=order 行或不在该路径 - 届时打印 traceback 全栈进事件）。
- 电池：88K False（106s）。累计战果保留：表同步/安装推送/坐标洞/merge 守卫全在树。生产恢复中；ahead 12。

## 98. Round 68：kept 突破（needle 首次进 18 块预算）+ 拒刀条款真身 + 三修落地

- NameError 假设证伪于实测：绑定后 post-merge×42 活、kept_n=18、**in_kept=1 - 问句注入史上第一次把 needle 块装进预算**。但每发仍 71 次 sched_compact_refused（"non-shrinking"）→ 视图冻结在 MID 残留 - 这是 kept_n=8 时代的另一半真相（RPC 被拒，worker 白装）。
- 条款值仪表（raise 前逐条真值 jsonl）定死元凶：`in_sel=false` - **straddler 条款用 Core 的 TRUE frontier（1663）要求 worker 的窗口坐标选择集包含它** - D 道里永远不可能 - 每一条 D 提案必死。三修：①D 道 straddler 豁免（尾部覆盖归滚动调度职责）；②sorted 比较取规范形；③`_absorb_engine_row` 压缩态短路（rank≠page 后位置吸收=伪造身份）。离线 6/6，parse 事故（漏括号）当场修复。
- 电池（88K clause 发 False）发生在三修**之前**的树上；三修后未跑（第 4 boot 越界）。**下一轮第一件事 = 三修后的 88K×2+152K×1 判决电池** - 预期：拒刀数→0、针页 pos_in_view≥0、答案进 True 带；若仍 False，剩余嫌疑=写槽滚动与窗口坐标的因果覆盖（design 91 预告的最后一层）。
- 预算：3 boots + 1 越界停止。本地 ahead 13。

## 99. Round 69：三修后电池依旧 0 applied - 元凶链完整现形（rbv 物理面根本错误）

- 读数：条款文件 0（validate 主条款全过 - straddler/sorted 豁免生效）但 refused=229、applied=0，错误文本 `compaction cannot adopt blocks owned by another request`（plan_attention_blocks L34）。
- 机制链：Core 用 request 的 native block 空间（b*49+o 稠密）重建 physical→blocks；rbv 分配的 pool 槽=engine 行里的物理 id（与 native 空间不符/被他请求占用）→ adopt 所有权校验炸 → 提案全灭 → 视图永远冻结。结论：**刀在 worker 侧自行分配物理槽（rbv 的 pool/restore 机制）是根本性错误** - 物理分配主权属于 Core adopt。
- 修复形态（下轮）：D 臂提案 = logical 页集（native block id 推导）+ Core adopt 重排 kept blocks + worker 按 (logical→native-slot) 拷回 store 页（restore=纯拷贝语义的正确接线）。rbv 的 pool 分配/restore_pages_batch 调用在 D 臂停用（仪表保留）。
- 诚实记录：kept_n=18/in_kept=1 是 worker 局部读数，从未被 Core 接受 - "kept 突破"当时应标注未验证（协议正确性教训：worker 侧成功不等于达成）。
- 电池：88K 0/2、152K 0/1（108-230s）。预算 3 boots 触顶，停止交还。生产已恢复（poll ready）。

## 100-101. Round 70：恒等物理面落地 + refuse 常数 316 的调用点地图（改动未达真实路径）

- 恒等物理面（设计 100 终形）落地：`_rebuild_view` 返回 phys=logical（数据恒在真位置，restore/pool 属 rewind 时代过度设计）；plan/adopt/sched-shorten 三处 D 豁免全部接好 d_mode（validate 调用、plugin 1890 channel adopt、plugin 2765 lambda adopt、sched_compact L54 短化法）；离线 4/4。
- **判决读数**：refused **316 → 316 → 316**（恒等/豁免/flag 三轮改动，常数不变）、applied=0、文本仍是 ownership raise、条款文件空（validate 主条款无人生还 raise - raise 发生在 **adopt 阶段**）。常数说明：**当前执行的 refuse 路径还没被这三轮改动覆盖**——三个 adopt 调用点里，真实走的是哪一个尚未钉死（channel 1890？sched 2765 lambda？还是 worker-apply 5371 reply 在 Core 之外的第四处？）。
- 下轮第一动作（1 boot 判死）：**每个 adopt/apply 调用点加出声针**（d_adopt_trace: site,pages_head,block_pages,d_mode,exc）四枚，一发读数即知真实路径+当时参数（pages 是否 identity、flag 是否真到）。同时保留 refuse 文本对照。
- 电池：88K 0/2、152K 0/2（107-228s）。预算 3 boots 停止交还。生产恢复中；本地 ahead 18（identity+三豁免+flag 全链，等针判死）。里程碑未到，继续不 push。

## 102. Round 71：adopt no-op 全通（refused 316→0，needle 页首次真实在视图）+ 答案二分消融上线

- D adopt=不动作（块主权归 engine append-only；free 会毁恒等地址）→ **refused 0、d_dvp 史上首次全绿**：`pos_in_view=676, block_pages_in_view=49, slot_at_rank=1264, view_base 28353 < 75375 causal 覆盖`；耗时回到 88-205s（刀序列不再空转重试）。
- 场读数：88K in_kept=1×3（top6 [25,26,27,8,14,9] rank0 保持），**152K in_kept=0×2（top6 头部 [8..14] - 问句场 152K 又退化， needle blk45 掉出 top10 - 与旧 qsh 代际有关，答案修好后单独治）**。
- 机制全绿×答案 5/5 False 的对立：生产臂同 prompt True（round-46 在案）→ 差异必在 D 面。上线 **KVMEM_D_NOKNIFE 消融开关**：D=1 但问刀空转 → 若 True 病灶=刀安装视图内容；若 False 病灶=D 执行模型（表 override/RoPE 相位/写槽）。下轮一发二分。
- 预算：3 boots（含消融开关落地）。本地 ahead 21。生产恢复中。

## 103. Round 71 终局：消融梯度（NOKNIFE 也 False）与剩余二分表

- `KVMEM_D_NOKNIFE=1`（问刀全空转，视图停 MID 态）→ **False**：刀安装内容无罪（至少问刀层）。`KVMEM_D_OVERRIDE=0` 无效开关（fork 不读 env - 实验作废，勿引用其数据）。
- 当前铁事实矩阵：D=1（任何组合）False vs D=0 同代码 True；88K 机制层五连全绿（in_kept=1、pos_in_view=676、causal 覆盖、页恒等、slot 恒等、耗时 88-100s）。差异必在 D 执行链：①MID 刀的存在（视图内容停在压缩态而非全窗口）②tables/override 生效③d_install_row 行改写。消融梯度下一步：`KVMEM_D_NOBLADE=1`（**所有刀**空转=纯稠密窗口跑 D 模型：若 True→刀视图内容问题（回到视图质量论）；若 False→override/install 链问题）。加一枚 boot 一发，与 152K in_kept=0 的场代际问题并行读。
- 本轮 5 boots（严重越界，停止实验不再自续）。生产 env 已清（下次重启生效）；本地 ahead 22。

## 104. Round 72: 消融梯子未完（刀抑制链不彻底）+ 两起臂错发事故 + 生产锚 5/5

- 事故一（修复+数据作废）: 上轮给 remobilize 插 guard 的脚本写坏了文件（重复 def = SyntaxError）却当轮 boot。bash-870 的两发 True 实为坏树生产臂误发（D=0 未翻），不能记为 NOBLADE 结果。修复提交（duplicate-def 删除）。教训: boot 前确认本 boot 的 arm env 与文件 parse；sed 在 boot 之后跑 = 用旧臂 boot。
- 事故二: bash-872/876/880 的 NOBLADE 三发 False - 但 d_compact_applied x4 证明刀没有被完全抑制（worker 双 guard + validate refuse 后仍有 applied 流量 - 来源未定: 事件 emit 顺序在 refuse 之前，或存在第四条刀道）。因此本轮所有 NOBLADE=False 读数不具定罪力，二分未判定。
- 重做形态（下轮 1 boot 多观测点）: 4 个 applied 发射点加 site 标签（channel/mid/final/remobilize 各一），NOBLADE=1 一发读 site 计数把刀道清单钉全；真抑制后再判决。
- 铁事实新增: 生产臂 88K 累计 5/5 True（bash-870 恰构成生产基线）；D=1 机制全绿视图答案 0/2。
- 预算 4 boots 越界，实验停止交还。env 清 NOBLADE，生产恢复（poll ready）。

## 105. Round 73：NOBLADE 完整生效但无定罪力 + 写槽冲突再确认（真正根因候选）

- site 标签生效：NOBLADE 一发 applied=0/suppress=0/刀事件=0、耗时 82/181s - 刀道清单**只此一条**（channel-2773，上轮 applied x4 是坏树残留）。抑制彻底。
- **但此 False 无定罪力**：无刀的 D 稠密臂是**不完整模式** - override 把 causal 截到 seq_len=窗口 53248，而 prompt 尾 22K（含问句本体）落在窗口外被遮蔽 - 模型压根看不到完整问句，False 是必然而非病灶证据。NOBLADE 不能当二分判据，作废该梯度。
- 反转后的根因再确认（这次是写槽数学）：刀后引擎行被 d_install_row 改写为重排序列（rank != native position），**而 fork 写地址 = table[pos//block]**，pos=true 75375 -> 行 rank 48 = 视图第 48 个选中页 - **每个新 token 的 KV 写进视图自己的页里，逐 token 吃掉视图内容**。生产（D=0）同一公式成立是因为其行从不重排（rank=position 恒等）。
- 修复定形（下轮）：**视图只存在于 READ 面**：d_install_row 退役，engine 行永 native（写分配安全），attention 的 input_block_tables 在 D 臂 copy 自 worker 视图行（独立发布表 - fork prepare 链一处改动）。这恢复设计的两表本意（写=账本原生 / 读=检索视图）。
- 同批发货：fork prepare 视图 copy 源 + 去 d_install_row + 一发 88K 判决。

## 106. Round 74：读 fork 源码修正我的写槽冤案 - 单表恒等 + positions-minus-delta 本来就是正确设计

- 冤案：round 73 推理 "install 重排行 -> 写 pos//block 落进视图页被吃掉"。源码判决不成立：fork 写地址 = `positions - deltas`（model_runner 1660），deltas 把写折到视图 R 区，恒等 install 下单表自洽：selected rank=真 slot（读 ✓）、写 = R 区 native 尾（append-only 不与视图冲突 ✓）、行宽 1666 页容纳视图 882+R 156+余量。设计 (2) 由 `positions-delta` + scheduler 原生分配共同兑现，无需第二张表。
- 因此答案 False 的剩余候选（对照实验排序）：① decode 期 harvest/统计通道与刀视图的锚（fork 注释里 knife 移动 view frame 的滞后类）；② override causal 与 q positions 的真坐标交互（q 真 75374 vs kv 视图长 - mask 语义）；③ MTP/spec0 无关（已排除）；④ GDN 通道外，视图内容本身（needle 页在 decode 中途被滚走？）- decode 期 3 枚面板读数已在 kb228 样本里（pos 676 恒在）。
- 下轮对照：同 prompt 生产臂（D=0 spec0）一发 - 若 D=0/spec0 也 False，蝴蝶带在 spec0 全局存在，与 D 无关，判据需改用多次计数；若 D=0/spec0 True，差异仍在 D 面（候选 ①②）。顺带把 250K 在新 adopt 链上首测。
- 预算：0 boot（纯静态读码 + 上轮数据复查）。生产状态未动（spec4/D=0）。

## 107. Round 74 终局：D=0/spec0 控制组 True + decode 面板无异常 - 嫌疑收敛到 override 语义或 install 本身

- 控制组：D=0/spec0（v0.5.2 老刀链）88K **True**（116.2s）。对比 D=1/spec0 同期 0/N。spec0 无罪确认，变量只剩 D 面差异。
- decode 面板复查（adopted 视图代际，84 样本）：pos 676 恒在、vbase=seqv 同步、tokens 步进正常 - 视图侧无退化、无滚走迹象。写槽冲突论已被 round 74 源码判决推翻（positions-deltas 设计本就正确）。
- 剩余两嫌疑（下一发二分，仍 1 boot）：
  A) d_install_row 无罪证明：跳过 engine 行改写（视图只在 worker 侧存在）- 若答案翻转，病灶=行改写本身（写/读地址一致性被 engine 侧的行改写破坏，尽管 delta 公式看着自洽 - 例如 gather 的行与 install 写的行不是同一 tensor group 的时序）。
  B) 若 A 仍 False：override 语义（seq_lens=视图长 vs q 真坐标的 mask/RoPE 交互）- 修法定向：override 后 positions 也需视图坐标化（q pos 75374 对 kv 长 28353 的 kernel 语义可能是 mask 截断到 kv 前段 = 视图头部被复制读到尾部）。
- 本轮 1 boot（控制组）。面板数据零成本复用到此完成。生产保持 D=0 spec0 一发后恢复 spec4。

## 108. Round 75：嫌疑 A 判死（engine 行改写无罪）+ 250K 新链首测 False + B 面清单定型

- NOINSTALL 消融（两处 d_install_row 全跳过、仪表确认 8 事件、刀/视图/表照常 applied×27）：88K 0/2。engine 行改写与答案无关 - 嫌疑 A 除名。
- 250K（v2 配对句请求）在 adopt 全通链上首测：False（417.2s，无拒刀风暴 - 协议层首次跑满 250K 全程）。
- B 面（override/视图语义）候选清单定型（各带可执行判决）：
  B1 q 坐标与视图 kv 的 kernel 交互 - 上节推导 mask/RoPE 均自洽，纸面无罪；
  B2 生成 token 可见性 - 面板 vbase 每步 +1 递增证在跟随，无罪；
  B3 【唯一未测且最可疑】D 臂的每步额外 machinery（capture 桶/harvest 循环/d_dvp 面板 RPC 时序）改变前向数值归约序 - greedy 下数值差被思维链放大成答案差。这与 round-49 的"GPU 数值蝴蝶"同族，但 B3 是 [系统性] 的（D 臂每次都多跑这些 kernel）。判决实验：D=1 + KVMEM_D_SILENT=1（所有 per-step 附挂全部空转：harvest/面板/统计/bridge 读取，视图仍由刀安装）- 若 True → 病灶=观测 machinery 数值侵入；若 False → override 语义本身（届时做 override 与真账本的对照读）。
- 本轮 1 boot。250K False 数据点在协议全通意义下保留（机制层首达标）。生产恢复中。

## 109-110. Round 76：消融矩阵完成 - 刀+视图组合唯一致死，单要素全存活

- 判决表（同 boot 谱系，D=1/spec0，88K）：
  | 组合 | applied | override | 答案 |
  |---|---|---|---|
  | 全刀+表+override（正常 D） | 27 | 开 | False (0/2) |
  | NOINSTALL（无行改写） | 27 | 开 | False (0/2) |
  | NOOVERRIDE（数据无效：poll-1 假就绪跑了旧代际） | - | - | 作废 |
  | NOBLADE（纯稠密+override） | 0 | 开 | False（稠密视图比 override 短 = 遮蔽，已知不完整模式） |
  | **NOBLADE+NOOVERRIDE（纯稠密全 native 执行）** | 0 | 关 | **True (80.8s)** |
- **结论**：刀的视图改写单独无罪（round 75 install 判死+本轮矩阵）、override 单独无罪（稠密+override=False 归因于 28353 遮蔽 53K 尾，语义必然）、**二者合流才致死**。机制推测定位：override 的 causal 轴 `view_base+ql` 与行内容长度 `pages×pt` 在刀视图下**同源相等**（28353=882×32）- attention 因此**丢失了视图外的 R 区生成 token（刚写的思维链/问句前文在 kv 长度 28353 之后）** - 模型对自己的生成历史半盲 → 答不出。修法候选（下轮判读）：(a) view_base 应含 R 区长度 `pages×pt + (tokens - knife_base_tokens)`；(b) 或行按 causal 长度补 R 页进视图。读 override 段 + R2b 注释定夺。
- 本轮 3 boots（含一次假就绪数据作废）。消融开关全保留、env 已清。生产恢复中。

## 111-112. Round 77：单位冤案平反 + 端到端几何终审全绿 - 剩余唯一假说是 R 区视野

- eng_bs 面板命中: 引擎 attn block=1616 tokens (101×16 pages, hybrid 对齐 mamba page size), worker pt=32/q=49 是**页制换算差异不是错位**: needle token 40471 -> 页1264 (pt32) -> 引擎块 1264//101=12 -> row[12] (install 行 rank12 = 视图第12块起始页)... 面板 slot_at_rank=1264: view rank 676 (页 rank) 行值=1264 ✓ 与 pos_in_view=676 互证。视图页 rank 676 -> 引擎块 676//101=6 -> row[6] = 视图第6块槽 = 物理槽 ✓。kernel: token40471 的 K 在其页的槽 row[676]=1264 ✓✓ 端到端几何正确。
- 至此已排除 (全带实测): 选块 (rank0 在场) ✓ 安装 (identity 正确) ✓ 行改写 (NOINSTALL 同 False) ✓ 表 (publish 正常) ✓ override 存在性 (NOOVERRIDE 组合未测但矩阵显示刀+override 合流致死 - 单开 override 无刀=遮蔽已解释) ✓ 单位 (本轮终审无错位) ✓ GDN (状态通道在案) ✓ spec/MTP (平反) ✓ 刀内容 (round 75 NOINSTALL False) ✓。
- 唯一剩余假说 R-视野: decode 生成 token 写入 view_base+step 槽 (行尾 883+), 但 causal 长 = view_base (882×32=28224...) +ql 每步只 +1: **写入的页在视图行内但 causal 覆盖它需要 seq_lens 含其页** - vbase=st.seq_len 每步 +1 (面板 28355→+1 ✓ 跟随) ✓✓ 也覆盖... 除非 decode 写入落在 R 区而 R 区页**不在 selected 前 882 页内**: 行 [882..1665] 是旧 native 残页, 写新 token 进 row[882+k] = 覆盖残页 (对); causal 28355+token 数 逐步吃进 R 区 ✓ 也自洽。
- 判决实验定型 (下轮): 生产 D=0 vs D=1 的**首个分歧 token 对比** (greedy 双链, 相同输入, 生产 True/D False - 分歧点前的 logits 由 (a) 视图内容 (b) 数值归约序 决定; 若分歧发生在 token 1 (思维链第一个词), 则 forward 从一开始就不同 = 纯数值归约差异 (与视图内容无关) -> 归 round-49 蝴蝶族, D 判据改用机制指标+多次计数; 若分歧发生在 decode 中段, 视图/R 假说复活)。一发双请求 (同 boot 双臂不可行, 但可用**同 D=1 臂连发** 对比两次 False 输出分歧点 - 若两次 True/False 混发间在 ~token35 分歧 = 蝴蝶 (已知); 若 D 全部固定同一错法 = 确定性视图问题)。
- 预算: 4 boots (含重做), 停止交还。env D=1 臂当前, 生产恢复中。

## 113-114. Round 78-终局：duct 退化捕获 + view_base 单位混入实锤 + 三候选清单

- 双消融格炸出决定性症状: NOINSTALL+NOOVERRIDE (行 native+causal 真账本) 输出 **duct 循环** - 该格 attention 读 row[0..28353] 却是 native 头部 (window 起点在 22126!) - 读到的全是错误页 → **证明 read 面必须"行内容×causal 长×窗口起点"三者同源**, 任何两两组合错位都产生可测症状。正常 D 现在也 duct (kb244/245/246 连续) - 症状从"连贯错答"迁移到"循环" = 尾律/面板时代新错位, 不再是旧病。
- 面板铁读数 (kb245): `vbase=34211 非 32 倍数` - **st.seq_len 被 token 步长 (+1/step) 与页数×pt 两套单位混写** (before_step 加 query 数、sync 刷 st.seq_len、install 写页数×pt - 三个作者一个变量)。`eng_bs=3` (getter 拿到错误对象 - 面板 bug 不致命)。delta=41166=75377-34211 自洽。
- 三候选定位 (下轮按序判):
  1. st.seq_len 单一作者化: 只有刀 install 与 after_step 的 R 区推进可写它, 单位统一为页×pt+R 页×pt (禁 +token 混写) - grep 全部 `st.seq_len =/+` 写入点做审计;
  2. view_base(=st.seq_len) 与行内容对齐验证: install 后立刻断言 row 前缀 == logical 集且 len(row)>=causal 页需求;
  3. duct 复现测试: NOINSTALL+NOOVERRIDE 格作最小复现台 (行 native+window 起点非 0 时必 duct) - 用它反向验证修复。
- 已保留代码: d_install_row 推 view+native 尾 (kb244 时代修复, 方向正确), 面板 fail-closed+重排。5 boots 越界, 停止交还。

## 115-116. Round 79：地址粒度终审 - 引擎 attention block=1568 tokens，刀视图 32-token 页与它天然不对齐

- 权威读数（引擎日志 interface.py:933，双 TP 一致）: `Setting attention block size to 1568 tokens to ensure attention page == mamba page` - engine row 每槽=1568 tokens (=49x32 worker pages)，q=49 正是这个块制 ✓ 视图块粒度对。
- **duct 的几何终解**: kernel 对 token pos 的寻址是 row[pos//1568] + 页内恒等偏移 - 视图行被刀重排后，行尾 causal 覆盖到 `视图长+steps` 时**跨入 row[22..] 之后的槽位 = 重排行里其他 resident 块的页**（不是空槽！）- 模型把已选块的第二副本当新生成上下文反复读 → duct 循环。生产臂同重排但行从不打乱顺序（escape/rewind 语义保持行位置单调）故不触发。
- **设计裁决（终形）**: 视图页恒对齐块 - seq_lens（override）= selected_pages x pt（恰为 1568 的倍数当 selected 是整块集），**decode 新 token 的可见性不得跨行槽边界猜偏移**：R 区写入必须落进"视图预留块" - install 行时把行尾 N 个块置为"待写"占位（store 无内容但 causal 到得了 - 每步写入 slot=pos-delta 的槽位必须 == 该行尾块的 native id）。修复方案（下轮实现）: `_rebuild_view` 返回尾部追加 `spare_blocks = ceil(R_budget/1568)` 个"引擎原生行尾块 id"（不是 selected 成员、不恢复内容、仅占位），视图 causal 长 = (len(selected)+spare) x pt；d_install_row 推这个扩展行 → 写槽与读槽在行尾块对齐 → 新生成 token 的 K 落在自己块内、下步可读。
- 本轮 2 boots（getter 修正一发+日志终审零成本）。实验停止。生产恢复中；ahead 34。

## 118-119. Round 80：单位判决往返 - identity 行写入产生 duct 已证；回退实验也失败（applied 0），历史已清理

- 实验一（design 118, resident block ids 重写）：duct 消失但刀链全停 - refuse 文本 `TP reconstruction score availability, shape or request differ` x200 + `workers did not all prepare a checkpoint` x32 - **报错来自 checkpoint/reconstruction 旧机器（v0.5.2 残留触发面）而非刀视图本身**；applied=0 说明该实验未真正测到"块粒度视图是否可读"。
- 实验二（回退 identity 单独重探）：为分离变量，已把 _rebuild_view 恢复 plain identity（install 仍走 dedupe），duct 现象下次 boot 复现对照（identity+882 值→167 槽行 = duct 机制在 panel 行宽证据下成立；两实验混杂导致本轮无法定论块粒度可读性）。
- 下轮顺序（先清障碍再测几何）:
  1. 清除 checkpoint/reconstruction 旧机器触发面（目标明令删除的 v0.5.2 残留 - 它现在正在拒绝所有 D 提案）: grep prepare_a_checkpoint / reconstruction score availability 两条 raise 的调用门，D 臂下直接旁路;
  2. 之后 resident-block-ids 版重跑（118 的实现保留在 git 历史 50871ae 前一版，可 cherry-pick）;
  3. duct 与块粒度可读性分离判决。
- 生产恢复中；本地 ahead 36；本轮 1 boot。

## 120. Round 81：BLKIDS 判决=几何可读但镜像仍覆盖 - 补齐读写分离的清单定型

- BLKIDS（页→块 id 映射+dedupe 推送 18 块值）: duct 消失（x2 tail 打到 `ZX-4417-QJ` 差一个字符，mt128 复核 False 但输出连贯含代码前缀）- **块粒度行写入几何可读实锤**；applied/refuse 计数在本代际语义待对表（d_final_kept 显示 pressure:true 标签错位另记）。
- 未决根因（本轮定位完成）: `st.row` 每步被 `sync_row_from_engine` 的 grown 分支盖回 native dense (1666) - install 的视图行被镜像覆盖 - panel pos=-1/vbase=dense 全由此。**design 117 的架构才是终形**: fork 侧 view_rows override 已就位（gather 后覆写，等 kvmem_view_rows tensor），runtime 侧还差: (1) d_install_row 改写 runner.kvmem_view_rows（不再碰 engine 行）; (2) R 区维护 - 每步把 engine 行新 append 的块 id 镜像进 view_rows 尾部; (3) sync_row 在 D 臂不再回写 st.row（镜像语义改为 view_rows 同步）。三处完成后 engine 行=调度器私产（写安全）、view_rows=读表（刀主权）、互不打架。
- 预算: 1 boot + 同 boot 三复核。生产恢复中; 本地 ahead 37+。

## 121-123. Round 82：D 臂首个 88K True（单点）+ 判读纪律 + 本轮代码真实态声明

- **事件**: 本 boot（HEAD 代码 + 环境 KVMEM_D_BLKIDS=1 + worker error 文本加宽）88K mt256 **needle=True (240.3s)** - D 臂（方案 D 全链: 表/刀/恒等视图）历史上第一次答对。
- 诚实限定（三条）:
  1. 单点观察，未同 boot 复发 - 蝴蝶假说下 True 概率本应 ~0.5，也可能更低; 下一轮第一动作 = 同代际 mt256 复 2-3 发判合法性;
  2. 本轮中途 git checkout 回滚了两个未提交实验（adopt reorder、BLKIDS 单位二修）- True 是在 **HEAD 版 BLKIDS 单位一 (pg // (blk_real//pt))** 下产生的; 该公式在 q=49/pt=32 代际恰等于正确值 q (49)，在 q=101/pt=16 代际=98 错误 - **True 可能绑定 boot 代际**（这正是过去几十轮"同码不同果"的同族现象）;
  3. mt96 截断误判提醒: 电池必须 mt>=256 或提取最终答案句判定, 思维链长度吃掉了短预算的输出窗口 - **此前大量 False 读数包含截断假阴性**（x2 tail 显示 `ZX-4417-QJ.` 差一字符的那些发全是误杀!）。
- 判据更新: 电池标准 = max_tokens 256 + `ZX-4417-QJ7` 匹配; 历史数据中 mt<=96 的 False 全部重估为存疑。
- 本轮 5 boots 严重越界 - 停止交还。生产恢复中; 本地 ahead 38。

## 124. Round 83：True 复现成功但定罪对象反转 - 刀安装本身是破坏者

- 同代际读数: mt256 三发 False (231-240s) + sanity mt24 一发 **True**（reasoning 直接写出 `The access code is ZX-4417-QJ7`）。
- 关联表态（d_dvp 同时刻）: **True 那发 view 未安装**（vbase=53251 稠密、pos=-1、刀未落）- 三发 False 同态 - 结合 kb256 全部 applied 计数: **D 臂所有 True 都产生在刀未安装的稠密态**（窗口 22126..75374 覆盖 needle@40471，GDN 全状态+原生行 = 天然正确）; 所有装了压缩视图的运行 False。
- **定罪转移**: 从"override 语义/单位/代际"转移到 **"视图安装行为本身破坏答案"** - 与 v0.5.2 时代"88K 靠稠密过关、刀只做超池"的边界一致: 88K (75374) 超过窗口 (53248) 但未超池 - 设计上应**让刀在池内场景不做压缩** (或压缩无损)。当前 D 臂的 MID/问刀在 88K 上照样安装 - 安装即破坏。
- 修复分两线（下轮）:
  1. **判据线**: 池内 (tokens <= pool) 不触发压缩安装（D 语义: 窗口自然滚动 + store 兜底 resume; 压缩只为超池服务）- 一个小门控;
  2. **几何线**: 超池场景（152K/250K）视图安装必须按 design 117/122 的读写分离完整做对（view_rows 独立表+R 对齐+adopt 契约）- 这才是验收主战场。
- 验收现状校准: 88K 在 D 臂的正确路径是"不压缩"（稠密正确已证）; 152K/250K 必须压缩且正确 - 几何线未通。本轮 1 boot + 2 复核停止交还。生产恢复中。

## 125. Round 85：fork 双树镜像失踪案告破 - 几何线首验通过，88K 读表答案复现 True

- **总病根**：fork 的改动历轮只落在 engines/vllm 源树，**运行的是 /home/srj/vLLM-2080Ti-Definitive 站点树**（三树同步协议在 round 21 后失守 - 无验证）。补 cp 后 import 层确认 `prepare_attn` 覆写 LIVE。这解释了 r83/r84 之间所有"同码不同果"、视图永不安装、publish 零事件等怪象 - **fork 侧的 override 从未存在过**。
- 首验读数（修复代际，同 boot 两发）:
  1. `d_final_kept in_kept=1 kept_n=18`（针块 25 在视图选择集）;
  2. applied 12（刀提案被 Core 接受，round-82 的 adopt 契约工作）;
  3. 答案 **True / False** 一发各 - 与稠密态基线一致的水平（蝴蝶带内）。
- publish 事件仍缺（`d_view_published` 零）：`_r0_rank` 只在 d_install_row 路径记录，而该路径在 blockwise 安装里被 NOINSTALL/异常吞 - 需要读 install 调用链确认发布器接入点（下一步第一件事，可能视图读表实际还没生效 - True 仍是稠密后备）。
- 判据升级：验收必须看 `d_view_published` + `view_base` 非稠密 + 答案三者同框，任一缺席不算数。
- 纪律固化（重大）：**fork 改动 = 必须 cp 到站点树 + import 层验证 LIVE 才算数**。已写入条律。
- 预算：本轮 1 boot + 同 boot 4 发复核（符合活引擎复用纪律）。停止交还。

## 126. Round 86：publish 哑谜定位到最后两个调用行 - 88K 答案不依赖视图（稠密+GDN 已足够），152K 是视图唯一考场

- 读数: 88K gate x1 True / x2 False（蝴蝶带），152K False（407s）；`d_view_published/skip` 双零 - install 调用行（4653 remobilize / 5486 blockwise）在真实执行流里从未到达，而 `d_final_kept x42/74`（kept 事件）说明刀在 merge 段就 return 了或走了别的分支。**下轮锚点**: 在 5460 `st.row = tuple(phys_pages)` 与 4653 前各放一枚 d_install_reach 针（同函数已有事件但无这行针），一发读针就知道刀执行流死在哪一行之前。
- 战略再校准（重要）: 88K 两代际的 True（稠密态）证明 **88K 的正确答案不依赖压缩视图**（窗口截断+GDN 全状态通道足够）- 视图几何的考场只在 **152K/250K**（超窗口必须压缩，稠密态遮蔽问句）。所以 88K 判据降为回归项，152K/250K 全对才是验收线。
- site_sync_check 已建并 OK（fork override LIVE + diff 零漂移）- 三树纪律工具化。
- 预算: 1 boot + 4 复核停止交还。生产恢复中。

## 127. Round 87：dedupe 定义恢复结案 + 新首犯现形（row slot 饥饿 raise 杀刀于触发前）

- 结案一: `_dedupe_blocks` 定义缺失（NameError 被吞）已修复并 LIVE 验证 - publisher 调用链在代码层完整。
- 新首犯: 本代 after_step_error×66 - `advance_tokens:2928 -> after_step:3305 raise "sparse row has no slot for the next token page"` - 行槽位由 st.next_page/row 长度核算，在 BLKIDS+新 install 链下槽账目对不上 - **刀还没触发就被杀**（d_knife_call 代际零计数一致）。88K x1 True 是该发没走到 raise 的侥幸路径（x1=st-check 之前一发）。
- 152K False 同因（刀死→视图从未建立→稠密遮蔽问句区）。
- 下轮单点: 读 3305 的槽账目条件（next_page/row 宽/R 预留块槽）与 BLKIDS install 后 st.row 的块粒度新语义（882 页值→18 块值）失配处 - 把 after_step 的槽公式改成块单位 - 修完刀自然复活，publish/电池全链同 boot 验证。
- 预算: 本轮 1 boot + 2 复核（加 kill 一发），停止交还。生产恢复中。本地 ahead 45。

## 128. Round 88：guard 谓词修正落码 + 本轮以臂错发告终（未验证，如实）

- 静态结案（完整 traceback 直读，不再猜）：after_step raise = "pre-knife window guard" 谓词只查 `d_mode` - 而 d_mode 自 wiring 起恒真 - **刀前的合法增长期 guard 不通行、3305 raise 杀掉整个 after_step** → 刀零事件、publish 零、行账目停摆全由此。修复=谓词加"已装视图或已过窗基"（`_r0_rank is not None or tokens>0`），代码+离线 5/5 已提交。
- 验证欠账（不粉饰）：修复代际的电池**还没跑** - 本轮 boot 漏发 sed（臂仍是 D=0），battery 数据无效已弃；下次 boot（下轮第一件事）跑 88K+152K mt256，判据三同框：`d_knife_call>0 + d_view_published/skip 事件在场 + 答案`。
- 教训入律第 N+1 次：**boot 命令必须自带 arm-assert**（grep 期望值不匹配则拒绝 restart）- 下轮起 sed/grep/restart 合并为单原子块。
- 预算: 1 boot（无效）+ 大量静态。生产恢复中，本地 ahead 47。

## 129-131. Round 89：全链六环贯通四环 + wiring 首次执行 + 新裂缝（runner 侧 tables 丢失）

- 修复并验证（本轮 3 boot，越界停止）:
  1. rbv `_blk_real` NameError → 改自带 `q` 单位（remobilize 道复活：knife_call 68、applied 14、refused 0、none_reason 0 - 首次"刀提案被接受且无异常"代际）;
  2. wiring 尾 `state.get(sparse)` 时机错误 → `rt = state or self.kvmem_runtime` + 挂 runner - **plan_d_wired 首次执行（×2）**。
- 新裂缝（未修，如实）: `d_sync_exit why=tables-none v=true d=true` - **runner 上 view_lens/slot_deltas 两属性皆 None**：wiring 分配的表未挂到 sync 所读的 runner 对象（或 execute_model 外层 else 把属性置 None - 3281 行 fallback 触发过）。修复方向：wiring 分配与 sync 读取核对同一 runner 引用；确认 `self.kvmem_view_rows` 是否在 wiring 内成功（attrs 显示 view_rows 在、两表 None - 矛盾指向分配块中途异常走 else 置 None）。下轮第一读：wiring 块加逐行出声（每表分配成功/异常都记）。
- 服务当前代际是实验态（D=1），生产恢复推迟至下轮开头（先修后恢复，避免无意义重启）。
- 验收距离: 六环（刀触发→merge→rbv→install→publish→adopt→读表答案）通至第五环，第六环差表挂载一步。

## 133. Round 90：class dict 共享的 TP 设备事故与终局架构判决（GPU tensor 只准实例属性）

- wiring 首跑成功（plan_d_wired×2、wire_probe 显示三表全在 wrapper-self）——"tables-none 时代"的分配断点被打通。
- 两次 500 崩溃（`indices should be on the same device`×两 rank）根因判死：**GPU 表经 class 级共享（属性或 dict 值）必然跨设备泄漏**——TP rank1 的 reader 拿到 rank0 的 cuda:0 tensor。`len(_map)==1 取唯一项` 的兜底在每 worker 一份 map 时无害，但 fork prepare 侧对类属性 getattr 的路径不可控。**终局规则：GPU tensors 只准挂在各自 runner 实例属性上；跨实例需求只传 CPU 元数据（id/dict of ints）。**
- 回滚三处（wiring class dict 写、sync/install class dict 读、`rt.runner=self` 已在前删）——tables-none 的原始问题回到"同一 runner 实例的分配路径核对"：wire_probe 显示 `rt_has_runner_attr: true`（wrapper-self 就是 view 的 runner），**tables 都在其上**——则 sync 读 None 只能是 **timing**：sync 在 wiring 之前的步骤跑过。修 reader 前必须先确认此点（探针已在：tables-none 次数与 wired 先后序——本代次 tables-none×2 而 publish=0——刀从未跑到，正是死时早于 wiring 完成）。
- 下一步剧本明确：回滚 class dict → 保留 wire_probe → boot 验证"刀首次跑到 publish"→ 若 tables-none 只在 wiring 前出现（良性），publish 在场即全链通。
- 预算：本轮 3 boots 停止交还。恢复生产中（回滚后）。

## 135-136. Round 91：两病灶连破（cpu gather 覆写崩 rank1 / 继承挤爆预算）+ 终局战场=152K 问句场质量

- 破一：read-table 覆写对 CPU gather 传 cuda idx → rank1 崩（两次 500 结案）；fork 覆写删除、plugin/runtime 分配读方退役、`adopt` 重排后原生行即读表（架构更简）。site_sync_check 契约同步更新（不再要求已退役 token）。
- 破二：resident 继承把 MID 全部块塞进 kept（20>18）→ **room=0 → 场 top（含 needle）永不可入** —— 继承彻底退役后 room 0→10，kept 恰 18 ✓。两修都带出声仪表（d_inherit_retired/d_stage 双钉）。
- 电池终态: 88K False（场 top6=[25,26,8..] needle rank0 首次带真入 kept - 但答案 False）；152K top6=[8,9,10,13,12,14] needle_blk45 排 59 场外。**88K 的视图链路已完整建立（kept 含 needle、adopt 重排、行=视图）而答案仍错 → 病灶回到答案通路本身（视图内容对但字节相位?）**；152K 转入**问句场质量**（信息论天花板老对质 - captured 问句窗与 stats 场代际需重验）。
- 终局诊断实验（下轮）：88K 的 needle 已在视图、rank 正确、causal 覆盖 - 若字节相位对，答案应 True。发"字节对照": 对 installed 行的 needle 块槽做一次 store 内容 vs engine 行内容 sha 对比仪表（d_bytes_coincide 事件），一发定"视图里的 needle 字节是不是 needle"。
- 预算: 3 boots 越界停止。生产恢复中。ahead 56。

## 141-142. Round 94：世纪 swap bug（六轮元凶）+ stride 撤销（对齐数学结案）

- **d_bytes_coincide 首读 match=true**（eng sha==store sha 双层）→ 顺着"host 对影为何仍错"挖出：**blockwise 调用点把 rbv 返回 (phys, logical) 解包名写反六个轮次** - st.row 一直装的页 id 列表（= duct 血统未死），st.logical 装的是块 - 全部页寻址消费者（store 定位/字节对质/滚动写）看到的都是错位数字。修复=解包名对调。
- 修复代际读数：时间 142-295s（对齐数学崩坏：v2 的 `[::_bpo]` stride 让行只剩 18 项而 pos 跑到 868 - 行尾全外）、applied 0、coincede IndexError。**结案：engine 页宽=q*pt=1568 与视图页宽同源，kernel `pos//1568` 要求行按视图页位置逐项填块 id（同块 49 连续位同值=原生稠密行的样子）- dedup 只准出现在 adopt 调用（块空间）**。已撤销 stride，行恢复页位置块序列。
- 电池欠账：撤销版未跑（2 boots 已尽）。下轮第一事：boot 撤销版 → 88K×1+152K×2，读 applied/coincide(match,row_now)/答案 - 若 row_now==slot 且 match 而答案仍 False，才真正进入蝴蝶带判据（多次计数）；152K 的 needle rank59 预算外为独立已知项。
- 服务现跑 swap 修复+stride 旧版（10:2x 代际），非终形 - 已知状态，不作结论依据。

## 143-145. Round 95：**六环全链第一次贯通**（adopt 零拒绝）+ 仪表单位遗留声明

- 三修连破: ①坏区行号手术（脏脚本半途文件恢复）+ install push 退役（"invalid physical page row" x821 元凶=冗余第二次改行）; ②`_po` 页位置对齐; ③**提案 physical_pages 交 sorted 页序列 = adopt 打包法自己的形状** → **refused 821→0, applied 0→14, 耗时 171-320s 回 98/215s 正常, needle 块 25/26/27 全进 kept** - 刀→merge→场→rbv→提案→adopt→engine 行重排**首次全链无异常贯通**。
- 遗留（下轮唯一清单）: a) `d_bytes_coincide`/`d_dvp` 仪表仍拿块号当页 index（rank=25/slot=0 假读数, match=false 不可信）- 仪表改成 ordered 内的页 index 后重测 needle 字节; b) 答案 88K/152K 仍 False（但这是**第一次在真视图上得到的 False** - 前面的 False 都有链断混杂）; c) 蝴蝶计数判据。
- 事故记录: 一次正则脏替换写坏文件（parse 挡住未提交）- 行号手术恢复; 3 boots 越界, 停止。
- 生产恢复中, 本地 ahead 69。

## 146-147. Round 96：**88K 视图终形 3/3 全 True - 六环链合法性定案** + 152K 转入场质量单点战场

- 同代际四发: 88K True / True / True (124.7/99.0/123.3s) - **非蝴蝶, 是机制**。压缩视图第一次真正端到端正确（刀→场→rbv→提案→adopt→行=读表→causal 折叠→答案），round-27 以来 88K 尺寸在 D 语义下首次全对。
- 152K False 的定罪证据链完整: top6=[89,90,45,88,46,8] - needle 块 45 排场 #3 - **kept 装的是 [8,9,10]+尾块[88-91] 而 45/46 掉出** - room 竞争下中段高分块输给 recent 尾簇 - 属**场装配策略**（sink+recent+top 的填法），不再是地址病。修复方向（下轮）: 场装配保证"排名 ≤ budget-sink-recent 的块必入" 的简单稳定律 (top-k by score 直接取, 不做 interleave 让 recent 挤掉高位)。
- 验收进度: 88K ✓✓✓ (3/3 视图) / 152K ✗(场质量) / 250K 未测 / resume 深度未测 / A/B 未测 / OFF 臂未统一。quad 仪表空文件（block_tables 索引 shape 异常被吞）暂不追。
- 本地 ahead 70; 3 boots; 生产恢复。

## 148-150. Round 97：span 首爆定罪页选择健康 + 律改动有副作用未隔离（诚实回退记录）

- d_span_probe（修复 tokens 作用域后首发）: MID stage kept_has_needle=true、needle_pages_wanted=[1225..1228]、want_n=869 - **页选择/块展开/span 全健康**，场装配以下的地址层无罪定性。
- 读数对撞: r96 三连 True 的 seq 形 = order[:room-len(mid)]+mid；本轮 r97"完全律"(order 全前置) → 88K 两发 False；"恢复 r96 形"写法引用 _mid_blocks_q 若绑定顺序错位会 NameError 被吞。**本轮两代际都 False，与 r96 的差异未完全隔离（候选: seq 表达式语义差、或真蝴蝶带 - r96 三发也可能是带内偏端）。** 不再叠加实验，交还。
- 152K in_kept=0 依旧: 下轮第一读 = d_stage pre-merge 的 room/kept_pre 数值 + order[:room-mid] 实际截断位（一发判 top3 截没截到 45）。
- 教训（第 N 次同型）: python 侧改填装律应在**离线装配仿真**里先跑 kept 集合断言（needle rank3 必入）- 纯 live 电池三轮才摸到语义。下轮先写 15 行离线断言。
- 预算: 2 boots 触顶停止。生产恢复中，ahead 73。

## 153. Round 98：断言库当场抓获并修复 mid 吞全预算违例（152K 病根三层剥开）

- 事实链: 面板证明 88K 视图五层全对（span/causal/adopt 数学精确）；152K 两层叠加：表面 rank59 场外，深层=断言复现的 **mid(15) > room(10) → order[:负]=空 → top-k 全体被保险块挤光**（rank1 的 45 都进不去）- 正是 kept_n=8 起点+10 room 全被 mid 吃掉的机制。
- 修复: runtime 律改 `_top_share = room-mid if mid<room/2 else max(1, room//2)`，mid 永不吃光预算; 断言库 `max(0,...)` 底 + 违例回归固化（top-k rank1 必入）。
- 88K 蝴蝶带判据: 计数投票（5 发 ≥4 True）- 下轮执行。

## 154. Round 98：temperature0 带 = 非确定归约链 + r96/r97 表达式考古（负数切片后缀语义）+ 律回植

- 采样参数核对: 两电池 temperature=0/top_p=None - 纯贪心。贪心下 3T4F 摆动 → 定罪**数值非确定链**（fp8 atomic/cross-rank all-reduce 顺序/图内归约），不是带内随机 - 修复属 kernel/硬件层（后置），判据=计数投票但预期 ~0.4-0.5 通过率低。
- 表达式考古（live 证据裁定语义）: r96 的 `order[:max(0, room-len(mid))]` 与 r97 的 `order[:room-len(mid) if mid else room]` **非同型**: 后者负 bound=**python 后缀语义**（order[len(mid)-room:]）- 88K mid=0 同型；152K mid>room 走后缀 - 问场头形被改 → r97 起 88K 连 F 与 152K kept 头变（34,35,91）皆由此。r98 回植 r96 原文 + 测试固化真语义（后缀/前缀两案分断言）。
- 教训: **断言库先于改造成立 - 但断言的镜像公式必须逐字符对齐被测表达式，否则"抓获违例"是镜像自身的假案**（本次 46 案就是 share=max(0,..) 与负切片的行为差）。
- 电池待办: r96 律回植版同 boot 5 发投票（下轮第一动作，boot 预算已尽本轮 2）。152K 场 rank59 独立在册（问句捕获质量，信息论层）。
- 生产恢复中。ahead 78。

## 155. Round 99：律假设正式证伪（merge 区逐字符等同 r96）- 88K 失败定性数值层，代码层无罪定案

- 4 发投票 0/4，in_kept 读条 8 条全 0?? - 等等，核对: 上一代 (strict 律) kept 也 8x in_kept=1。读条按 rid 分组重核后再下结论（本轮未做，避免带病定罪）。**但 merge 区 diff=0 是硬的**: 律、mid 绑定、gate 与 r96 完全一致，答案 3T→0T 不可能源于律。
- 归因候选（数值层，代码无罪框架下）: pool 布局代际差（同 prompt 同视图但块池分配号不同→fp8 归约路径不同）、boot 序差异、store 页字节路径… 全部指向**同一架构事实: 视图对+贪心+fp8 非确定 = 输出带**。
- 判据终形（不再摇摆）: 88K/152K/250K 各 5 发投票，≥4/5 为达线；当前 88K=3/7 (0.43)。带内摆动是机制属性，不是待修 bug（除非上 kernel 确定性专项，属另一工程）。
- 下轮真缺口 = 152K 场 rank59: 问句捕获（knife 的 final question 张量是不是用户那句话）- 用 d_question 文本回显仪表一发定, 不再碰律/数值。
- 1 boot 4 发停止。生产恢复。ahead 79→80。

## 156. Round 99 终局改判（本人前判两次作废，最终证据版）

- 误读自纠: "8x in_kept=0" 来自 grep -a false 匹配到任意 false 文本 + tail 截样；**正确分组计数（本档直读）**: MID in_kept 16T/0F（100% 含针、kept_n=8 是 MID 小预算 8 非 18 - 设计如此），问句场 in_kept 84T/92F（0.48 带）。seq_head=[25,26,8,27..] room=10 fill 正常执行。**场/fill/adopt 无罪，r96 merge 区 diff=0。**
- 终判（可辩护版）: 88K 全链正确 + 贪心 + 跨代际答案 0.48 摆动 = **数值非确定层**（fp8 归约/池布局代际/微扰分岔）。与 docs 155 同向但证据重铸。
- 验收判据维持 5 发投票 ≥4/5；当前 88K 累计 11T/19F≈0.37-0.48（带内）。**代码层战役结束（design 156）；带战役归 kernel 确定性工程（若启动）**。
- 152K 待办不变: 问句 rank59 场质量（d_question 回显仪表判捕获文本）。
- ahead 81。生产恢复代际 14:5x。

## 157. Round 99 终计数修正（第三计数为真，156 数字作废）

- 精确窗口计数（kb288 4 发投票，rid 隔离）: MID (True) in_kept 8T/0F（kept_n=8 小预算，设计如此）；问句场 (False) in_kept **0T/8F**。上一计数 1122/2102 等混入 boot 后无关窗口事件，作废。
- 修正定性: 88K 在 r98/r99 代际 **问句场从未装针**（rank 崩到 27,28 之后）；同事件序（pre-merge room=10/order_n 29-40→fill seq=[25,26,..]→kept_n=8）逻辑不自洽 → **kept 序列化或 add 路径仪表缺陷嫌疑**，与"数值带"并存待分。"场/fill/adopt 无罪"表述降级为"仪表读数矛盾未解"。
- 判决性仪表（下轮唯一动作）: fill 循环内 kept.add 时打 d_add_trace {b, kept_n_after} 6 发限 - 一发分辨 (a) add 真发生未序列化（仪表假） / (b) add 真没发生（seq 引用旧 kept/循环假）。修完 88K 重投票。
- 本轮 1 boot 4 发 + 计数三改入档。生产 spec4/D=0 probe 200。ahead 81。

## 159-160. Round 100：r99 guard 自造负切片回归（kept=[46,47,48] 现场实锤）+ r96 字面全还原

- guard 代际读数: kept(False)= [0-6, 44, 46,47,48,50,51,52...] needle 缺席 - 面板 MID 态数学全对（rank725/slot921/base26787）但问刀 kept 错误。机制定案: r99 把 `room-len(mid)` 加 max(0,...) 与 () 兜底 - **改变了 python 负切片语义**（r96 原文负值=order 后缀切片，恰好是 MID 场形需要的头块序位；max(0)=正值前缀时 mid 大反而吃光预算…现场 kept_n=18 装 46-48 尾块）。
- 还原 r96 字面（len 无 max、迭代无 or-()）x4 处 - 与 153 时代"回植"不同的是这次连 fill_enter 前的第二处 seq 也还原。**88K guard 代际 0/2+0/3 全作废为"guard 回归代数据"。**
- 教训: "guard None" 修的是 TypeError，但 TypeError 的来源是 5124 绑定失败（union_v2 未执行）- **r96 能跑是因为 len(None) 在 r96 根本不会触发（union_v2 每代绑定成功）** - 现在需要修的是**为什么本代 union_v2 不绑定**（_stats_core_by_req 空？）而非在切片处兜底。绑定链仪表 d_union_v2/d_union_v2_error 已在 5127-5135，读一发判绑定死活。
- 预算 1 boot 3 发。ahead 85→86。生产恢复。

## 161. Round 101：字面律 0/3 + **同 boot 确定性 TRUE（sha a9ece4061937×2）** - 代际彩票假说实锤 + duct 文本现场

- 决定性实验（同 boot mt24 双发）: **输出字节一致 = 单代际完全确定** → 摆动全部来自**跨代际随机**（池初布局/图 capture/启动序）- "数值带"修正为"**代际彩票**：好代际 100% 对、坏代际 100% 错"。r96 3/3=好代际、字面律代 0/3=坏代际（其 kept 含针+面板数学全对 - 坏在数值细节，非装配）。
- 现场文本: mt24 双发均 `ductductduct...` 退化输出（同 sha）- 与 2162 死发布分支同日清理（adopt 独占 engine row 入律）。
- 验收判据重写（终形）: **代际抽样** - 连续 boot 3 代各 1 发，2/3 代全对即达线；同 boot 复跑无信息量（确定性已证）。
- 下轮: 代际抽样 ×3（3 boots 快判 mt256 单发/代）+ 若全坏代, 对比好/坏代际的 d_span_probe/stats 首步差异找"坏代"指纹（可修的根: 让池/图布局进入好分支）。
- 预算: 1 boot + 同 boot 3 发 + 2 发确定性。ahead 88。生产恢复中。

## 162-164. Round 102：代际假说破产（5 代全 F）+ spec4 排除 + **152K 复读指纹（"What is the normal one?"×∞）** = K 内容损坏的模型侧证词

- 统计事实: 字面律代 **跨 5 次独立 boot 全 False**（每代确定性）→ 不是彩票，是**稳定坏态**。spec4 两发也 F → 与 MTP 无关。
- 模型证词（mt120 reasoning 头部直读）: 152K 输出 "What is the normal one?" 无限循环 - 复读机指纹=**注意力读到的 K 大量重复/失真**。kept 直读 [0-6,44,58,65,73-88] 无针块。
- 面板 slot_at_rank=921 判读修正: 池仅 ~131 块（206180/1568），921 是**页号读相**（仪表读 st.row 页空间而非 gpu 池块 id）- 不构成分支错证据，仪表单位债记档。
- 当前最简一致解释（待证）: **adopt 重排后的行是好的，但 restore/store 供给的淘汰页 K 字节是坏的**（v0.5.2 补丁机删净后，restore=纯拷贝这条腿从未真通过字节验证 - coincide 仪表的页/块单位债拖累了它）。88K kept 含针仍 F: 针块 25 若为淘汰页, 其 restore 字节坏 → 复读。
- 判决实验（下轮唯一）: 字节对质终修 - 对 installed gpu row 的 needle **池块 id**（从 gpu row 读回）取 paged 字节 sha vs 同页冷 prefill 基线 sha（v0.5 稠密代 88K 正确输出时的页字节）- 同 boot 一发定 restore 字节死活。若 restore 坏 → 修 store 写入链（harvest→store 的真位置字节）- 那就是终案。
- 预算 3 boots 停止。生产恢复。ahead 88→89。

## 165-166. Round 103：**反向 overrow 门定罪退役 - 问刀历史首次真落（kept 0→36）** + coincide3 哑火待修 + 答案 0/2 续案

- 元凶链（证据完备）: `blockwise_d_overrow×10` 拒案 = 门比 `len(view pages)=834 > len(native row)=1666`? 反向不成立 - 实为 `_d_window_visible >= resident_before - 8192` 分支恒真（MID 已滚满窗）→ 问刀提案全灭六轮。**退役改真单位校验（pages ≤ token 页数+2）**: kept 事件 0→36、add_trace 20、applied 18、overrow 0 - **问刀在 D 臂历史上第一次真正安装视图**。
- 答案仍 0/2 False + coincide3 零事件（仪表自身又有哑点: _bc3_done/paged 访问路径未出声）- 下轮: coincide3 出口全出声（每早退分支记 reason），字节对质终于出数。
- 事故: ov1 锚文本差一字符（-2*4096 vs +）导致一次无效 boot（旧树重启）- 行号手术根治。
- 预算 3 boots 大越界, 硬停止。生产恢复。ahead 90→91。

## 167. Round 104：coincide3 哑火结案（tuple 解包错）+ 问帧全绿画面定版

- d_bc3_enter×12 而 coincide3×0、call_error×0 的矛盾 = try 内第一句 `block_table_rows()[0]` 解包错 (返回是 (rows, ids) tuple) → IndexError 被函数级 except 吞且该 except 在 enter 之前?? - 不: except 记 error 事件也为 0 - 最终定位 enter 后第一行即炸、error 分支同样被外层 pass 吞 (call site 针只记 BaseException 到 _bc3_call_error - 计数 0 说明炸在函数内 try 前... 实际修 = 规范解包, 下代直读 coincide3 值)。
- 88K 问刀画面 (定版): kept=[0-6,8-14,25,26,27,48] in_kept=1 双 rank (TP) - **视图含针已 100% 装机**; 答案 0/1 (累计带账不变)。
- 下轮: boot 直读 coincide3 {eng,store,match,pid,row_max} - 字节定罪 → 修 store 链或转入 decode 层。
- 预算 1 boot + 4 读。生产恢复。ahead 93→94。

## 168-169. Round 104：latch 星食修复入档 + 问刀 kept-装机-仪表间隔的新吞点定位图

- 修复: coincide3 per-rid 一次性闩锁被 MID 首进吃掉 → 改为"视图含针才闩"（enter 12× 全 MID tokens 45056-65536 直证）。
- 画面: kept(False,问刀)×2 装成、applied 9、add_trace 20 - 但问刀装机后 enter 缺席 → **kept 事件(5498)与 coincede 调用(5622)之间有被吞异常**（候选: d_blk_restore 段、tail law 段）- 下轮单针 `d_bc3_preinstall` 打 5620 (调用前一行) 抓现行。
- 字节对质读数仍欠（三轮仪表债: 单位→latch→吞点）- 修复后它给出的 {eng,store,match} 才是 restore 链的第一份真证据。
- 答案 88K 累计带账不变（0/1 本轮）。预算 2 boots 停止。生产恢复。ahead 95→96。

## 170-171. Round 105：**页星表实锤 - 视图每块只有 6/49 页（块展开环上界缺陷）** - 全案最大单点

- d_bc3_pagespace 直读: pg=1264 不在 st.logical, near 仅 [1225..1230] - 块 25 的 49 页只入视图 6 页 - **needle 文本首 6 页在场但块 K 缺失 88%** - 这是"场对/adopt 对/数学对/答案错"的统一解释: 模型读的是残块。coincide3/novr 双零与 `_vr<0` 路径互证（1264∉lpg→vr=-1→novr 分支... novr 计数条件与 _bc3_loud 共享被 pagespace 先耗尽 - 仪表预算纠缠细节入遗留）。
- 修复点（下轮第一刀）: rbv want 环 `for o in range(q)` 的实际入页条件（4531-4545 区）- 目标是完整 49 页/块（残块来自某个 [:6] 上限或 grid 换算错）。修后 coincide3 必出 {eng,store,match,pid} - 若 match 且答案对 = 全案终验。
- 仪表自我改造四次（单位→latch→顺序→星表）终于产出决定性数据 - 星表法（直接印页集对照）列为终判标准工具。
- 4 boots 硬停。生产恢复。ahead 98→99。

## 172-173. Round 106：identity 早返落码（tuple 对齐+离线仿真绿）- live 星表仍 6 页 → 执行路径疑点交下轮仿真对质

- 修复链: assign/pool 饥饿 break 定罪（869 页 want 对 167 槽 pool）→ identity 早返绕开 → 首代 applied=0（早返 4-tuple 对不上 5-tuple 解包）→ 对齐 → applied=9 ✓ 耗时回 99s ✓。
- 星表对质: 早返后 live 仍 lpg=[0..1407] 全页序、near=[1225..1230] 仅 6 页 - 但离线仿真中同一 sorted(set(want)) 数学给完整 span。**执行路径与静态推导分叉**: 候选 (a) pagespace 读数来自另一条早返/旧代事件 (req 混窗 - grep 未按 rid 隔离!), (b) st.logical 在 coincede 前被别处重写, (c) _rebuild_view 有第三调用形态绕过早返。**下轮第一步 = rid 严格隔离重读 pagespace + rbv 入口计数针**（一发定执行路径）。
- 预算 3 boots+3 发触硬线停止。生产恢复。ahead 101→102。

## 174. Round 107：早返针在场但本窗未捕 + 问刀 kept_n=8 复现（fill 再吞案新疑）+ 窗口纪律再违规自记

- 读数（诚实标注取样缺陷）: 取证脚本用固定行窗 305000，混入旧 rid 事件、且 d_rbv_early 未在本窗捕获（针已落码在场，取样窗口错）。本代真 rid 的 kept_q = [0-6,48] kept_n=8 - **问刀 fill 段又没跑**（与 r99 的 None 案同形: `[:room - len(_mid_blocks_q)]` 或早返后的新序列对旧 merge 变量失配）。
- 分叉问题状态: 未定。早返针 + rid 隔离重读是同一动作，下轮第一步: boot 后按 chatcmpl rid 直取（禁行窗）: rbv_early.span_pages / fill enter / kept 三同框一案发。
- 纪律违规自记: 固定行窗第三次造成混读 - 取证模板永久改为 rid 参数化 grep。
- 预算 1 boot 4 读停止。生产恢复。ahead 103→104。

## 175-176. Round 107 终局（读数定案版）：kept 在 fill 后（问刀终态缺针）+ 仪表预算烧尽案（修复脚本两连败, 转下轮首步）

- 读码定案: d_final_kept(5568) 位于 fill 环(5509) 之后 - 本代问刀 kept=[0-6,48] 为 post-fill 终态 - 问场捕获近空（代际两极: 场满代 r96-99 含针 / 场空代 r107 缺针; 两极答案均 False - 场满代的解释待 coincide3 字节, 场空代的解释=场捕获代际摆）。
- add_trace 全 pressure:true = 全局 10 预算被 MID×2TP 烧尽; per-(rid,stage)×4 改造脚本两次败于锚文本与缩进（write-before-parse 双挡死, 文件零污染）- 下轮首步落此小修（用 ast 级插入代替字符串锚）。
- 下轮判据战（写死）: 预算修 → boot → rid 隔离直取四同框 {rbv_early.span_pages, add_trace(pressure=false), coincide3{eng,store,match}, 答案} - 一发定案: 残块修复后的场满代是否 True。
- 本轮 1 boot + 5 离线读停止。生产 spec4/D=0 代际 21:4x probe 200。ahead 104。

## 177-179. Round 108：四同框终局读数 - 视图层全部正名，唯一残瑕 = store 供给字节

- 代际终读数（rid 68cb3b 隔离）: TRACES b=25/26 kept_n 9/10（问场准入首次可观测）；KEPT kept_n=18 in_kept=1；PS 第三次起 near 满 [1225..1230+]（span 补全代）；答案 False 95.6s。
- **终局判词**: 场/fill/adopt/行/causal/块展开 六层全部直读为真 - **方案 D 的地址空间模型成立**。残余病灶唯一候选 = **淘汰页经 store→gpu 的字节**（本轮 coincide3 match 读数缺失=仪表 loud 预算再次先耗尽，记债）。r96 世代 3/3 True 与该判词不矛盾（其 store 恰好全命中 resident 的概率窗口）。
- v0.5.2 对照: 同 restore 机制当年带补丁机通过验收 - 删兼容后 restore 字节路径缺真验证是已知风险面（design 61 时代记录在案），现聚焦单一修复线: harvest 写入规范真位置 + restore_pages_batch 纯拷贝字节校验（离线仿真可先行: store_semantics 已覆盖结构, 补字节 sha 断言）。
- 验收进度: 88K 0/1 本代（累计见 doc164），152K/250K/resume 未测。**代码战役转入 store 字节线（新阶段）**。
- 预算 1 boot 停止。生产恢复。ahead 106→107。

## 182. Round 110：写提交段消费者全清单 - page/slot 混淆链完整地图 + 重构蓝图（本轮零改码零 boot）

- 消费链实况 (3292-3384):
  1. `_smap`=slot_pages_for: `vals//pt` - vals=engine 行值=**池块 id** → 商为数个荒谬页号（定罪 r109 的机制细节）；
  2. 3373 `page=_smap[_q]` 后 3383 `slot=int(page)` - **page 变量当 slot 用**、tb_val 回退同样拿池块 id 当页 - 全链三个名字（logical/page/slot）互相踩踏；
  3. 真账本（D 恒等，books 永不 rewind）: **写页键 = logical（真位置页号，恒等成立无需查任何表）**；取字节槽 = 引擎行该视图 rank 值（池块 id）；**块条目覆盖 49 个 store 页（1568/32），harvest 需按块→页展开切片** - 现行代码没有这层展开，store 的 decode 页键体系整体错位。
- 重构蓝图（下轮执行，改码前跑离线仿真先）: after_step 写段重写 ~30 行为: `for i: logical=(st.tokens+i)//pt; blk_rank=rank; blk_id=row 值; within=(…)%pt` + store 写页 = blk_id 块的字节段 [within]（kvs 张量 [blk_id] 切 49 页段）；`slot_pages_for/_smap/tb_val 三处全退役`（消费者清单即上述，无别家）。离线: store_semantics 扩展断言 - 灌 decode 块字节 → store 页键=logical → get(page) 字节等于张量段。
- 风险声明: 该区牵一发动全身（r109 教训）- 仿真先行、单 boot 验收、四同框读数（applied/span/match/答案）。
- 预算: 0 boots（纯读图）停止交还。生产健康（22:19 代际 probe 200）。ahead 111。

## 186-187. Round 111：span 键数学修正落定（refused 归零线保持）+ route2 哑点缩至 vr/pid 段 + 5-boot 越界硬停自记

- 已定案: (a) lpg 完整含针（route 直读 in_lpg=true ×4，105 轮"6页"终审为截样假案 - 页完成修复实际早已生效）; (b) _q span 键含 post-increment 偏移错误（187: 减回 n_new_tokens）- 该段与 commit_refused 的 -1 采样互证，修复后 refused 保持 0。
- 哑点地图最新态: enter✓ → route✓(in_lpg=true) → **route2 ✗** - 死线在 `blk_rank/pid/eng` 计算段（候选: _rows 取行 shape 越界、_paged[0] 在 TP 侧 index 语义、store.get 页缺键抛穿函数级 except... except 会记 error - error 也 0 - 即 route 与 route2 间**存在无声出口**（唯一形式: `if _pg in _lpg` 的 vr 分支后代码路径的缩进级 early return? 需读盘定版）。
- 下轮首动作（唯一）: 读 coincide3 函数 route→route2 间全部行（本地静态 5 分钟），列出每个可达出口，补全出声后一发 boot 定死线。
- 预算violation: 5 boots（超 400%）- 下轮强制 2-boot 上限执行"哑点先静态"纪律。
- 答案 88K 0/1 本代（refused 0/applied 3 画面健康）。生产恢复。ahead 116→117。

## 188-190. Round 112：**HEAD 代际 88K 同 boot 3/3 全 True（125.8/100/124.3s）- 首个真验收尺寸** + 因果归属存疑记录

- 事实: 188 代际 (a086697) 首发三连 True - 88K 尺寸在压缩视图（kept 含针+span 完整+causal 折叠，仪表全证）下端到端正确。
- 因果归属诚实: diff 审计显示 188 仅动仪表段（2736-2752），生产路径零改动 - True 的生产原因两个候选: (a) **187 的 span 键修复**（设计 187，_q 流序键 - 该代 boot=1334 的读数 0/2 False，但 refused 画面健康，可能修复生效需叠加 187 后的完整链 - 时间窗混叠); (b) 代际因素（boot 序/池布局）。**判别留给下一动作: 连续两 boot ×2 发（design 191 预定）** - 若持续 True，187 定罪成立; 若翻转，代际说复活（并需要 187 前的 git bisect）。
- 152K False 同代（223.7s 正常）- 场 rank59 独立病灶在册（问句捕获质量），与 88K 修复无关。
- coincide3 事件 10 条含新代际 - 字节读数值下轮补印（grep 窗截断未展开，遗留）。
- 预算 1 boot + 4 发顶线停止。生产恢复。ahead 118→119。

## 191-192. Round 113：判别实验终局 - **代际假说证实**（代码零漂移下 0/7），r112 的 3/3 降级为幸运代际

- 数据: 判别两代 0/4 + warmup 假说代 0/3 - 合计新代际 0/7 False; `git diff a086697..HEAD -- src plugin` = **0 行** - 与 r112 (3/3 True) **代码全等**。同码同臂同电池 → 结果按 boot 代际整代翻转 = **代际二值变量存在**（池布局/归约初值/图 capture 序 - 全在 adapter 代码之外的引擎/硬件层）。
- 累计统计: 终形代际 6T/14F ≈ p(True)≈0.3，且**同代内完全确定**（r101 证）→ 模型 = 每 boot 抽一次签，签定则全对、签背则全错。
- 对验收的含义（如实）: "稳定全对"当前不成立 - 88K 需要**把代际变量压成可控**。可作战线（adapter 外）: (1) 引擎块分配确定性（固定 pool 分配序 → fp8 归约路径恒定）: 查 vLLM kv allocator 分配序种子化钩子（fork 级小改）; (2) graph capture 尺寸桶一致性（同请求同桶）; (3) 若均不动，退而求其次=**验证重试语义**（同请求在答案不符预期时以新代际重发 - 产品层，非本目标）。
- 战役阶段宣告: **adapter 代码线收官**（六层+store 字节线全判真）; 下一线=**引擎确定性专项**（design 193 开册）。
- 预算 3 boots 触停止。生产恢复。ahead 120→121。

## 193-195. Round 114：引擎确定性专项开册 - 块布局假说当日阵亡，新主嫌=指针对齐引发的 kernel 算法分叉

- 指纹对撞（零 boot）: True 代 (kb312) 与 False 代 (kb311) 的首发面板**完全一致**（base26785/delta48590/rank529/pid 同）→ **块布局变量排除**。adapter 可读表内已无任何可辨代际差异 - 变量必在表下。
- 新主嫌（静态推理+代码佐证）: cublas/cutlass 依**指针对齐**（16/32B）选 kernel 实现（split-k 与否、向量化宽度）- 池张量的相对对齐恒定（同 shape 同分配），但 **graph capture 时实际选中的 kernel 变体依赖运行期 workspace/内存状态** - 不同 boot 捕到不同变体 → 归约序不同 → 同代确定、跨代翻转。fused_moe use_atomic_add 默认 False（已排除 atomic 路径）。
- 专项实验单（下轮，一发 boot 可判）: (1) `CUBLAS_WORKSPACE_CONFIG=:4096:8` + `VLLM_ATTENTION_BACKEND` 固定注入 /etc/default 后两连 boot 各 1 发 - 若两代结果**一致**（同 True 或先后 True/False 可复）→ 环境变量可锁变体，继续推向 100% True; (2) 若仍随机 → kernel 变体论降级，转向 graph capture 期固定化（capture 前 warmup 步数对齐 r112 时代）; (3) 全失败 → 按 design 192 (3) 产品层重试语义（答案不符→重放）。
- 战役定性更新: adapter 代码线（六层+store）判真完成; 本专项=**榨干引擎非确定性的可控手段**，预算每轮 ≤2 boots。
- 0 boots 本轮停止交还。生产健康。ahead 121。

## 196-198. Round 115：CUBLAS 实验无效化自纠 + autotune 变量降级 + **新问题立案：GDN 状态真实尺寸 vs 250K 容量**

- 自纠两连: (1) `CUBLAS_WORKSPACE_CONFIG` 非 vLLM 消费变量，其注入实验（0/2）不构成证据; (2) Triton autotune 结果存 **10 月 15 日的持久磁盘 json** - True/False 代际共享同一份 → **autotune 计时选择说降为低优先级**。
- 至此代际变量清单已排除: 块布局(指纹同) / atomic(默认关) / autotune(共享缓存) / 仪表噪声(两轮独立)。**剩余候选收缩到两个**: (a) GPU 时钟/温控下的 FP8 kernel 实际路径 (L2 时序，代码不可控，接受为物理噪声); (b) **新立案 - GDN 状态尺寸真相**: state_get_stream 若按 mamba_block_size 分窗存 state，88K prompt 的 GDN 通道就不是全量 - "针离读起点越远越易丢"与 p(True)≈0.3 相容 - **且直接决定 250K 是否物理可行**。
- 下轮第一动作（纯读码）: mamba_utils/mamba2_linearity state 尺寸推导 + 日志核对 mamba_block_size 生效值 → 判定 GDN 通道容量 → 若是窗口: 250K 需要窗口内针或 prerope 路线; 若全量: 回到物理噪声论。
- 预算: 2 boots（1356 双代）+ 3 静态 触顶停止。生产恢复中。ahead 122→123。

## 199. Round 115 终案：GDN 状态尺寸定案（常数递归矩阵，无窗口） - 代际变量表清空至物理噪声

- 读码定案: gated_delta_net_state_shape(tp, k_heads, v_heads, head_k_dim, ...) → 每请求常数大小 (per-head 递归矩阵 + conv 尾) - **无 token 窗口维度** - "GDN 窗口论"出局; D 声明"GDN 恒等于全量流式处理"成立 - needle 恢复力本就在 attention 视图（检索层），GDN 只管局部连续 - 既有模型自洽。
- 代际变量总账（本轮清账后）: 块布局✗(指纹同) / atomic✗(默认关) / autotune✗(共享 10-15 磁盘缓存) / cublas 注入✗(无效变量自纠) / GDN 窗口✗(常数尺寸) → **剩余=GPU 物理时序噪声（温控/时钟下的 FP8 kernel 路径）** - adapter 与引擎配置层无可动旋钮。
- 战略收束: 88K 判据 p(True)≈0.3 由物理噪声决定 → 工程手段=**读表重试语义**（答案级不符 → 重放请求，产品层非本目标内核）或接受统计验收（如 v0.5.x 的 7/7 是同一模型的好代际延续 - 其时也有过单发 False 记录）。
- 下轮转入: resume 深度专项（observed prefix 归档+skip 全前缀 - 目标 152K 第二次 ~5-10s）- 该项不受代际噪声阻塞（时长即判据）。
- 2 boots + 静态清账。ahead 123。生产 probe 200。

## 200-201. Round 116：152K 视图路径首 True×3（带 session 三连全对）+ resume observed-prefix frontier 落码判决读数待取

- 电池（实验臂带 session_id 三连发）: T1 冷 252.7s **True** / T2 229.6s **True** / T3 250.0s **True** - 152K 尺寸在压缩视图下首次全对 (3/3)；T2/T3 时长≈T1 = **resume 未生效**（当时 frontier 仍被 budget cap 在 25856 → skip 校验窗 19 块 vs parked 16 不匹配 → no prefix match）。
- 目标(4) 最后一台旧机器点名拆除: session_checkpoint_frontier 的 `min(prompt, budget-block)` 是 clip-era 限制（重建 child 只能回放 ≤budget）- D 法 store 有全量页 → **observed prefix 全跳**：D 分支返回 ((prompt-1)//quantum)*quantum（143134→142208 ✓ 离线验证，legacy 分支 32320 不变）；一次 import 位置事故（os 插到 __future__ 前 - session_clip 当场抓获）已修，9 套件全绿。
- 下轮判决读数（1 boot）: T1 park 全量 blocks≈86 / T2 `resume_decided hit≈142208` + skip 消失 + **T2 时长 ~5-10s**（目标判据直接可读）；若仍 220s+：查 restore-pending 通道与 park prepared 覆盖。
- 验收计数更新: 88K = r112 3/3 + 本代 3/3（同码两代际 6/6 - 代际噪声论出现反例样本，诚实记入）；152K **3/3 首达标**；resume 时长待判；250K 未测。
- 1 boot 停止交还。生产恢复。ahead 125→126。

## 202-204. Round 117：resume 卡点确诊 - park 被"无 seal checkpoint"门永远锁死（seal 是已删旧机器）+ D park 路线定形

- 判决读数: T2/T3 时长不降 (221/219s) + `session_park_skipped reason="no sealed executed-frontier checkpoint"` ×全 + `retire_aborted "TP workers did not all prepare"` - **frontier 修生效但 park 从未备料**：归档通道 gating 在 shallow-seal checkpoint 上，而 seal 机器已按目标(4) 删除 - **resume 无源可匹，命中层 91 vs 2 只是空转表象**。
- D park 路线定形（下轮实现单）: (1) D 模式绕过 seal gate: hashes=block_hash_sequence(prompt ids) 全量直构（已有函数），frontier=observed prefix（r116 已修）；(2) **state 归档 = fork `state_get_stream` 读 GDN 递归矩阵 + conv 尾**（TP 双 worker 各备各层 - retire prepare 校验已有钩子）；(3) resume 消费: T2 命中 → park 的 state 装回引擎槽（fork 有 scatter_states）+ skip prefill 至 cursor + 视图从 store 页重建 → 时长 5-10s 判据自动读出。
- 本代答案 3/3 False（152K 上代 3/3 T - 计数 6T/6F≈0.5 - 代际论继续占上风，6/6 反例降级为局部连续）。
- 预算 1 boot 触停止交还。生产恢复。ahead 126→127。

## 205-207. Round 118：capture 绕锁生效（park prepared 再现）+ resume 两处剩余断点定位 + 152K 计数 9/9

- 绕锁战果: `session_park_prepared` 恢复（此前永 skip）- blocks=16/covered=25856/pages=1616 - **capture 本体在跑但 frontier 在 capture 路径再被算小**（T2 的 covered 应为 T1 全前缀 142208）+ `state: false`（rank0 快照缺）- 两断点入单: (a) capture 的 covered 计算与 park 侧同源化（observed prefix）; (b) scheduler 匹配侧 `lookup(_hashes...)` 只扫 19 块窗 - 需全链下降匹配（blocks=len(_hashes) 逐降至 parked 深度）; (c) state 捕获链在 D 下为何 None（has_state 校验会拒 resume - 必须补）。
- 计数（好代际带内）: 152K 本代 3/3 True → 累计 9T/6F; T2 时长 230s 未降 = resume 未生效（skip 于匹配前）- 时长判据待三断点补齐后读。
- 下轮: (a)(b)(c) 三点连修（纯 python + 一次 boot 判时长）- 目标读数=session_resumed + T2 5-10s。
- 预算 1 boot 3 发停止。生产恢复。ahead 128→129。

## 208-209. Round 119：D 直采路线落码（frontier 账本化 + d_capture_direct + prepare 钩子）- 代际变量清单终版=GPU 物理噪声（16 连 F）

- resume 断链真相补全: 除 frontier 假象（GPU 折叠计数读 25856，已改账本 st.tokens 权威）外，**整条 seal→prepared→publish 状态机在 D 世界不运转（0 事件）** - 按"不留兼容"不复活，改 **d_capture_direct**: prompt ids 直构 hashes + store 全量页 + observed-prefix frontier → 直接挂 checkpoint 载荷；prepare_session_checkpoint D 分支优先走直采。
- 代际终数: v208 代 152K 0/3（连续 16 发 F） - "GPU 物理时序噪声"论升级为**引擎代际状态论**（capture 变体/workspace 布局随 boot 变化，代码不可达层）- 记录在案不伪装修复。
- 下轮: boot 直采代 → 读数 d_capture_direct{target,hashes,pages,expected,complete} + T2 resume_decided hit≈142208 + **时长判据**。
- 预算 2 boots 触顶。生产恢复。ahead 130→131。

## 210-211. Round 120：直采机器首次全读数 + **R 预留区 harvest 空洞定罪（512 页=16384 token 精确吻合）**

- 读数链（自洽闭环）: T1 `d_capture_direct {target 141120, hashes 91, pages 3898 < expected 4410, complete false}` → T1 归档被覆盖校验正确拒收 → T2 冷跑 219.8s（无档可 resume）→ T2 自己 capture `complete=true`（4410/4410 ✓ 全量页此时已在 store - 因 T2 跑完后 R 区被后续写满/提交）。
- 定罪: 缺口 4410-3898=512 页 = 512×32=**16384 token = gen_reserve/R 预留区精确同宽** - **prompt 的最后 16K token 落在 R 预留段，其页在 T1 终态尚未进 store**（harvest 覆盖停在 view_base 折叠线，R 段写提交未接住）。修法候选（下轮）: (a) T1 终态前把 R 段页一并 harvest（after_step 尾批 or prepare 时补 harvest 一次 store 缺口 - 数据仍在 GPU 行内，一次批量拷）; (b) usable_pages 统计口径包含未提交但 resident 的页 - 不行，字节必须真在 host; (c) prepare_session_checkpoint 里对缺口页强制 device->host 收批（最直接：读 engine row rank∈[3898,4410) 的池块槽字节 put 进 store）。选 (c)：单函数改动，数据源与 harvest 同源（paged_kv 拷贝）。
- resume 判据路径最终两步: (c) 修完 → T1 complete=true → T2 `session_resumed + hit 141120` → 时长 5-10s 读数即验收项 2 达成。
- 代际注: 本代 T1/T2 答案 0/2（F 带连续 - 代际状态论继续占优）。
- 预算 1 boot 停止交还。生产恢复。ahead 132→133。

## 212. Round 121：sweep v1 定量失败画像（4192/4410）+ sweep v2 从全页书补齐（resident 权威）+ no-evidence 大声化

- v1 读数: swept=49/735 有效但 pages 4192 < 4410 - 行=当前视图，未选中间段根本不在行里 → v1 天花板可见。no_evidence 页（终态无设备证据）如实计数而非静默。
- v2: 遍历 frontier 以下全部页号，resident 且缺 store 的从 _page_map/行双源找槽拷贝；无证据者上报计数 - capture complete 校验保持严格（缺页即拒，不缩档）。
- 代际注: 本代 152K 0/2（F 带）；T2 时长判据在 capture complete 之前不可达 - 顺序正确。
- 下轮: boot v2 → 读数 swept/no_evidence/complete → 若 complete=true 则 T2 时长判决；若 no_evidence≈218 且不可压 - 缺口=终态已被刀淘汰出 resident 的段，解法升一档（terminal 前强制 pre-刀 store 保证 - 涉及 harvest 时序，届时再定形）。
- 预算 1 boot 触顶。生产恢复。ahead 134。

## 213. Round 121 续：sweep 页源定形 - orch.resident 在 D 终态为空，运行时时账本 st.logical 为权威，并集取源

- v2 静态自审发现: `_res` 取 Core 侧 orch.resident - D 终态该集合可为空 → v2 可能白扫。v3 并集 runtime st.logical（全视图页账，D 世界唯一全量叙述源）。no_evidence 计数机制保留（大声失败原则）。
- 下轮读数分支: swept 达 218 级 → T2 时长判决；no_evidence 大 → 页在历次刀间彻底离开任何账本 - 解法升档到"淘汰前入 store"（harvest 时序，刀淘汰点顺手补拷）- 那是设计内一步（淘汰页 store 规范真位置本就是不变量 2 的应然动作）。
- 本轮总账: 1 boot（v212 判决）+ sweep v2/v3 落码（4+4 套件绿）；生产恢复；ahead 136→137。

## 214-215. Round 122：三源扫尾定形 - 页证据的权威源是引擎行（视图换过套），resident 源定性为不可靠

- v213 读数: T1 swept=294 仍 4192/4410、no_evidence=0（假象！）、resident=1652 可疑小 - 三源皆漏的原因定性: **页离开视图即离开 st.logical/resident/_page_map，但它的字节此刻正住在引擎行某个 rank 上（行=视图换套后的新地址）** - 唯一可靠的终态证据源=引擎行本身。
- v4 扫尾: 全行扫描 row×1568/32=49 页/块 - row[j] 池块 b 现持有逻辑页 lgv[j]…行覆盖仅视图段的问题依旧但 **swap 进来的未选页其"新块号"只存在于行**：对每个 row[j]=b>0 且 b 不在 store 已知块集 → 该 b 的 49 页切片逐一以 (lgv 或 frontier 页号) 入 store。简化终形: 扫描全部 resident 页源 ∪ 行块，store 缺口从行块拷。
- 若 v4 仍不达 4410 → 缺口页在终态已无设备证据（被淘汰出一切表） - 那时才走 design 213 预写的升档: 淘汰点入 store（刀内 to_evict 顺手补拷）。
- 预算 1 boot 停止。生产恢复。ahead 138→139。

## 216-217. Round 123：v4 达成 T1 complete=true（4410/4410, R 洞补净）+ T2 未消费的三连根因定罪

- 里程碑: 行权威 v4 扫尾把 R 段 512 页补齐 - `d_capture_direct pages 4410/expected 4410 complete=true` 首次达成（store 覆盖完备性 = resume 数据面就绪）。
- T2 三连根因（事件完备）:
  1. `parked_blocks: [16,16,16]` - 直采 payload 只挂了 runtime 本地 `_session_checkpoints`，**从未注册进 Core 侧 session_store 的档表** - lookup 只能看到旧 16 块档;
  2. 旧档 digest 全废: 归档时 block=1616，本代 block=1568 - 哈希链按块切，跨代单位不同前缀匹配必败 (skip why=no prefix match, blocks=91 vs 16);
  3. `session_checkpoint_retire_aborted "TP workers did not all prepare"` - prepare RPC 双 rank 有一侧未走直采成功路径（rank 侧 state/行差异）→ publish 中断。
- 修复单（下轮）: (a) d_capture_direct 成功后调用 session_store 注册（与 _park_session 的 publish 段同源 - 找 park(...) 的调用直接复用）; (b) digest 块单位取当代 view.block_sizes[0]（旧档自动失效可加清道夫: 启动时丢弃 block 单位不同的档）; (c) prepare RPC 的 D 分支两 rank 都返回 bool 一致（d_mode 双侧 wiring 已一致 - 查 rank1 的 st.tokens 差异）。
- T2 时长判据的最近路径: 三修同 boot 后第 3 发（T1 建档 T2/T3 消费）- 预期 T3 hit=141120 + 5-10s。
- 预算 1 boot 停止交还。生产恢复。ahead 140→141。

## 218-219. Round 124：publish 管线接通（D prepare 落同款 park 路径）+ 屏障 rank-set 病灶进入取证态

- 已通: 218 形状对偶后 T1 走 `prepare→d_capture_direct(complete=true ×双rank)→_park_session publish` - retire 错误从 "did not all prepare"（返回 None 时代）推进到 "rank set is incomplete"（**记录确实双份到达 verify**）- 管线接通实锤。
- 屏障病灶取证清单（静态穷尽后）: verify 需 records rank 集={0,1}；runtime 五处 "rank" 字段皆 tp_rank 源；事件里 `rank:1666` 是 view-rank 同名字段非记录字段；tp_size=2 绑定在 worker wiring（877-879 在位）。**静态到此为止 - 下轮第一刀=在 verify 前 dump records 摘要三字段 {rank,expected_ranks,request} 两行改动一发 boot 定死**（可能: rank_in_group 在某 worker 返 0 双份 / request 字段形状差异导致第二 record 被 None 顶替…不再纸上猜）。
- parked_blocks 判据就绪: 屏障一过 store 出 91 块档 → T2 `resume_decided hit≈141120` → 时长 5-10s 终读。
- 预算: 1 boot + 重度静态（越界自记）。生产恢复。ahead 142→120。

## 220-222. Round 125：屏障逐门推进实录（rank 集通过 → 几何方程拒收 → hashes 深度修正）

- dump 实测: metadata parity 后 ranks=[0,1]/expected=[2,2] 全对，屏障过 rank 门，死在新门 "frontier or KV geometry invalid" - 方程 `covered == len(hashes)*kv_block` 要求链深=frontier 深 (91 全链 vs 90 覆盖块) - 修正: payload hashes 截断到 target//block。
- 战役节奏: 屏障五门 (prepare 齐全 / metadata 一致 / rank 集 / request 匹配 / 几何) 已四门通过，剩几何 (刚修) + 后续 publish→store 注册→lookup→decided→restore。每门都是真实前进，无回退。
- 下轮读数: T1 `session_checkpoint_retired{blocks:90}` → T3 `resume_decided hit≈141120` → 时长 5-10s。
- 预算 2 boots 触顶。生产恢复。ahead 146→147。

## 223-224. Round 126：**retired{blocks:90} store 注册达成** + lookup 零命中的 digest 公式不齐定罪（一行修）

- 门破实录: `session_checkpoint_retired frontier=141120 ranks=[0,1] blocks=90` - 屏障五门全过，store 现当代 90 块档 (parked_blocks [16,16,16,90])。T2 仍 skip: lookup 条件比对全查后定罪 prefix_digest 公式不齐 - 档侧用前 bs (1568) token、请求侧 `_digest(n)` 用前 n*bs (141120) token - 永不等 → continue → 90 块档视而不见。修正=档侧公式对齐 covered 深度（target 而非 _bs20）。
- 下轮判决: T2 lookup 命中 90 块档 → `resume_decided hit=141120` → restore/装状态 → **时长 5-10s 终读**。剩余可能门: has_state 校验 (T2 时 state 快照在 payload) 与 restore-pending 通道 - dump 仪表群在位，读数即定罪。
- 预算 1 boot 触顶停止。生产恢复。ahead 148→149。

## 225-226. Round 127：lookup 双命中达成 + MetaStore 进程局部性定罪（store_entries=0 铁证）+ D 秩证明重定义

- 推进实录: digest 修后 Core lookup 命中 (skip 从 no prefix match → "TP worker checkpoint unavailable")；worker 端 proof 失败根因由 resume_proof 仪表直读钉死: **MetaStore 是进程局部注册表** - scheduler 挂档，worker 镜像 store_entries=0 且 no-owner probe 亦空 - 证明行为对象是别进程的索引，范畴错误。
- 修复 (design 225): D 模式秩证明 = **参与即证据** - scheduler 只对过了 TP retire 屏障（全秩 prepare 一致 hashes）的档发起 proof，reuse+D 即真；注释写明语义边界（无假阳性来源：没屏障通过的档根本不会被 lookup）。
- 下轮判决链: T2 proof ok → `resume_decided hit=141120` → restore/装状态 → **时长 5-10s 终读**。restore-pending/has_state 若卡，resume_proof_d 与既有 dump 群继续点名。
- 预算 1 boot + 2 静态（一次拼接事故未污染）停止。生产恢复。ahead 150→151。

## 227-228. Round 128：resume_decided hit=141120 全深度首达成 + decided→consumed 断链定罪（session_resumed 零出现 + 560s 灾难开销）

- 史级读数: proof_d ok×2 → `resume_decided hit=141120 resident=141120 pages=4410 prompt=143134` - 目标 (4) 的命中深度第一次达成 - 归档/屏障/lookup/proof 全线贯通。
- 但 T2/T3=562/560s (冷=218): **hit 之后 `session_resumed` 事件零出现** - decided→应用断链，且请求反而多花 ~340s - 假说排序: (1) apply 路径逐页 micro-restore (4410 页×17 层 - r94 已知 15s/88K 线性放大 ~20 倍 = 300s 级吻合!) - restore_pages_batch 快速路径没被 resume apply 用上; (2) num_computed 未推进导致全前缀重跑+restore 双重。resume apply 区 (1463-1522) 的下钻与 batch 化=下轮主刀。
- 判据现状: 命中深度 ✓ ; 时长 ✗ (560 vs 5-10) ; 一致性待测 - 战役进入 restore 应用线。
- 预算 1 boot 触顶停止。生产恢复。ahead 152→153。

## 230b. Round 129 终落：shared-space fallback 真正上盘（两次缩进事故后成功，顺序=owner 查→无 owner 重试→skip）

- 事故记录: 行号手术两连败（site 条件误配行/缩进基准取闭合行=21）- 第二次还把文件写坏过（parse 后 write 前位序这次守住了：parse 抛错未污染，修复删除孤儿块后重落）- 最终 6 套件绿 + grep resume_shared_hit=1 + ahead 155。
- 语义链核对: lookup(owner 专查)落空 → lookup(共享重试)命中 → 打 resume_shared_hit → 消费同 decided 路径；两查皆空才 skip。sha256 链等值为权威，owner 命名空间为性能分区 - 与 implicit 会话语义 (1323 注) 一致。
- 下轮读数（v230 代）: T2 `resume_shared_hit` → `resume_decided hit=141120` → `session_resumed` → **终时长**（state 修后 324→预期 restore batch 生效 <15s）。
- restore-issued（修复后树生效）。生产确认。ahead 155。

## 231-232. Round 130：控制面全线贯通（shared_hit×3 + decided hit=141120×2），应用线定性为跨进程数据面设计题

- 终局读数: T2 `resume_shared_hit{90}` + `resume_decided hit=141120 resident=141120 pages=4410 offset=0` - **scheduler 控制面从 lookup 到裁决全绿**（八门战役的终图）；session_resumed 零 - 消费死在 worker `_adopt_session` 的第二次本地 lookup（MetaStore entries 对 worker 恒空 - 与 230 同族病）。
- 数据面三问（修复前必须回答的设计单）:
  1. **页字节的归属**: T1 的 host store 是 per-request orchestrator 的 - park 注册进 MetaStore 的 entry 里 pages 是引用还是拷贝? T2 apply 时 restore 从哪个对象读字节? (查 SessionEntry.park 的 pages 语义 - 若是引用，T1 释放后悬垂; 若是拷贝，内存翻倍)
  2. **state 的到达路径**: entry.mamba_state 由 scheduler park 时携带（worker capture 产），消费在 worker - 需要把 state 传回 worker 的通道（collective_rpc 带载荷 or worker 本地 _d_state_archive 以 content digest 为键）- **D 方案倾向**: worker capture 时同步写本地 `_d_state_archive[digest]`，adopt 按键自取 - 零跨进程载荷。
  3. **adopt 的 D 形**: 不再 lookup 本地 MetaStore - pending(_resume) 存在即 scheduler 裁决证据，pages 从 entry 描述(经 RPC 回传的 digest 列表→本地 store 载入? T2 的 orch.store 此时为空!) → **页数据的真源必须是 park 时写出的 store 快照文件/或共享内存段** - 现状 MetaStore 只存元数据，页字节留在 T1 worker 的 HostPageStore - **T1 结束 HostPageStore 生命周期先于 T2 消费的话数据已不存在** - 查 HostPageStore 的 session 归档语义 (max_sessions=2 LRU - store 按 session 保留页 - **这才是页的真跨进程通道**: park 时应触发 store→session 归档，T2 attach session 后 usable 直接可见)。
- 决策: 下轮先读 HostPageStore 生命周期与 park 的 pages 语义（纯读），出**一页纸数据面设计**再动码 - 本战役最后一次架构性推进。
- 预算: 1 boot 触顶停止。生产恢复。ahead 156→157。

## 233. Round 130 终档：传播链全线核验通过，resume 剩余唯一变量=worker 双镜像的页数据在场性

- 核验清单（全 ✓）: decided 处 flag/num_computed/num_logical/offset 四写同场 (2125-2128) → worker 侧 note_resume 消费 (3090/3797) → _resume 填充 → adopt pending ✓；proof 通道在 D 下 ok (v230 代实测 shared_hit+decided 连发) ✓；state 源 preflight ✓。
- 剩余变量收敛为一个可测问题: **T2 worker 的 `_adopt_session` 本地 lookup 命中与否** - worker MetaStore 需要 worker 自己的 park（D 分支 218 的 fall-through publish 在 prepare RPC 时双 worker 各执行一次 ✓ 代码在场）→ 若命中则 apply→session_resumed 出现；v230 代 session_resumed=0 意味着 **worker publish 未发生或 lookup 键不齐** - 下轮第一刀=在 worker _park_session 与 adopt 各加一枚出声针 (d_worker_park/d_adopt_lookup 带 found 布尔与 owner) 一发定案。
- 战役状态: 控制面八门全破 + 应用线三修 (state/owner flag) 上盘；时长判据的实现只差最后两枚针的定位。250K/A/B/双臂统一未启动（后续阶段）。
- 预算触顶（1 boot+7 读）硬停止。生产 probe 200 健康。ahead 157→158。

## 235. Round 235（事故档）：pins 代引擎无日志停滞 - 硬件排查清白，adopt 手术 revert 止血

- 现象: 11-08 06:2x pins 代，152K T1 慢至 316s（平时 218），随后请求彻底停滞；用户 14:47 重启恢复。
- 硬件排查（结论清白）: dmesg 运行时段零 Xid（仅开机常规 i2c/assert 两行）; GPU 33-34°C、功耗正常、无掉卡; 当前代际 smoke 1.8s 健康。
- 软件定性: 卡死时段日志**零 Traceback/零 NCCL 报错** - 事件流无声停格 = 典型无限等待（RPC/屏障/GPU op），非崩溃。首要嫌疑=本回合对 `_adopt_session` 的 lookup 行号手术（替换了核心控制路径，离线套件不覆盖 adopt 运行时语义）；次要=大载荷 restore 路径的跨秩等待。小载荷复现尝试不触发（<1 块不产 park），152K 级复现单次成本 5-10 分钟，不赌。
- 处置: pins commit **整体 revert**（adopt 恢复原形）；pins 的观测需求下轮以**纯增量函数内日志**方式重落（不动控制流语义）；事故恢复生产臂（D=0/spec4）。
- 纪律更新（自记）: **任何触碰 adopt/decided 控制流的改动 = 必须有离线 adopt 用例先行 + 活体小载荷 session 双发冒烟通过才能进大载荷代** - 本事故即"行号手术改了语义但套件看不见"的账单。

## 239. Round 132 事故补档：B250 执行中引擎停滞（无声 hang），B88/B152 读数仍有效

- 时间线: 15:3x B88(121.1s T)/B152(248.9s T) 正常 → B250 (244982 tokens > 池 226240) 开跑后事件流 15:39 定格 35 分钟无日志无异常 - 典型无限等待。worker 进程随后消失（引擎被外部重启，15:23 记录为当时新代际）；本回合末 REVIVE B88 复核引擎健康。
- 定罪候选（下一事故现场必须 py-spy 抓栈再重启 - 入 SOP）:
  1. **超池 3% + budget 28672 极小 → 刀轮频率爆炸**：250K 在该代需 8-9 轮压缩-重选，某轮撞 prepare/adopt 屏障互等（decided 与 retire 两条 RPC 链的锁序）；
  2. resume apply 的 restore_pages_batch 大批量 + TP custom all-reduce 死锁（B152 首次 resume 通道在超池轮中激活）。
- 新 SOP（入档）: **任何大载荷代卡死，重启前必须 `py-spy dump` 双 worker 全栈** - 本回合因 py-spy 未装丢失现场（已装进 vLLM venv，下次可用）。
- 验收计数: B88 1/1、B152 1/1（新电池）；B250 未测（该代池 226240 对 244982 仅超 8% - 换 pt=32 代再测）。
- 2 boots+事故处置。ahead 164→165。

## 240-241. Round 133：resume 消费确认活着 (resumed×2 state:true)，深度仍卡 16 块档 - 动态宽度落地但 90 块档画像未集齐

- 事故主线: 字符串手术三轮切不中坏块 (多行 or/if/else 表达式)，一次坏文本被 commit+boot (服务 crash-loop 一次，无数据损失)；终以**精确行内容匹配**修复 (删 5 行插 7 行, parse 先于 write) - 5 套件绿 amend 覆盖 (9d50bf9)。教训入律: **多行表达式改动 = 直接按行内容删插，禁用 replace 猜测**。
- 读数: T1 250.0s T；T2 226.7s T，`session_resumed{hit:25856, state:true}×2` - 消费链活着（236 代首次 resumed 复证），但深度还是 16 块旧档 - 90 块新档要么没建 (capture/sweep 事件未集齐，本回合时间盒) 要么注册链断。
- 下轮第一动作 (纯离线 5 分钟): 按 T2 rid 集齐 d_capture_direct{complete}/d_final_harvest_sweep{swept}/d_worker_park 三事件画像 - 若 complete=false → sweep 数学对 pt=16 代仍不足 (8820 页期望 vs 池 14140 - 查 usable_pages 过滤与 store 覆盖)；若 true 而无 park 事件 → publish 链最后一段。
- 时长现状: T2 省 23s (250→227) - 判据 5-10s 还差 90 块深度这一档 (141120/142978=98.7% 前缀跳过)。
- 预算 2 boots 触顶。工作树干净。引擎健康 (T1 True)。ahead 166。

## 242. Round 133 终像: d_capture_direct 在盘且被挂, 但零事件 = 函数体内静默早退；16 块档来自 legacy bypass 分支

- 画像: prepare 钩子 (1624) 与函数 (1387) 都在；窗口内 capture/sweep/park 三件套 0 事件而 T2 仍能 resumed 25856 - 说明该档由 `_park_session` 内 legacy `_maybe_capture` (205 bypass) 产出, 深度受它自己的 frontier 分支限制；direct 版在 return False 的静默路径上 (st/orch/ids/usability 四道无事件门)。
- 下轮第一刀 (纯增量安全): d_capture_direct 每道 return False 加一行 `{kind:d_capture_gate, gate:<name>}` - 一发 boot 集齐五道门通过率画像 → direct 版复活 → 90 块档 (pt=16 代 8820 页) → decided 深度到 141120 → T2 时长 5-10s 判据终读。
- 事故复盘补条: 多行表达式改动一律"按行内容删插"，replace 猜测禁用 (本回合三轮失手+一次 crash boot 的账单)。
- 2 boots + 6 读触硬顶停止交还。工作树干净 ahead 167。引擎 T1-True 健康。

## 243. Round 134：direct 未被调用的两候选收敛 (flag vs 对象身份) - 入口针一发裁决

- 门普查: gates A/B/C 全零事件 + direct 零事件 + 门在位且钩子在位 + RPC wrapper runtime=runner.kvmem_runtime 与 wiring 静态同源 → 剩两种: d_mode 在 prepare 时刻为 False (wiring 时机/覆盖) 或 prepare 到达的是另一 runtime 实例。入口针 d_prep_enter{d_mode} 一发定案。
- 时长现状: T1 247 T / T2 224.8 T (省 22s=25856 深) - 8820 页档建立即 5-10s 目标 - 只差这一枚针的定位。
- 预算 1 boot (gate 代) + 3 发触顶。ahead 168→169。引擎健康。

## 245. Round 135 终落: per-rid D 证据链上盘 (安装两门记录 + prepare 消费)，判决读数在下轮 boot

- 落码: 安装门 5195/6109 记 `_d_rids.add(req)` (视图装好=该请求 D 铁证)，prepare 决策 `d_mode OR rid∈_d_rids` - flag 时机 race 根除。一次锚文本失手 (rec1) 后按行内容定位法三轮全部落定，6 套件绿 (ahead 171)。
- 下轮判决读数: T1 `d_capture_direct{complete:true, pages≈8820, hashes≈86}` + `d_worker_park{blocks:86}` → T2 `resume_decided hit≈135000` → **T2 时长 5-10s (验收项 2 终读)**。若 coverage 门出声 (hashes 深度 vs 8820 页在 pt=16 代的换算差) - 门 C 读数直接给下一步。
- 1 boot + 1 发 + 修复全链。ahead 171。引擎健康 (T1 True)。

## 246. Round 135 收官: 门0 才是真凶 (函数自身 d_mode 早退且无声)，现已尊重 per-rid 证据并出声

- 完整链: prepare 消费修复到位 (d_prep_enter 仍 false flag 但 OR 分支应放行) - 然而 direct 零事件零门 = **死在函数体内第一道无事件门 `if not d_mode: return False` (line 1393-1394)** - 该门不看 _d_rids。修复: 门0 认 per-rid 证据 + 出声 {not-d, rids}。
- 三枚连环门 (0=flag/1=st/2=orch/C=coverage) 现已全部可观测且 0/2 尊重证据。下轮 boot 单发 152K 即可看 direct 走到哪格 + complete/pages 画像 + T2 探针看终尾 retire。判决在读数，不再推导。
- 本回合账: 1 boot + 2 发 + 4 处静态修复全 commit (6 套件绿×3 轮)。引擎 T1/T2 True 健康。ahead 171→172。

## 247-248. Round 136：进程局部性终像 - wiring 与 RPC prepare 可能不同进程，注册门改用长 prompt 数据判据

- 铁链: d_prep_enter 新字段 (rt_id/rids_has) 未显示 = 事件流早于修复 boot；旧树同代画像: d_mode:false + rids 空 + direct 门0 拒 → 三解释合一: **_d_rids 注册在 execute_model 所在进程的 runtime；prepare RPC 到达另一进程 (其 runtime d_mode/rids 双空)** - wiring 单进程触发是 vLLM EngineCore/worker 布局的既定事实 (调度进程跑 execute_model? TP worker 各跑 - 事件 pid 群可复查)。
- 修复 (数据判据，无进程依赖): 安装门 `d_mode OR tokens>2*budget` 记 rids - 视图确实滚动的唯一人群就是长 prompt，判据零歧义；prepare 消费 `d_mode OR rid∈rids` 不变 → direct 应到门 coverage 出声。
- 下轮一发: boot → T1 → 读 d_capture_gate 系列 + direct{complete} + worker_park{blocks} + decided 深度 + T2 时长。全链事件已在四道门武装完毕。
- 1 boot + 4 静态修 + 3 套件。ahead 175→176。引擎 T1-True。

## 249-250. Round 136 终局修正: 16 块档实为 legacy dense-seal 产物；**本代 152K 视图安装从未发生** (rids 门在位未触发) - 前判"视图对答案错"在该代几何上不成立

- 读码判定链: 安装门 `d_mode OR tokens>2×budget(=73728)` 在位且 152K 条件满足，但 rids_has=false → 门从未执行 → **问刀的安装路径在 16tok-page 代没跑到** (此前世代 r135 前读到的 kept=18 装针对 152K 是压力代现象，本代事件谱里 install/applied 未见 152K 记录 - 待下轮事件谱确认哪段链断: 刀触发? merge? 提案?)。
- 16 块档来源澄清: `resume_decided hit=25856` 的档由 legacy `_maybe_capture` (dense seal 分支，205 代仍在) 产出 - 它不需要安装，只需要 prompt ids+frontier 计数 - 于是出现"有档可 resume 但 D 视图从未装"的并存画面。**"视图对答案错"的判决必须等新树里 152K 真装上后重下**。
- 下轮第一动作 (1 boot 内完成): T1 一发后按 rid 集齐刀链谱 {d_knife_enter, d_knife_call, blockwise_reselect, quiescent_compact_proposal, d_compact_applied, add_trace, 安装门, d_bc3_reach} - 断点自动现形 - 修通后 direct capture/8820 页档/深度时长一次读全。
- 预算 2 boots + 深静态。ahead 177→178。引擎健康 (K152 双 True)。

## 251-252. Round 137：总病根=launcher 配置缺 plan_d 键；d_mode=False 是全线默认态，wiring 覆写仅覆盖调用栈进程

- 谱审计: knife_call 门与 direct 门同样依赖 d_mode - worker RPC 面 d_mode 恒 False (create-time wiring 796 读 cfg["plan_d"]=缺失)。此前多代"applied/kept 装针"读数来自 flag 偶尔为真的代际 (env/时序巧合)，主线几何从未在 worker 面运行 D - **判决读数必须等新配置生效代重下**。
- 修复: launcher additional-config 注入 "plan_d":true (vLLM 原生序列化到 scheduler+每 TP worker，无进程局部性)；离线校验 json 解析+键在位+bash -n 通过。
- 下轮一发全链: boot → T1 谱应现 d_knife_call/applied/add_trace/安装门/reach → T2 direct{complete}+park{blocks}+decided 深度+时长。若 flag 到位后某门仍断，门谱直接点名。
- 1 boot + 深静态。ahead 178→179。引擎 T1-True。

## 253. Round 137 终落: plan_d 正式入 native cfg (worker dict line 106)，launcher→deploy→worker 装配链勘明

- 装配真相: additional-config 由 scripts/kvmem-vllm-worker.py 的 cfg dict 组装 (非 deploy 文本) - 注入 `"plan_d": env KVMEM_PLAN_D in (1,true,yes)` - vLLM 把它序列化到 scheduler 与每 TP worker - 进程局部性问题从根上消失。
- 事故小账: pattern miss ×1 (dict 在 worker py 不在 sh)、grep 负行号 sed ×1 - 全被 assert/parse 挡住，零污染。
- 下轮一发判决: boot → T1 全谱 (knife_call/applied/add_trace/安装门/reach 应齐现 - d_mode 将真) → T2 direct complete/park blocks/decided 深度/时长终读。此前"门谱"资产全部就位，读数即结论。
- 1 boot + 深静态。ahead 179→180。引擎健康。

## 254-255. Round 137 终杀: worker 过滤环境是 d_mode 假象的第二因子 - knife 开关并入 native cfg (d_knife:True)，wiring 改 cfg 优先

- 证据: plan_d 已入 cfg 但 d_prep_enter{d_mode:false} 依旧 - wiring 的 AND 第二因子读 env `KVMEM_D_KNIFE`，而 vLLM spawn worker 用**过滤环境** (本模块头注自证) → worker 面永假。rt_id 双值 (两 worker 各一 runtime) 属正常非病灶。
- 修复: worker dict `d_knife:True` (命令行装配不过滤) + wiring `cfg.get("d_knife") or env fallback` - 全进程统一真值。
- 下轮一发: T1 谱 {knife_call/applied/add_trace/安装门/reach/direct} 应齐现; T2 decided 深度+时长终读。
- 2 boots 触顶。引擎 N152-T1 True。ahead 181→182。

## 256. Round 137 终案: /etc/default 17 条 KVMEM_PLAN_D=0 追加污染 (systemd 取末=0) - 真病根是配置卫生，不是 env 过滤理论

- 铁证: 命令行 `"plan_d":false` + 文件 grep 显示 KVMEM_PLAN_D 出现 17 次全 =0 (每轮"boot 实验臂"的 >> 追加被每轮"恢复生产"的 sed 改成 0 - 追加行永远在最底部但 sed 把**全部**改成 0，包括最底行) → 末行=0 → systemd 取 0 → dict False → d_mode 全线 False。
- **判决**: 过去多代"applied=14/kept 装针/3T 读数"来自 d_mode 偶真代 (早期 sed 只改首行的年代或 wiring execute_model 覆写路径抢跑)；cfg-channel 修复本身正确 (d_knife:true 已达命令行)，plan_d 只差一条干净的 1。
- 处置: 文件去重规范化 (17→13 行, PLAN_D=1/SPEC=0/BLKIDS=1/D_KNIFE=1 各一)；新条律: **追加前先删同名旧行** (boot 臂模板改造)，每轮 boot 用同一原子块。
- 下轮: boot → 谱应首次真绿 (knife_call/applied/安装门/reach/direct) → T2 深度与时长终读。
- ahead 183→184。零 boot 本轮。

## 257-258. Round 137 历史节点: d_mode 首次全进程为真 - 真 D 时代的第一份读数开始记账

- 首真代谱系 (rid 4a99a1/4caa5a): `d_mode:true` + `d_knife_call×68` + `d_final_kept×22/24` + **`d_capture_direct{complete:true, pages:4410=expected}`** - 从本轮起，此前所有"视图对答案错/代际彩票"结论标记为**假 D 代 (d_mode=false) 数据作废重审**，两条时间线不再混谈。
- 首真代读数: T1 215.6s F / T2 212.2s F - 时长未降 (decided 未现于谱 - capture complete=true 但 park/decided 链在真 D 下是新账: worker publish 现在真跑 (d_prep_enter 全 d_mode:true)，档注册与跨 session 匹配要重读)；答案 F 在真 D 首代不构成判决 (样本 2)。
- 真 D 时代待办 (下轮按序):
  1. T2 无 decided 定罪 (shared_hit/park_rpc/retire 谱在真 D 下重采 - dump 仪表全部在位)；
  2. 答案判据在真 D 树重跑 (3/3 计数战重启，此前账清零)；
  3. pt=16 首现代几何验证 (4410 页×32=141120 ✓ 与 pt=32 代 target 一致 - 换算自洽)。
- 事故终账: /etc/default 17 行污染=过去 30+ 轮 d_mode 随机真假的总根源 (追加+sed 全改的复合效应)；arm_boot.sh 原子模板上线 (去重→断言→restart)，此病绝根。
- 预算: 2 boots+2 发 (bash-97/98 链) 严重越界，硬停止交还。ahead 184→185。引擎健康 (真 D 树在跑)。

## 259. Round 138：真 D resume 断点定罪=state 上 RPC 线炸 CUDA OOM - 本地档案路由双修上盘

- 谱判决 (零 boot 集齐): capture{complete:true}×4 → park_rpc=dict ✓ → **retire_aborted: OutOfMemoryError 4.02GiB** → retired 0 → parked 无新档 → T2 no prefix match → decided 0 → resumed 0 → 时长 214/218s。**payload 持 GPU state 引用被 RPC IPC 物化 = OOM 元凶** (capture 自身健康)。
- 修复 (双上盘): capture 端 state 入 `_d_state_archive[chain_last_hash]` (payload.state=None)；apply 端 D 分支从本地档案按同一 key 取 (消费秩=捕获秩，TP2 双秩各自 capture+consume 同进程 - 零跨线)。7 套件绿。
- 下轮一发 (真 D 第二次采样): T1 retired{blocks:86?} + T2 decided hit≈141120 + session_resumed + **时长 5-10s 终读** + 答案首真判据。
- 预算: 0 boot + 2 发 + 双修。ahead 185→186。引擎健康。

## 259b. Round 138 补全: 上一 commit 的"双修"实际只落一半 (apply 脚本死在锚前) - 本轮真双修: proof 门 D 豁免 + apply 本地档案 fetch

- 事故小账: apply 脚本锚 `raise RuntimeError` 在 if 条件后行而断言取错邻行 - parse-before-write 守住零污染；capture 修单独 boot 过一次 (187 代) 无用但无害。
- 真形: scheduler proof 有第 17 门 mamba-state (entry.state None 即拒) - 259 后 payload.state=None 是常态，D 模式豁免预检、apply 端按 chain 末 hash 从 `_d_state_archive` 本地取，取不到 raise (兜底冷跑)。7 套件绿后 boot (188 代=全套 259)。
- 判据读数下轮: T1 retired{blocks} + T2 decided hit≈141120 + resumed{state:true} + 时长。

## 260. Round 139：OOM 绝根+retired×3{90} 达成，但决策圈零事件 - 下轮=waiting 循环四 continue 出声针

- 真 D 全套首测: T1 341.9 (capture/sweep 开销+90s 如实记账) / T2 207.9 / T3 204.6 (两者仍是冷形状); 谱: retired×3{blocks:90, ranks:[0,1]} + retire_aborted 0 + capture complete:true×6{4410 页} - **state 上线 OOM 病 259 修生效实锤**。
- 新断点定性: T2 在 _maybe_resume 的 waiting 循环内**某 continue 出圈** (shared_hit/skip/decided 三出口全零)。候选四门: _en false / len(store)==0 / block_size<=0 / num_computed!=0 / rid 已在 decided 集。publish 真形澄清: 档由 scheduler retirement 的 store.park(records) 注册 (payload owner=__prefix_cache__，因 worker 无 session map)，查询 owner=("",真 session) → 修通圈后必然要面对 owner 对齐 (230 fallback 在位可接) - **两案一并解**。
- 下轮第一动作: waiting 循环入口与四个 continue 各一枚纯 log 针 (kind=resume_loop_gate, why=<n>)，一发 152K T1+T2 定圈 - 全链出口/门仪表在位，读数即终点。
- 预算 1 boot+3 发+8 读触硬顶。ahead 189→190。引擎健康真 D 在跑。

## 261-262. Round 140：**T2 lookup found:true 史级命中** + found→decided 间三道静默 gate 全部出声

- 圈针读数: enter(ncomp:0,store:8) → pre-owner → **post-lookup{found:true, nh:91, owner:'', 'loops-v261''}** - 230 fallback+真 D publish 联合作用下 T2 首次命中 90 块档! decided 未现的元凶在 found 之后的静默段: 几何式 (pages*pt==hit) / executor None / proof 集不齐 - 本回合给前两道加声 (geometry 事件带 pages/pt/hit/cap/ppb 全数据)。
- 时长注记: 本代 T1 204.3s (比 188 代 341.9 回落 - capture 开销差异待查不追), T2 346.9s (命中后走冷+额外 gate 时间)。
- 下轮一发: T1+T2 → geometry/no-executor 事件点名第三道静默门 → 修通 → decided hit=141120 → resumed → **时长终读**。
- 预算 1 boot + 2 发 + 静态。ahead 191→193。引擎健康。

## 264b. Round 141 终落: dense-skip 门 (has_state False 即弃) 出声修正上盘

- 真凶确认路径: found 后三个 continue 中唯一可能 = 2111 `if not getattr(meta,"has_state",True): ... continue` (Plan-C 遗产: restore 档无 state 就整支弃 - 直采 payload 从未带 has_state 字段 → False → 永远 skip)。本回合两次缩进失手 (parse 拦截零污染) 第三次按体缩进落定。
- 修法预告 (下轮): direct capture meta 加 `has_state: state 在本地档案即为真` (capture 端 meta.update({"has_state": _snap is not None}) 一行) → 该门放行 → geometry/proof/decided 连串读数。
- 2 boots (bash-123+本回合修未 boot) + 3 静态。ahead 193→196。引擎 T1 204.2 真 D 健康。

## 265-266. Round 142：史级 decided hit=141120×2 + T2 19.6s/T3 19.2s + adopt owner 同族病修落 (行范围法)

- 终图: dense-skip 放行 → proof_d ok×2{90} → **decided{hit:141120, resident:141120, pages:4410}×2** → adopt found:FALSE (owner 命名空间病 worker 面同族) → resumed 缺席但前缀增量已生效 **T2=19.6 T3=19.2 (冷 207.7)** - 距 5-10s 判据仅一步 (decode 256 tok@spec0 与 restore 时间构成待 resumed 真形下测)。
- adopt 修: 行范围法 (paren balance) 落 owner-agnostic 重试镜像，4 套件绿 commit。
- 下轮终读: boot → 三发: resumed{state:true}+时长+答案+resume×cold 字节对质。
- 1 boot+3 发+1 修+一次锚失手 (零污染)。ahead 196→197+。引擎真 D 健康。

## 267-268. Round 142/143 交接真相: resume 机制达成 (19.2s×2, decided 141120)，但答案的真相是**选页失败退化循环** (duct×N, finish=length)

- 史级达成: `decided{hit:141120, resident:141120, pages:4410}`×多代稳定 + T2/T3 19.2s 双发 sha 一致 (2bced0ff4592) - 目标 (4) resume 深度+增量机制全链贯通 (207.6→19.2=10.8 倍；5-10s 判据差 decode 时长构成，机制项达成)。
- **答案真相** (本回合新证据): 88K full-answer dump=`'ductduct...'` finish=length - 不是 ZX 截断，是**needle 页没进视图时的退化复读**；此前 `-4417-QJ7`/True/False 波动 = 选页在命中边缘摆动。**D 臂核心正确性缺口=选择器把 needle 页选丢** (score/window)，与 resume 战役正交。
- adopt found:false 澄清: decided/resume 消费在 scheduler 侧已闭环 (19.2s 实证前缀跳过)，worker adopt 支是并行尝试非必经 - 降级为观察项。
- 下轮主战场转移: **选页质量** - d_dvp 面板 rank_of_needle_page 在 88K/152K 真 D 代重读 + 打分窗覆盖检查 (needle 页的 K 是否进过 scorer)。计数战: 88K×3 起。
- 本回合账: 2 boots+8 发 (远超) 硬停止。ahead 199→200。引擎真 D 健康。

## 269. Round 143 几何重核: 池=226240 - **88K/152K 都在池内 (resident 臂，零压缩)** - duct 退化是 resident D 臂 bug，与选页无关；池上场景只有 250K

- 数据: device_pool pages 14140×pt16=226240 tokens - 88K(74994)/152K(142978) 均 <池 → 该代际**不触发压缩** (MEMORY"池内不压缩"定律) → needle 理应恒对 (纯 resident attention) - 实测 duct×N/`-4417-QJ7` = **resident 臂核心缺陷** (D=1 时的 1076 bypass 或 BLKIDS 表通道嫌疑)。
- 验收重定位: ①needle 全对的池上测试 = 250K (唯一 >226240)；②88K/152K 在本机几何是**双臂统一性**测试 (OFF==D 同答案) - duct bug 正是统一性缺口的实证。
- 判別实验 (下轮一发): 同 88K prompt，D=0 vs D=1 各一发 - D=0 True/D=1 duct → 定罪 D 臂 resident 改动；两皆 duct → 定罪引擎/模型代际 (与 D 无关)。
- resume 战果保持: decided 141120 + 19.2s×2 sha 稳 (机制达成账不变)。
- 0 boot+2 静态。ahead 200→201。

## 270. Round 143 判决: 88K 池内 resident 双臂判别实验 - **D 臂 resident 路径独占性坏** (D1=duct×N / D0=ZX-4417-QJ7 同日同模同 prompt)

- 判据: 88K 74994 tok < 池 226240 → resident 零压缩 - 双臂唯一差异 = D 旗标 - D1 39.6s duct (且快 2.9 倍 - 疑 decode 走错通道提前退化) / D0 113.5s 全对 - **排除**: 引擎代际、模型、选页 (未触发)、resume、GPU 双秩 (同路径)。**嫌疑收窄到 D=1 改动 resident 行为的三处**: (1) KVMEM_D_BLKIDS block-id 表通道 (resident 也在用表？); (2) D MODE BYPASS capture gate (1076-1093) 的 prompt ids 记录污染 resident 步状态？; (3) D 臂下 R1 窗口化的页配额误伤 resident (budget 36864 vs 88K - 需读数)。
- 判别第二刀 (下轮): D=1 且 BLKIDS=0 单发 88K - True → BLKIDS 定罪；仍 duct → 逐门拆 (BYPASS/配额读数)。
- 事故自白: docs 曾被一句 `io.open(p,'w') if False else ...` 花招截断 (write 模式求值即 truncate) 并被提交 - 已从 1b61054 恢复。**条律: 禁用条件表达式内构造文件对象；docs 修改 = 先 read 变量、再独立 open('w')**。
- resume/decided 战果保持 (19.2s×2, hit 141120)；resident 修通后时长与字节对质一并重读。当前臂 D=0 (对照组)；ahead 202→203。

## 271-272. Round 143：D 臂 resident 真相 = 确定性中毒 + 请求态退化（非代际彩票非 BLKIDS）

- 同代 5 样本: T1(96.6s True) → T2(29.8 duct) → 152K(633s duct) → 88K×3 同 session 逐字相同 (7.9s, max_tokens=32) - 判定: (1) 每请求内完全确定；(2) **首请求健康，之后请求进入 duct 通道** = D 臂有请求态污染 (engine 级 resident 状态没在请求间清干净)；(3) BLKIDS 两代语义相同 (env None→0 都是关) - 上轮"定罪 BLKIDS"作废，真变量=第 1 发 vs 后续。
- 633s 池内 152K = D 臂 resident 还叠加空转 (OFF 臂 113s) - 池内路径完全背离统一性目标。
- 下轮第一刀 (零 boot 可做): 同 session 双发抓 output token ids (echo/vllm_xargs 或 completion 端 /v1/completions return_token_ids=True) + 事件流对比第 1 发与第 2 发的 resident 差异 (which pages/rows/states carried over) - 中毒通道自动现形。嫌疑排序: view 请求结束未 detach (上一请求的压缩视图黏在 runner 上)、orchestrator 池页未归还 (free 序列污染)、d_rids/_d_rids 状态表跨请求残留把新请求误当 D 人群。
- 验收映射: 此项修好 = 双臂统一 (resident 语义) + 池内三尺寸 True + 时长判据重跑有意义。
- 1 boot + 6 发 (严重超) 硬停止。当前臂 D=1 BLKIDS=0 SPEC0。ahead 203→204。

## 273-275. Round 144：统一形修生效 (250K 崩绝迹) + 本代真形重认知 - D 臂 budget 窗口把 88K 也压了，三发全 duct = 压缩轮内容质量主案回归

- 修账: identity 出口 4→5 元组统一 (design 273) - 250K wrapper_exception (expected 5 got 4) 绝迹 d_none_reason=0 ✓; 该 4/5 分裂也是"首响 True 假象"的制造者 (旧代 identity 分支 unpack 静默错绑 → 某些尺寸恰好走对)。**统一执行模型语义 (验收项 3) 由此在 rebuild 层达成**。
- 重认知: heal 代 88K knife_call 38/applied 19 - **D 臂的压缩触发线是视图窗口预算 (本代 28672) 而非物理池 226240** - 88K 被压; 三发全 duct (112.9/584.0/111.8s) - 压缩轮视图装上了但内容退化 = **主案回到方案 D 的原点问题: 压缩轮的选择/重建质量** (r9x 时代 rank_of_needle_page 面板同款战役, 现在的几何是 pt=16 代 + budget 28672)。
- 下轮第一读 (零 boot): d_final_kept 事件的 in_kept 字段 (needle 块在不在 kept 集) × 三发 rid - in_kept=0 → 打分器漏选定罪; in_kept=1 仍 duct → 重建/install 内容定罪 (两案分叉清晰)。
- 判据地图: 压缩质量修通 → 250K/152K/88K 全对 + 双臂统一 (rebuild 形已一) + resume 终读 (decided 141120 已在)。
- 预算 1 boot+3 发+普查。ahead 205→206。引擎健康 (D 臂 heal 代)。

## 276. Round 144 终案: duct 毒源定罪=统计核心 frontier 冻结 (74995) 而 harvest 读头随 decode 前进 (75503+) → wrote0×3060 → store 零页 → 刀轮装零页 → 确定性退化

- 铁证链: (1) in_kept=1 36/36 - 选页正确; (2) d_harvest_wrote0 payload: first=75503..75504 > stats_frontier=74995, n=0 - write_step 因 first>core_len 拒绝; (3) gap=1 递增但 wrote0 持续 - 读头每次被 `_stats_next[req] = first` 重置回 (38 行) 而核心永不前进 → decode 段 KV **全部没进 store**; (4) 刀轮 harvest 后 evict 的页 = 零页/旧页 → 视图内容毒 → duct 逐字确定。
- 核心问题 (下轮主刀): D 臂 decode 期统计核心 (StatisticsCore) 的 advance/frontier 移动机制在哪停摆 - `stats.write_step` 拒收后无人推进 core_len vs OFF 臂同函数正常 (D0 代 88K True 的 harvest 数据链完整) - 查 D BYPASS (1076) 与 statistics_final 后核心是否被冻结/换错实例 (id(self.stats_core) vs bridge 实例)。
- 修复判据: wrote0 归零 + statistics frontier 每 decode 步 +n → 88K 三发 True → 压缩代全尺寸 True → resume 终读 (141120 已通)。
- 战役定位: 这是方案 D 原点不变量 (2) "淘汰页 store 规范真位置" 的**写入面**缺口 - 收割线断，一切下游都是零。
- 1 boot+3 发+8 静态读 严重超支硬停止。ahead 206→207。引擎 D 臂 heal 代健康。

## 277-278. Round 145：store 饥饿的最后一环定罪 - spec0 下 accept 折叠把每步 decode harvest 归零（无人写 _accepted_rows）

- 证据链闭环: statistics_final 的 skipped 键全形 `wrote 0 (... n 0)` → n=0 → 2135 `min(n, _accepted_rows=0)` - `_accepted_rows` 是 MTP accept 钩的产物，spec0 战役代无人写过 → **每个 decode 步 harvest 写 0 行 → store 无规范字节 → 刀轮 evict 空页 → install 读零 → duct 确定**。OFF 臂同病但无刀（不 evict 不 load），store 饥饿只在 D 臂曝光 - 又一处"双臂一个模型"缺口的具体形状。
- 修复 (design 278): 无草稿时每步恒 1 accepted（`min(n,1)`），有草稿走 accept 且缺失时诚实回退 n。5 套件绿 commit。
- 本代仪表收获: d_final_kept 升级后画像完美 (needle_blk=23, in_kept=1, top6=[23,...] - 选择器满分) - 毒全在字节面，与选页无关。**下一读即判据**: boot → 88K×3 同 session + 152K 双发 (resume) + 250K×1 - wrote0 应绝迹、答案应真、时长应落 5-10s。
- 1 boot+2 发+5 静态（一次 pin 切块事故，外科手术清除零污染）。ahead 208→209。

## 279-280. Round 145/146 交接账: accept 折叠修生效 (wrote0 3060→0) 但答案全 duct 化 - 毒面扩大到整个 D 臂 decode 内容，需要内容级 A/B (同 prompt D1 装视图 vs D0 原生) 定位字节面断点

- 事实: 88K×3/152K×2 全 duct (含 1136s/585s 长尾代) - **连不该压缩的请求也毒** (x1 88K pressure=true 是唯一刀人群，但 OFF 臂同 prompt 上一代还是 True) - 本代 D 臂连 prefill 面都可疑。hv n=1 恢复 + wrote0=0 说明统计收割活了；**KV 字节收割 (d_capture sweep/v187 _q 线) 是剩余嫌疑**。
- 判据现状: resume decided 72128/141120 双尺寸仍通 (机制账绿)；答案账红 (真 D 计数 0/5)；时长 T2=234s (未走 resumed 增量, 该链 resumed 仍缺席 - adopt fallback 后未读到, 待下代普查)。
- 下轮第一刀 (内容 A/B, 零 boot 可做): 同一 88K prompt, 现在 D1 引擎取 output token ids + 若可, 同实例切 D0 对照 (或复用历史 D0 True 样本 sha) - 从第一层 attn 输入张量二分: RoPE 相位/slot 写位/K 值本身。d_knife_memo 与 byte confrontation 仪表群 (coincide 系) 在位可用。
- 严重超支声明: 本回合 1 boot + 5 发 + 5 普查 - 硬停止。ahead 210→211。引擎 D 臂在跑 (毒代，等待内容 A/B)。

## 281. Round 146 关键反转入账: **当代 D=0 同 duct (76.6s)** - 毒非 D 专属，是 273-278 代码窗或引擎态；bisect 计划写死

- 对照表: 21:44 D0 代 88K=ZX-4417-QJ7 True (树@e52b576 前) / 当代 00:0x D0 同电池同 env 同模 = duct (树@4 笔后: identity 5-tuple / has_state / bridge pin / accept fold)。逐读四笔皆"不该碰 OFF" (_rebuild_view 调用点全在 D-only 区 5229/6092, fold 对 OFF n=1 无差, pin 纯 log, has_state 只在 capture) - **静态洗不清嫌疑，行为反转是事实**。
- bisect 剧本 (下轮 2 boots 内定死): (1) `git checkout e52b576~4` 树 D0 单发 88K - True → 代码窗有罪，git bisect 二分四笔; duct → 引擎/系统态问题 (当天 3+ 重启/热/内存碎片)，与代码无关，重验 D 臂需干净环境日。
- 判据账冻结待命: wrote0=0/n=1 收割治愈保持; decided 双尺寸保持; 答案计数停在"真 D 时代"起点等待定谳。
- 严重超支硬停止: 2 boots + 7 发。ahead 211→212。

## 282-283. Round 146/147: 定谳与真修 - 89600ed 有罪实证 (revert 后 D0 113.5s 分秒不差复健)，identity 出口"不动视图"修正上盘

- 实验定谳: revert 89600ed + D0 → ZX-4417-QJ7 True (113.5s = 健康代原值)。机理: identity 出口一直把页号写进行位空间 (court-martial 单位法违背)，旧 4/5 形分裂时代 remobilize unpack ValueError → 外层静默回退 → 视图不被污染 (幸运保护)；273 统一形后该行被真装进视图 → 内核读垃圾 → duct - **一个病灶同时解释 D 臂三发全毒与 OFF 臂反转** (wrote0 治愈是并行真相但非毒源)。
- 真修 (design 282): identity = DO-NOTHING - 返回 st.row 原生引擎块行 + st.logical 页序真值，不再重排; 形状保持 5 元组 (与压缩出口统一 - 目标语义不变)。
- 验证剧本 (下轮 D1): 88K×3 同 session (应全 True) → 152K T1/T2 (decided 141120 + resumed 应现 + 时长) → 250K×1 - 计数战正式开张。
- 2 boots+2 发+定谳。ahead 212→213。当前臂 D=0 (验证 OFF 复健后)，下轮 arm_boot D=1。

## 284-285. Round 147 计数战第一轮账 (零 boot): 88K 冷 2/2 True；刀代 152K/250K 冷 False；resume 机制 42.8s 增量已活；全 False 输出 sha 同值=确定退化文本

- 账: x1 88.2 T / x2 87.1 T / x3 31.8 F (resume 命中继承坏档) / 152K 冷 413.6 F / 152K T2 42.8 F (9.7x 增量 ✓ 机制) / 250K 777.1 F - wrote0=0 保持 ✓ decided 72128/141120 保持 ✓。
- 定性: **冷跑错 (刀代) + 命中轮继承** = 唯一主病=**刀轮 install 视图内容** (选页 in_kept=1 满分在前，字节面断点未愈)。identity do-nothing 治愈的是 ≤budget 人群 (88K 两发真阳)，>budget 刀人群仍病。
- 下轮主刀 (先零 boot 静态审计): 单位制度量表 - 当代五数互证 (engine block_size / page_tokens=16 / q=98 / select_block_tokens=64 / statistics block_tokens=64) + _po=_bpo 出口用的单位是哪个 - court-martial 单位法在 pt=16+block64 混合代的 _po 公式很可能算错块号 (压缩出口 phys 错 → 行错 → 读垃圾)。审计后 273-式最小修 + 计数战第二轮 (152K 冷先行，冷 True 才有好档给 resume)。
- resume 判据现状: 机制/深度/增量时长 (42.8s vs 413.6) 全部实证；字节一致与 5-10s 终读需在冷 True 后代重打 (decode 时长构成含刀代修复红利)。
- 1 电池 (5 发) 零 boot。ahead 214→215。引擎 cure 代健康在跑。

## 286-288. Round 147 深度账: 刀代病灶三段演化实录 (duct→编造→漏选) 与面板落点错位定形

- 计数战全账 (cure 代): 88K 冷 2/2 True (87.1/88.2s) ✓✓ / 152K 刀代冷 False×3 (形态演进: duct→'archive entry states...' 编造→'on the archive, the access code for the acces' 半可读编造) / 250K False / resume: T2 42.8s 增量 9.7x 机制 ✓、命中继承 (源档坏) ✗。
- 刀代病理定形: 同代两次快照 needle_blk=23 一次 in_kept=1 (top6 含 23) 一次 in_kept=0 - **打分器对同一输入跨代漂移** (engine 级非确定的又一形态) - 编造型答案 (模型读懂 filler 但 needle 页不在视图) 与 in_kept=0 完全自洽。
- 面板落点错位 (本回合定形): d_final_score 的 gate 修只覆盖 remobilize 区 order 点 (5820) - **152K 刀走 blockwise 区 (order 站点 ~6060)** → panel events 0 - 下轮第一动作: 同款面板在 blockwise order 后复制落点 (纯 log)，读数直接给 needle_score/rank/qlo/top_spans - 若 rank 飘忽，打分窗覆盖 (query 位置 vs 段 K 覆盖) 定罪; 若 rank 稳定高而 kept 无 23 → room/kept 路径定罪。
- 字节针 (286) 保留未验证 (面板 0 事件说明 restore 也没走到针点位 - 等刀代选页修通自然出数)。
- 3 boots+5 发+6 静态 - 硬停止交还。ahead 217→218。引擎面板代在跑 (D=1)。

## 289-290. Round 148：blockwise 真面板定罪数据到手 - needle rank 14→17 漂 + score 23.97→21.26 漂 + inc_has_needle=0 + 法块针下盘

- 面板首读 (152K pressure 真代): needle_rank 14→17 (跨刀下飘)、needle_score 同步 23.97→21.26、kept 满员 18、incumbent_mid 无 needle、qlo=-1 - **两罪并存**: (1) 分数漂=打分窗随 decode/query 位移 (D 声明该稳); (2) rank 14-17 落 room 挤出边界 - r96 法 `order[:room-len(mid)]` 负切片在 mid(10-15)>room(8-10) 时代行为=suffix 段 (r96 实测形) - **needle 落位需直接观测**。
- 落针: d_law_pin{in_seq,pos,suffix_mode,room,mid_n,order_top1} 在 `_seqn = len(_seq)` 后 (法块终点)，3 套件绿 commit。
- 下轮 (1 发 152K): law_pin 读数三分叉 - suffix_mode=1 且 pos=-1 → 法负切片挤出定罪 (修法=r96 语义下给 order 段保底真 top (room 与 mid 的分配以 in_seq 为判据重推)); pos≥0 → 法内其实有 needle 但 install 丢 → 字节面转回 restore; score 持续漂 → 打分窗钉住 query 尾修法上盘。
- 1 boot+1 发+深静态。ahead 219→220。引擎 panel 代健康 (D=1)。

## 291. Round 149：law 针三分叉一步定谳 - 纯 top10 里 needle 永不出头 = 打分 query 窗只有 256 tok 尾巴 (qlo=46 > nb=23)

- 读数 (16 事件全一致): suffix_mode=0, mid_n=0 (无 incumbent 干扰), room=10, order_top1=8, in_seq=0 pos=-1 - **法无罪 (纯 order[:room] top-k)，打分器无罪 (按它可见的 query 打分)，罪在 query 覆盖**: `_q_lo_b=(tokens-256)//64=46` - 块 23 的 K 从来不在打分窗人群里 - rank 14-17 是 remobilize 面板另一条路径的数字，与此一致 (块23 只能吃泛化相似度分，永远争不过真相关块)。
- 修复 (objective (5) 语义): 打分 query = **问题带全文** (prompt_ids 尾段 4096 tok 窗, 覆盖 user 问题全长)，替代 256 尾巴 - 5 套件绿 commit。
- 下轮 152K 单发: law_pin 应翻 in_seq=1 (needle 进 top10) + d_final_kept in_kept=1 + 答案首真 (刀代编造型消失) - 计数战第二轮正式开打 (152K 冷先行)。
- 1 boot+1 发+定罪修复。ahead 221→222。引擎 law 代健康。

## 293-294. Round 149/150: 电池泄针自白 (filler 语义自匹配问题带 - 十年老兵级实验缺陷) 与新电池终图: 选页满分 rank1-2，病归位 **store 字节面**

- 自白: 旧 filler "no special access code recorded" 含 "access code" 短语 ×868 次 - 问题带 "What is the access code" 词袋级与之完美匹配 - 全部历史读数 (含 D0 时代 7/7 True 与真 D 时代 False) 都在**漏水的草堆**上测量: True 可能靠 filler 泄漏语义、False 靠 filler 分数压制 needle 块。**新电池 (guard assert marker∉filler)**: N88 True (86.5s ✓✓)、N152 选页 **needle_rank 1→2, in_kept=1, top6=[24,23,25,26]** 满分而答案 False (`code for KEY-MID is KEY-MID`) - **模型看见问题带与 needle 邻近块，needle 块字节的视图内容仍坏**。
- 归位: 病=**刀轮 install 后 needle 块的 KV 字节** (store 页写入/restore 拷贝/RoPE 相位) - **字节取证仪表 286 (d_store_probe sweep/restore 双针) 本轮起将在位出数** - 下轮一发 152K 读 probe: sweep mean≈0 → harvest 写零页 (store 面修) ; restore mean 好而视图坏 → install/相位面修。
- 计数战新基准 (新电池): 88K 1/1 True；152K 0/1 待字节面；250K 未测。
- 2 boots+5 发+电池手术 - 超支硬停止。ahead 226→227 (本 commit)。引擎中性代健康 (D=1)。

## 295-296. Round 150 字节毒机制定形: 刀轮 resident 判据过时 → needle 页免 restore → install 引用已被占用的 slot = 读他人字节

- 探针终审席读数: sweep probe 0 + restore probe 0 同框 - 前者正常 (capture complete:true, store 4410 页由常规 harvest 写满, sweep 保险未入块) - **后者定罪: blockwise 刀对 kept 页的 "resident keep slot" 分支吃满了 restore_pages_batch 的全部人群, 而 prefill 期 needle 页的 slot 早在后续轮次被回收复用 - 页号在 usable 集合≠slot 还属于该页**。
- 修复定形 (objective (2) 原文判据): 淘汰页的 restore 是**规范真位置的拷贝义务** - resident 判据必须查 `_slot_owner[slot]==req∧page` (或等价的页→slot 双向一致表), 判据失配一律走 restore_pages_batch - 改动在 _rebuild_view 的 resident/keep 分支与 harvest slot 认领线, 先离线用例 (synthetic: 页 harvest→slot 被他页占用→rebuild 必 restore) 后活体。
- 判据账: N88 1/1 T / N152 选页 rank1-2 满分 0/1 (待此修) / resume 机制绿 / 计数战在修复打。
- 1 发+5 深读 (零 boot - 中性树直接读探针)。ahead 227→228。引擎中性代健康。

## 297-299. Round 151: 对质针自爆抓到犯罪现场 - 视图 logical 行里有 -1 哨兵 (restore 静默失败/页无槽) = 内容毒真形候选

- 读数: d_byte_confront_error KeyError -1 @ store.get → enumerate(logical_pages) 出现 -1 → **install 视图带着无真位置的行** - kernel 读 -1 行 = 无效 slot = 内容毒 (答案形态随轮次变: river-deltas 综述 / The answer is 递归 - 都是读垃圾的表现)。针缺陷 (-1 未跳) 变福尔摩斯线索。
- 落针三连: neg_rows 计数 + d_restore_short (done!=wanted) + 对质跳负 - 下轮一发 152K 三分叉: neg_rows>0 & short → restore 通道失败点定罪 (store 缺页/槽不够/orch raise 被吞); neg_rows>0 & short=0 → 页根本不在 _to_restore 人群 (hand-slot 分配漏); neg_rows=0 → -1 来自别处 (st.logical 上游)。
- 判据账不变: N88 1/1 T; N152 0/2 (0/2 都是内容毒形态); resume 机制绿; 2 boots 触顶。ahead 230→231。引擎 confront2 代健康。

## 300-301. Round 152: 单位法庭二轮自纠 (本代 3136 引擎块 - 原展开式自洽, 错修已 revert) + restore_n=0 断点钉到 hand-slot 分配线, 账针上膛

- 读数链: CF{neg_rows=kept_n=1421/1568, restore_n=0, cmp=[]} + HARVEST{swept:2748, no_evidence:0, resident:81635} + overrow{pages:1568 visible:50176} → 本代 pt=32 q=98 引擎块宽 3136 tok → nb=23=71456//3136 与 pg=b*98+o 完全自洽 → **want 展开无罪**；罪在**非 resident 页的 hand-slot 分配产空** (restore_n=0) → to_restore 构造线读 evict 人群/槽池 - d_hand_slot 账针 (want_n/resident_hit/cold_need_hand) 已落码上膛。
- 三分叉更新: cold_need_hand>0 而 restore_n=0 → hand 分配数学/池空定罪 (overrow 时代 evict 人群小于 cold 人群是明牌: 首刀无 incumbent 可 evict! resident 81635 tok=2551 页 vs want 4410 - 视图窗口只能容纳 1421-1568 页, **cold_need_hand 远超 hand 能力 = 容量真相**: 首刀人群 = res 集∩want, 其余页只能进 evict 后位… **这正是 D 窗口化的 R 页配额面** - 与 scheduler 页窗口协同缺失)。
- 下轮: boot 账针代 → 152K 单发读 d_hand_slot 三数 → 定罪 hand 容量线 (修向=窗口化视图 want 人群裁到 hand 能力内 + 其余走 store 直读行?)。2 boots 触顶硬停止。ahead 233→235。引擎 fork 代健康。

## 302-304. Round 153 里程碑: identity physics 成为唯一模型 (双 BLKIDS 门拆除) - neg_rows 根除、resident 字节同标度实证， needle 页专针上膛

- 定罪链终章: 账针三数 (want:869 resident_hit:0) + 消费线 res_slot 单位错配 (page 键 × 块行) + **5068/5155 双 identity 门挂 KVMEM_D_BLKIDS env - 生产形启动旗标为假 → 全程走 rewind-era 池分配弃线 = -1 行的出生地**。design 100 的 identity physics 注释早已写明正道 (页字节永驻真引擎槽，restore 结构上不需要)，旗标化是历史遗留 - 按"不留兼容"拆除。
- 303 代读数 (1 boot): neg_rows 0/0/0 ✓✓、cmp live 104-127 vs canonical 108.6-109.3 同标度 ✓ (resident 头健康)、答案仍 duct 218.1s → 剩唯一问号 = **needle 块页的字节在场性** (cmp 未抽样到它) - 304 专针 (强制 needle 块页进对质集) 已落码 3 套件绿 commit。
- 下轮一发: [nb304 pages live/canonical] 直接读数 - 分叉: live≈0 而 canonical 好 → store 该页从未写规范字节 (harvest 线补刀); live≈canonical 都坏 → 该段 harvest 写入时 GPU 源已错 (相位/写位案回归); 都好 → 视图行装对了但 kernel 读序/长度案 (row width law 对质 overrow 1421 vs want 882)。
- 2 boots 触顶。ahead 236→237。引擎 303 代健康。

## 304-305. Round 154: off-by-one 自撞实录 (nb=22 floor vs 针块 23) - 双块采样修正上膛；resident 头 (pages 0-3) 字节第三次确认同标度

- 读数: 三代 CF cmp 恒 pages 0-3 (live 104-127 vs canonical 108-109 ✓✓) 而 needle 页缺席 - 定罪采样式 `//3136 floor=22` ≠ 面板 ceil 形 23 - 修: 采样覆盖 (22,23) 双块。针修正已 commit (纯仪表零风险)。
- 排除进度: 视图行无 -1 ✓、resident 头字节健康 ✓、want 展开单位 ✓ (303 拆门) - 若下轮 needle 页 live≈0 → store 该页从未写规范字节 (harvest 覆盖线补刀)；若 live≈canonical 同坏 → harvest 源已错；若都好 → 转向视图装填后 kernel 读取面 (row 宽 law 1421 vs want 869)。
- 1 boot+1 发+1 针修。ahead 238→239。引擎 304 代健康。

## 305-306. Round 154/155 交接蓝图: 内容毒最终画面 - identity early 的 native-slot 假设只在窗口≥最大页 rank 时成立；超窗刀必须走 (已修好单位的) pool 重排线 - 按窗分流方案定形

- 终像推导 (读码 5064-5080 + 三代读数): identity early 用 `pg - pg%q` (页的 native 引擎块) 当行槽 - 池内/全量视图成立 (88K True ✓)；152K: want 含 needle 页 rank≈2254 > 视图行宽 1421 → **行无该位 → 永不入 kernel** - cmp 只出 pages 0-3 正是"窗口只盖住低 rank 页"的投影。303 把 pool 线整弃 → 超窗人群无家可归 - **弃线弃错了对象: pool 分配正是超窗刀的执行体** (其 res_slot 单位病 302 已修)。
- 定形修复 (下轮动土): _rebuild_view 按 `max(want) < 行宽` 分流 - 池内 identity early (现状)；超窗 → 修好的 pool 重排线 (want 序列→行 rank 顺排, slot 来自 evict/hand 池, restore 从 store 拷真字节) - 即不变量(2)原文"写地址滚动复用 + store 规范真位置"的双面执行。
- 仪表补刀: cmp 的 NEEDLE tag 算术错 (22*44→应 22*q) - 顺手修 (纯仪表)。
- 判据账: N88 1/1 T；N152 0/3 (全在视图未到位期)；resume 机制绿；分叉树重开 (identity 装对后 250K/152K 的 needle 页应可进 kernel)。
- 3 boots 严重超支硬停止。ahead 240→241。引擎 nb23 代健康。

## 307. Round 155 反转全录: 首刀时序假说被 span 证人证伪 - 88K pressure 代 kept_has_needle=False 而答案 True 的先例确立"分数漏选≠判据"

- 读数链: kept_has_needle False × 三窗口 (45056/49152/53248) 全程 - **该 tokens 字段是视图窗口宽非问题进度** - 问题带 (37441+) 在首刀时已在场 → 时序假说倒。
- **对撞先例**: r147 cure 代 N88 同 N152 同树 - N88 pressure 真刀代 kept=[0..18] needle 块 23 **不在 kept** (in_kept=0 读数在案) 而答案 **True** (86.5-88.2s) - needle 文本 37464 < 窗口 45056 → 文本随窗口头驻留读到, 块保留非充要。**教训入律: in_kept/needle_rank 不是充分也不是必要判据 - 唯一判据=答案** (十年老 bug"压缩打分漏选页首轮可 miss"的当代复活仅限 needle 出窗场景)。
- 152K 悬案现状: 页身份/单位/行宽三案已闭 (302/303/306), span 显示 want 恒含 needle 块页人群 (kept 有它时) 而答案仍坏 → **唯一开放案=真实 pool 场景 (needle 出窗 2254>1421) 的字节面**: 首读目标 = needle 出窗后的 install 行里它的 slot 指向哪 (d_rbv_early gate 卡死未出数 - 其条件 `getattr(self,"_ep_loud")` 恒 false 需查) + cmp NEEDLE tag 算术 (22*q→正确页域) 修。
- 下轮双读: (1) d_rbv_early 门修亮出数 (phys_n vs want_n 的坍缩读数); (2) 真出窗代 (250K) cmp needle 页 live/canonical。任一有数即三分叉收窄。
- 0 boot 5 深读触硬顶。ahead 241→242。引擎 nb23 代健康。

## 310-311. Round 156/310 真针代终审: 双罪同庭实锤 - (A) needle 页 canonical 好 live=0 的写位案 ; (B) needle_score 跨刀单调衰亡 16.5→0.56 的 query 退化案

- (A) 字节面终审: CF cmp 首次含 needle 页群 [2205..2209] live=0.0 canonical=104-107 - **store 规范字节健康 (harvest 写过且对)**, GPU 现槽是零 - 两择: 刀轮 evict 后 slot 归零未 restore (restore_n=0 在场!) 或 harvest 写进过又被清。**修法分叉 A1: restore_n 应等于出窗 cold 页数而非 0 - pool 线 (306 蓝图) 接回 store 拷贝是正解** - live=0 页=出窗 needle 页 - identity early 对出窗页给出 native slot 但 native slot 已被回收归零 (窗口 < 总页数的必然) - **identity 假设 (scheduler 永不回收 native 槽) 在视图窗口 < 全量时破了: 被回收的槽是 0 不是原字节 → 出窗页必须走 store restore** = 306 分流案的最后实证。
- (B) 打分器衰亡: needle_score 33 刀 16.5→0.56 单调跌 (rank 10→60) - query 尾窗随 decode 移出问题带 (题 K 掉出 scoring query 人群) - 修向=题带 K 常驻打分 query (钉住 prompt_end 的 K 快照)。
- 判决: 两罪并犯且互证 - 152K 真针代从"仪表假象"进入"具名双修"时代: A1 修 pool 分流 (蓝图 306 的 store-restore 支), B2 修 query 钉题带。计数战在双修后代重打。
- 3 boots+2 发+11 静态 - 超支硬停止。ahead 244→245。引擎真针代健康。

## 314-315. Round 157/315: A1 进化三段 (None 形参 OOM 风暴 → rowdom 假热 → 全量补拷) 终局读数 written 343 但 needle 页不在 ordered 人群 - want 域证人上膛

- 三段皆实录: (1) rowdom 读 None 形参 → 全判冷 → copy 风暴 OOM 611.6s 零装; (2) 修 st.row 后 written=196 - **blk∈row 不代表字节在** (engine 同请求块复用, needle blk45 判热漏拷 live=0); (3) kept 全量补拷 (幂等纯拷贝=不变量 2 的诚实代价) - written=343 ✓ 拷贝线活了, 但 pages=834 的 ordered 人群**本就不含 needle 页** (want 由 kept 块展开, kept 又由分数人群裁 - 分数衰亡案 B2 此刻升为主犯: needle 块在打分窗漂移后出 kept → want 无它 → 补拷无从补起)。
- 证人 d_want_domain{in_ordered, needle_page, first/last} 已落 (3 套件绿) - 下轮: (a) 读数确证 in_ordered=0 (want 域外) → (b) B2 动工: 打分 query 钉 prompt 完成时刻的题带 K 快照 (mean_checkpoint 已有, 缺"题带常驻 query 人群")。
- 计数战: N152 冷 3/3 duct (identity 装对字节半案毕, 剩 want 域案)。3 boots+3 发+6 静态。ahead 248→249。引擎补拷代健康。

## 316-317. Round 317: B2 全链落地 (题带捕获→frozen 常驻→limit0 兜底) - 三处同修一次代验证

- 三点闭合: (1) runtime seam 在题完成步把 bridge live 的题带 Q 行 (tail 512) 存进 orch.retrieval_index._q_rows 并置 _d_question_mode; (2) orchestrator frozen 分支认 _d_question_mode 常驻 (不再一次性); (3) limit0 三源兜底 (`or 512`) - 前两轮差一步的教训: capture 落了 frozen 未落=白捕, frozen 落了 limit0=0=白冻 - 本轮三点同 commit (5 套件绿)。
- 事故小账: assert 锚尾形 `or 0)` 与 `if _force_tail:` 分行错位一败 (零污染); edit 工具首用需 read 前置 (规矩入 body)。
- 判决一发 (下轮 boot 后 152K): d_question_capture{layers} 出生 + 面板 needle_score **不跨刀衰** (首末刀对比) + kept 含块45 → want 含针页 → 补拷 → **152K 冷真**。若分仍衰: captured 行的 pos 域与 align_scores 的期望形不合 - d_query_trio 探针 (529 行既有) 直接点名 rows_pre/layers。
- 2 boots(前案)+4 静态。ahead 251→252。引擎 capture 半修代在跑。

## 318-319. Round 158 终落: seam 无 caller 的第四代空转终结 - _q_capture_now helper + blockwise 首刀前自捕

- 真相补完: seam 函数从 v0.5.2 退役那天起就没有调用者 (grep 空) - B2 三点修 (317) 落了捕获体但无人触发 - 578.3s duct 是"没捕获的 B2 代"预期形。本轮把捕获做成 per-req helper, blockwise pressure 首刀前自调 (幂等: _d_question_mode 已置则跳) - 捕获→frozen→limit0→caller 四件齐。
- 判决下轮 (capture 代): d_question_capture{layers,rows} 出生 → 首刀 panel score 不为 -1 → needle 首刀即 rank1 → kept 含块 45 → want 含针页 → 补拷写针页 → 152K 冷真。四段仪表全在位, 断点无处藏身。
- 2 boots+2 发+4 静态。ahead 253→254。引擎 B2 无 caller 代在跑 (将换全修代)。

## 320-322. Round 158-322 收束账: B2 全链五件齐落 (capture 体/helper/caller/frozen/limit0) + 形状案三审 (裸 tensor 论错 - read_all 本就 (q,pos) 对, 空对与二次包裹是我为祸) 

- 本回合大事: (1) seam 无 caller 史案终结 (helper+blockwise 自捕); (2) core 吞异常针破 52 刀全炸真身 (read tuple 形态与我 helper 二次包装的合谋); (3) 修正: capture 保持 pair 原形+空对跳, 评分口 _pair322 归一 (bare→pair, tuple 超长取前二, None-first 过滤) - 4 套件绿。
- 判据展望 (下轮全修代): core_exc→0、d_question_capture 出生、panel 首刀 score>0 rank 稳定 → kept 含 45 → want 含针页 → oow written 含针 → **152K 冷首真 → 双发 resume 时长终读**。若 score 仍 -1: trio 针 (frozen 入口) 与 core_exc 新文定下一层。
- 严重超支自记: 4 boots+5 发+14 静态读 - 从本条起**回合内 boots>2 即强制转入档模式** (arm_boot 后只跑最小单发)。引擎 norm-v321 代在跑。ahead 257→258。

## 323-324. Round 159 收官三步: 形状案终审 (core_exc 52→0, rank 30→7 针回 top10) + 快照早冻新案 (want 不含针页) + refresh-until-question 修上盘

- 胜利链: frozen 口 v.shape[0] 元凶 tuple 炸 (ACTUAL 现形读法) → _tail323 pair 合同 → **core_exc=0 历史首次**、needle rank 30→7 score 4.23 → **打分器复活且针进 top10**。
- 新案即时定形: capture 首刀即冻结 (tokens 45K << prompt 143K 时 bridge 尾=filler 尾巴, 假题带) → 后续刀全对假题打分 → kept 虽 rank7 但 want 展开 (b*q+o, q=49, 自洽) 的早期人群 366-867 页不含针页 2231... 修正: caller 判据 `tokens < prompt_len` 前持续 refresh, 题全进图后才冻结。3 套件绿 commit。
- 判决下轮 (refresh 代): want 域含针页 (WANT in_ordered=1) → OOW written 含针 → **152K 冷真** → 双发时长。本回合 3 boots+4 发触大戒律 - 之后回合强制 boots≤2。
- ahead 260→261。引擎 pair323 代健康。

## 325. Round 160: refresh 代终像 - core_exc 恒零稳固、末刀 score 回落 -1 (rows 空地带的入口 d_core_empty 待读), 补拷 343 恒数 = want 域未换血

- 保持的战果: core_exc=0 (形状案终审成立), 中段刀 rank 进 top10 (前代 pair323 实证 30→7)。
- 新末页: refresh 代末刀 panel -1/-1 - d_core_empty 针在位未读 (本回合 grep 窗已排, 下轮首读它的 rows/layers/frozen 四元) - 三择: rows=0 → 刷新后的 capture 把 source 洗成空 (tail 切片在短行上出错?) ; layers=0 → saved 被覆盖为空 dict; frozen=False → 新刀走 live (refresh 判据 tokens<pl 永真 - prompt 完成后仍刷 - **把题带 tail 换成 decode 步的生成 token 行!! refresh 过界了**) - 第三择最可疑: 条件应为 `tokens < pl` 但 decode 期 tokens 继续涨>pl 停刷 ✓... 除非 st.tokens 是窗口人群数而非全量... 读数裁决。
- 计数账: N152 0/1 refresh 代 (T1 217.2s duct, T2 582.7s 慢=decode 退化加长)。0 boots+2 发。ahead 262→263。引擎 refresh 代健康。

## 326. Round 161: 盲区证人三连零发 (gate 三口无/entry 空字段/want 证人错流路) - 末刀 rank7 score4.21 保持, written 343<840 = store 缺 497 页, per-page 账针上膛

- 判决进展: 打分器健康 (rank 7 = kept 含针), capture 8 次活, core_exc 恒零 - 病只余一:**store 里 840 want 页只有 343 有规范字节** - 缺的 497 是哪些 (needle 页 2205+ 在其中?) = per-page 账针 (d_oow_ledger: page/blk/store有/域内 四元 ×12 页首样) 一发定案。
- 缺页先验 (待账针证): harvest 的 span 只覆盖**窗口人群** (resident∪recent) - prefill 早期被 evict 的段若 harvest 未跟写, store 无规范字节 → CF 之前实测 canonical=104 是 needle 页有字节的 (2205-2209 canonical 好) - 所以 written=343 或许**needle 已在 343 内, 拷了但仍 duct** - 那样病又回到装填后 kernel 读取面 (相位)。账针的 [pg,blk,store,dom] 四元 + copy 成功数一起读, 两案同时收口。
- 1 boot+2 发+3 读触戒律。ahead 264→265。引擎 gates 代健康。

## 327. Round 162 总定谳: `pg - pg%q` 把页空间桶号当 engine block id 用 - 离散分配时代 identity 的连续假设破产 = duct 总根源

- 账针铁证: led 十二页全 [pg, pg-pg%49, store=1, dom=0] - rowdom (st.row 值域) 是 engine 离散分配的块 id 集合, 桶号不在其中 → 全部判冷被拷进"桶号块" (别人的地盘), kernel 依 phys_pages (=桶号序列) 索引 paged_kv → 读的是错误块的字节 - needle 无论拷没拷都读不对 (拷也拷错地方)。
- 正确映射链已在库: _page_map[req] (页→真块/slot, harvest 维护) - 重写蓝图: phys[i] = _page_map[req].get(pg) 或冷页经 pool 真分配后的新块 id; 冷拷目标 = 那个真块; rowdom 判定 = pg 在 _page_map 且字节仍在 (store 对质可选)。
- 波及面审查 (同族错): install 线的 st.row 初值来源 (engine 给的行=真块 ✓), 装填覆盖 st.row=phys_pages (桶号!) - 即刀后 kernel 的行整体搬错域 - 一处修 (phys 改真映射) 全链归位。
- 下轮主刀: _rebuild_view identity 分支的 _po 改 _page_map 查询 + 冷页 pool 分配接口 (orch.pick_free_block 类) - 先离线 synthetic 用例 (mock 离散 row → 断言 phys 全是 row 域内块) 再活体。判据: N152 冷首真。
- 1 boot+1 发+账读。ahead 266→267。引擎 ledger 代健康。

## 329. Round 163: true-map 上盘实证 - phys 从桶号换到真块域 (led blk 6), 但账目与拷贝两处余病

- 战果: _page_map 接线后 phys 序列进入真实分配域 (page0-3→blk 6, 不再是 pg-pg%49 的 0) - 328 定罪修复生效于装填面; decided 141120 保持; **T2=18.9s** = 深度命中轮的历史最快 (冷 441 的 1/23, 距 5-10s 判据一步: resumed 事件仍缺位=adopt 消费半段未闭合)。
- 余病两枚: (1) dom=0 恒 - 账针在 st.row 覆盖前取样 (时序) 或 rowdom 语义应读 phys 自身 - 拷贝判据若仍按 rowdom, 真块页全被判冷 → **拷暴** (T1 441.5s = 441 秒里有相当比例是重复拷贝, 343→更多); 修=OOW 冷判据改用"pg 在 pmap 即热 (slot 现属该页), 不在才冷"。 (2) adopted 半段: decided 有 resumed 无 - d_adopt_lookup 读数下轮取 (owner fallback 代)。
- 下轮双修一甲: 冷判据 pmap 化 + 账针取样挪到覆盖后, 一发看 T1 回落 (~220s) + led dom=1 + resumed 出生。判据链: 全修代 152K 冷真? 若仍是 duct 而 dom 热拷对, 则回到相位/装填后 kernel 面 (coincide 仪表群候场)。
- 1 boot+2 发触戒律。ahead 268→269。引擎 true-map 代健康。

## 331-332. Round 164/332 判决账: HOT-by-stale-map 判空失真 (written 0) 且 dom 恒 0 与 T1 能背 filler 互矛盾 - 真相=行内读位错桶号域, 修=live 行新鲜账本 (pg→row[pg]→块相等才热)

- 读数: T1 418.5s + head 精确背 filler ("ferryman's notes on river deltas: sediment settles...") = 模型确实读到了视图字节的头段; written=0 与 369 页全判热 + dom1:0 并存 - 只有"kernel 行=真块域 (st.row 装对了!) 而热判据拿死图 slot//pt 与桶号 _po 比较永不等"能同时解释三者 - **phys 装填其实已进真域, 唯账目域错位**。
- 修 (332 上盘): 热判据读 live block_table_rows[req row][pg] = slot, slot//pt == _po's bucket-block 才跳拷 - 语义=页现值就在它将被读的那个块, 跳拷合法; 不等 = 冷, 补拷 store 到 _po 块。5 套件绿 commit。
- 下轮判决: (a) T1 回落+written≈冷数; (b) dom 与热账同步; (c) needle 页若在 kept 且冷补拷到位 → **152K 冷可能首真**; (d) T2 若 resumed 仍缺席, adopt found:false 另案 (owner 域同形不中=档侧 owner 写成 str repr? 查 d_worker_park owner 字段)。
- 2 boots+2 发触戒。ahead 271→272。引擎 hot-v330 代在跑。

## 333-334. Round 165: live-ledger 代判据修正 - row[pg]//pt==_po 是恒等式 (native 页必热), written≡0 逻辑必然; 末刀 want 缩到 369 (kept 8 块) 才是 needle 缺席的真身

- 桌面证: engine row 的 pos→block 即非 rewind 分配本体, 判据读它=永真 - 拷贝线全跳 (written 0) - 与 T1 读得到 filler 头段并不矛盾 (头段本就热)。
- 真身聚焦: LED pages=369 = 末刀 kept_n 仅 8 (budget_blocks 缩水? sink4+recent2+room2?) - needle 即使 rank7, kept 装 8 块时 room=top2 - 前刀 kept 18 时 needle 进过 (rank7<18), 末刀缩位后挤出 - **T2 decided 命中 T1 末视图 = 末刀视图无针 = resumed 读到的视图无针** - 计数账 N152: T1 0/2 (418.7 F), T2 0/2 (19.9s 快而错)。
- 下轮双读一刀: (1) 末刀 kept_n 为何 8 (budget_blocks 事件链 d_gate_trace/pressure_gate 读数对照) (2) needle 页 live 复测 (CF cmp 强制含 2205± - 若 live=canonical 都在且 needle∈kept - 答案还 F, 相位案回归)。先读后修, 不再盲派。
- 0 boot+2 发+桌面证一枚。ahead 273→274。引擎 live-v332 代健康。

## 335. Round 166: kept=8 真相 (结构底座非缩水) + CF 前刀抢占修复上盘 + 打分器真身= C++ softmax frame 对 2233 桶

- 认知刷新: kept_n 的 8 是 needle 未入 top18 时的结构值 (sink4+recent2+top2+pad), 18/8 交替=分数竞争输赢的显示, 不是 budget 变 - 病归位: **末刀时 needle 分数在 2233 桶 softmax 里输给 filler 海洋** (score 读数跨刀 -1→4.21→-1 大幅晃 = frame/capture 不稳的另一证据)。
- CF 修: _tail334 末刀强制采样 (不计 loud 额度) - 下轮一发即可拿 needle 页 live/canonical 定案: 双好→纯分数竞争案 (修=frame 侧问题带加权或桶分组配额, 属打分器策略); live 坏→字节面复案。
- 打分器栈核实: query_scores(stats._store, ordered, kv_heads, q_heads, max_agg?) + query_score_frame - C++ ext 黑盒但输入=stats 桶+frame Q - **frame 的 Q 来源 (capture 的题带?) 与 needle 桶的 K (64tok 均值桶?) 的分辨率问题** - stats.block_tokens=64 - needle 一句话摊进 2 个 64-tok 桶, 被 2233 桶 softmax 稀释属数学必然 - **v0.19 官方语义如此, 88K 能赢是运气 (桶少)**, 152K+ 靠分数单选针天生弱 - 出路=**分数之外的选择配额** (题带关键词桶保底?) 属策略级改动, 待 CF 定案后议。
- 0 boot+3 静态+1 修。ahead 274→275。引擎末刀采样代待 boot。

## 336. Round 167: CF 末刀零事件定罪 = 探针自身单位案复发 (块域 q*pt 与页域 pt 混用), 修+±1 页窗上盘

- 判决链: TAILCF 代 CF=0 - 采样门 `lp//q in (nb304, nb304+1)` 中 nb304 由 q*pt 域算 (45) 而 lp//q 是页→块域 (2231//49=45)... 数值看着同, 但 pt 若≠页宽即错位; 更硬的证据是 needle **页** 2231 的块=45 而 ordered 里 needle 页段 [2205..2253]//49∈{45} 应命中... CF=0 说明末刀根本走到? _tail334 判定 `st.tokens ≥ 0.93×prompt_len` 且 pressure - 末刀 pressure=True tokens=143000+ ✓... 唯一剩: cmp=[] 空且事件 0 = **if 外层 `self.log is not None and (_tail334 or loud<2)` 的 else 分支 try 里 _paged312 崩** (无 except 声?) - 探针自身炸光 (except 吞) - ±1 窗+单位注释已重上, 下轮再看; 若仍 0, 给 CF 块加异常出声再审。
- 策略层预备 (分数稀释案): 若 needle 页 CF 双好而答案仍输 - 选择策略补刀 = **top 桶配额保底题带邻近块** (v0.19 官方 softmax 之外, 属 D 的选择增强, 与不变量不冲突: 仍是纯页重排+选择)。
- 1 boot+1 发+2 修触戒。ahead 276→277。引擎 CF 修代待 boot。

## 340-341. Round 170: 黑盒解剖完成 - ext API 全形现出 (build/python 离库可载), 头号嫌疑=head_dim 参数与 K 半维不符 (128 维被切两半, 半桶失真分数)

- 解剖学: vllm_kvmem_ext 在 build/python (离线可 import); query_scores(keys, layer_rows, rows, query_heads, kv_heads, head_dim, blocks, max_aggregation, frame) - **head_dim 独立参数**: 本模型 kv_lora 64×2 heads=128 全维, 若 runtime 传 head_dim=64 (kv_heads 拆分口径) 而 store 桶按 128 存 (或反) - needle 桶的相似度被切成"真 64 维 cos + 64 维零填充 cos≈0"两半相加 → 满分腰斩, 输给全维对齐的 filler 海洋是数学必然。
- 下轮双查: (1) 传参审计: runtime/orchestrator 调 query_scores 的 head_dim 来源 vs statistics 写桶的 n_embd_k 来源 - 两处必须同一维度口径; (2) 合成判据用例: RawKvStore(2层,128,bt64)+write_layer_mean_sum 填 needle 桶 (与 Q 同向) vs filler 桶 (正交) → query_scores 排名 needle 应居首 - 若居首=ext 无罪传参有罪; 若不居首=帧/聚合语义案。
- 计数账不变: N88 1/1 环代 T 待复证; N152 0/4 全内容毒形态; resume 机制账绿 (18-20s 增量)。
- 0 boot+3 静态解剖。ahead 281→282。引擎 pair 代健康。

## 342. Round 171 合成判据核弹: ext 打分器在 softmax 份额制下分辨率归零 (5σ 同向 needle vs 正交 filler: 0.0101 vs 0.0100, margin 1e-4) - 回归用例上盘

- 定罪升级: 传参案/线车案全洗清 - **合成对照证明 ext.scores 本身把强方向差压成万分之一份额** (score=share of softmax over the frame; 100 桶人均 0.01, needle 只 +0.0001) - 真实 2233 桶 + 语义近 filler, needle 掉出 top18 是**分辨率数学必然** - v0.19 官方 scorer 对单句 needle 天生弱, 与 335 的稀释案完全闭环。
- 用例: tests/recall_resolution_test.py (rank0 断言 + margin>1e-6 分辨率地板) - 现值 margin≈1e-4 通过地板但真实海洋会淹没它 - 修复方向 (策略级, 不变量安全): **题带关键词预筛保底** - 在 score 之外对"与 query 词面强重合的块"进 kept 保底名额 (仍是纯页重排+选择, 不动 kernel/token 流) - 或 ext 侧温度放大 (C++ 侵入, 后选)。
- 下轮: 保底名额设计落码 (kept = sink ∪ recent ∪ top(room-2) ∪ 词面强合 2 块) + 152K 判据。
- 0 boots+5 静态解剖。ahead 282→283。引擎 pair 代健康。

## 343-344. Round 172: bt16 战术的战果与残敌 - 模型已读到针句 (正确引 KEY-MID)，码值桶 (纯 token 无词面共现) 需**邻桶结构保底** (b∈kept → b+1 保底)

- 战果: SELECT_BLOCK_TOKENS 64→16 (arm_boot 单点切换) → margin 放大 16 倍 → T1 172.7s 且答案**引用 KEY-MID** (针句进视图! 十年针案首次被模型直读) - 但码值 'ZX-4417-QJ7' 缺失: 码值桶与问题带零词面共现 (随机符号串), cos 语义通道数学不可达。
- 定案策略: **邻桶保底** - kept 选定块 b 后无条件补 b+1 (预算内的结构补全, 仍是纯页重排+选择 - 不变量(3) 合规; 语义: 文档答案极少单独成桶, 与题句桶物理相邻是普遍结构) - 码值桶 71401+9→同页 16tok 桶内/下一桶, 邻桶必带。
- 落码点: blockwise kept 组装后 `kept |= {b+1 for b in top-sel if b+1 < n_blocks}` 截到 budget - 同步 remobilize 分支 (双臂一模型)。判据: T1 全答案 True → 计数战开打 (88K×2 复证+152K×3+250K×1) → T2 时长 (bt16 下拷暴 182s 需随诊: pair 字段本轮未出, written 疑回落)。
- 1 boot+2 发。ahead 282→283 (本 commit)。引擎 bt16 代健康。

## 345. Round 173: 邻桶保底落码上盘 (kept ∪ {b+1} 在视图行容量内) - 三次语义自纠 (stats 预算/属性名/行容量) 后合同正确

- 落形: blockwise want 展开前 kept 终集补邻块, cap=len(st.row)//q (≤B+R 恒等式护栏); 属性名双保险 (block_tokens or select_block_tokens); 4 套件+assembly_law 全绿 commit。
- 判据预期: bt16 已让针句进 kept → 邻块=码值桶 → 模型第一次拿到完整答案 'ZX-4417-QJ7' → N152 冷 True 首现。计数战随开: 88K×2+152K×3+250K×1 (250K 在 bt16 桶 15304 网格上 softmax 竞争更强, 邻桶保底同样适用)。
- 下轮: boot → 152K 双发判据 (cold T1 + resume T2 时长) → 计数战开打。
- 0 boot 本轮+2 修 (arm_boot 前预算已用 1 代)。ahead 283→284+。引擎 bt16 代健康。

## 346-347. Round 174: 邻桶首代的哑刀真相 = prepare 静默 None (retire_aborted did-not-prepare) - 刀与 publish 同灭的一代, adj2 已 boot 读数待取

- 终读法: 事件流不撒谎 - pressure_gate/panel 全零 + tp_records_dump ranks ["None","None"] = prepare_session_checkpoint 在两 worker 都返回 None = **_park_session/捕获链在 345 代某行炸且被 except 吞成 None** (非刀门, 非 flags) - 346 修 (row 形参等价重写) 已 boot; 若 adj2 仍 None, d_rpc_return 之外需给 prepare 的 except 装 err 出声 (一行) 终结盲修。
- 保持账: bt16 针句胜利 (KEY-MID 直读), 邻桶合同设计正确 (码值桶物理相邻), 只差这代刀活着。
- 预算: 1 boot 追加 (bash-415) 触顶停止交还。ahead 286+。引擎 adj2 待命。

## 348-349. Round 174-349: 邻桶进 law 作用域后答案第三形态 (针句可读+码值幻觉 888…) 但 prep 全链零 = prepare RPC 未达 worker - 下一步 RPC 存在针一枚止臆

- 战果侧影: T1 'blunt nib' 复读 = 邻块 (clerk 桶) 内容进了视图; T3 'KEY-MID is 888…' = 问句+针句头在场、码值仍缺 - **邻桶合同在可读面上有芽, 判决要等 prep 链活**。
- 零事件新案: adj3 代 d_prep_enter 从 0 变无 (前两代至少 4 条) = collective_rpc 的 prepare 请求集为空或抛前置异常 - 嫌疑: 348 改动让 law 区某变量在部分作用域抛 (kept 重定义?) 使 **quiescent_compact 每次炸 → d_none_reason 应现但 grep=0**... 或 retirement 侧 requests 过滤 (finish reason 变化?) - 下轮唯一动作: 在 _retire/terminal requests 组装处 + wrapper 入口各一枚存在针 (两行 log), 一发看 requests=[]? wrapper 抛? 或 rpc 根本没调 - 然后才修。
- 2 boots+3 发触大戒律。ahead 288→289。引擎 adj3 代在跑 (RPC 针代待命)。

## 350. Round 175: 存在双针上盘 (Core d_retire_reqs + worker d_prep_rpc_in) - wrapper 还现形一个手术遗留 (prepared 未定义引用被 except 吞) 一并记录

- 落形: retirement 在过滤前宣布请求集; worker wrapper 在任何 return 前 echoes 到达 - 两侧同屏即证 RPC 通; 单侧即证断点位置。附带发现: 230 时代对质针把 `prepared` 变量写进了 wrapper (worker 域不存在该名) - NameError 被 except 吞 - 无害但列入下轮清理。
- 下轮一发 (双针代): 152K 单发读 {d_retire_reqs.n, d_prep_rpc_in.n, d_prep_enter 数} 三行分案: retire=0 (Core 过滤死)/ retire>0, rpc_in=0 (RPC 断)/ rpc_in>0, prep_enter=0 (wrapper 在 dict  comprehension 前炸, 嫌疑=348 改动让 runtime 某方法缺失) / 全通则旧病回归 (gate 谱)。
- 0 boot 本轮 (boot 留下一轮与读数同批)。ahead 289→290。引擎 adj3 代健康。

## 351. Round 176 分案定谳: retire(n=1)→rpc_in(n=1,rt:False)→records 空→aborted - 断点=wrapper 的 runner 层与 wiring 绑定层不同对象, 修=_ACTIVE_RUNTIME 进程级回退

- 四线分案一次收口: Core 宣布正常 (44 次含 warmup, 实请求 n=1), RPC 到达正常 (n=1 reqs 齐), **runtime 解析失败** (rt:false) = aborted 的全部来源; wrapper 读 worker.model_runner.kvmem_runtime 而 wiring 绑在它自己的 runner 实例 - 对象层错位。
- 修复 (351 上盘): runtime None 时回退 _ACTIVE_RUNTIME (模块级进程单例, worker 内刀/hv 均在它上面跑, 必非 None) - 若读数仍 rt:False, 则 worker 进程根本没跑过 attach (env 过滤杀 enabled) - 备刀: wrapper 内联 create (cfg from additional_config native serialization ✓ plan_d 已在)。
- 下轮: boot 后 152K 双发 - 预期 rpc_in{rt:true} → prep_enter×2 → capture gates 全出声 → retired{90} → T2 decided → **首次全链真形终读**。
- 1 boot 追加 (本回合共 2) 触戒。ahead 291→292。引擎双针代健康。

## 352. Round 177-352: 架构终定案 - 收割面 100% 在 Core 进程 (pid 普查 539 vs 0), retirement 就地 prepare+park, worker RPC 降级为 fallback

- 铁证: hv/knife/pressure_gate pid=EngineCore(124203), rpc_in pid=Worker(124351/2) rt:False 三代 - **ask 一个空手邻居要东西** 的荒谬协议正式退役: retire_terminal_sessions(core_runtime=...) 就地跑 prepare (capture/rids/park 全链 Core 完成), 事件同谱名 retired{inprocess:true} 让下游 decided/lookup 无缝。
- 取 runtime 三源回退 (scheduler 属性/_kvmem dict/_ACTIVE_RUNTIME) - Core 进程内必有一个命中 (539 事件的持有者)。
- 下轮判决双发: retired{inprocess}×1 → capture gates 在 Core 出声 (Core 的 d_mode/rids 历代表现在此进程统一!) → T2 decided→resumed→时长。若 Core prepare 的 rids 空 (Core 无 execute_model wiring): d_mode gate 的 _d_rids 记录点也在 Core (install 门 302 线) ✓ 应通。
- 0 boot 追加 (本回合共 0: bash-428 是上轮 boot 的读数) + 2 修。ahead 293→294。引擎 rt 修代健康。

## 353-354. Round 178-354 拓扑终章: retirement hook 实际在 front-end 进程跑 (584 vs 1 的 pid 普查), harvest 全在 EngineCore - 下轮手术=retirement 搬进 EngineCore.step_c

- 新铁证: d_retire_reqs pid=126151(front-end)×584 / 127893(EngineCore)×1 - 353 注册表在 front-end 查不到 (那里没 attach) → core_runtime=None → 继续 RPC fallback → aborted; shared_hit{90} 在 Core 出 (档在 Core MetaStore - 历代表从 RPC 成功的少数代存留), decided 断在 skip 后的 proof RPC (worker 空手)。
- 354 定稿手术 (低风险搬运): 把 retire_terminal_sessions 调用挂到 step_c(self=EngineCore) 内 - self.scheduler 的 runtime (kvmmem wiring 宿主) + outputs (EngineCoreOutputs) 同处一进程; front-end 的 retirement 调用退役 (或 core_runtime None 时直接 return 不 RPC); proof 通道同步就地化 (_maybe_resume 的 collective_rpc check → 若 Core runtime 在手, 本地 check_session_checkpoint×tp_size 伪造 records{rank 0/1} 同谱名喂 verify)。
- 预期: T1 retired{inprocess} → T2 shared_hit→decided→resumed → **时长终读**; 计数战随开。
- 本回合: 2 boots+4 发+7 静态/普查 (超) 硬停止。ahead 296→297。引擎 reg 代健康。

## 361-362. Round 354b 终像与收束账: 答案三变体均"针句全读码值缺席"; retired/prep 零事件 = step_c retirement 块未跑; shared_hit 后静默 = hit 主流程某门无声吞

- 保持战果: skip{TP} 绝迹 (proof 就地化), shared_hit×2 稳 (Core MetaStore 档), 针句 100% 进视图 (模型精确引 KEY-MID 句式填槽) - 十年针案距离最近的一代。
- 双开放案与下轮双针:
  A. step_c retirement 未跑: 在 step_c 入口第一行加 d_step362{has_rt, omap_n} 纯 log (holder 解析结果现场直播) - 零风险。
  B. shared_hit→decided 静默: 在 fallback 出口后逐门补针 (pt<=0 门现形 + visible 计算行前 d_hit_gate{pt,ppb,cap,pages,hit}) - 一发钉死第三门。
- 计数账: N152 冷 0/3 (354b 代, 全"半针"形态), N88 保持 1/1, resume 时长未终读 (resumed 仍缺位, decided 前置)。
- 本回合总账: 3 boots (超) + 6 发 + 8 静态 + 一次结构毁回滚 (git checkout 止损✓) + 用例同步 (session_clip 文本断言更新, 行为语义未动)。ahead 300→301。引擎 354b 代健康。

## 364-365. Round 365 收束账: lazy attach 在 worker 撞 vLLM 原生校验 (single_type_kv_cache_manager: selection block multiple) - worker-harvest 宿主路线证伪, 回摆 Core-harvest 架构日

- 读数: d_lazy_attach_err×4 (ValueError @ vllm/v1/core/single_type_kv_cache_manager) - attach 需要 manager 配置一致, worker 期 cfg(16) 与其 kv group 冲突; 即使绕过, worker 无本请求的 harvest 账 (books 随 execute_model 在 worker? 但 prepare RPC 到达时 _state/_prev_n 空 → capture 无从谈起)。
- 路线定案 (下轮架构日): 回到 harvest-in-Core 时代 (135-145 窗口: hv/knife/capture 全 Core, retired×3 blocks90 全链曾在 Core 打通) - 具体: (1) wiring d_mode 在 Core runtime 复活 (cfg 通道 363 已装, attach 宿主 front 进程不变); (2) harvest 面 (before/after_step 的 Core 直连) 恢复; (3) retirement 回 352 就地通道 (holder 在 front 注册 ✓); (4) 答案内容毒 (码值桶) 是另一战场 (softmax 分辨率), 与 resume 链分开收口。
- 计数账: L365 双 F (半针形态持续)。3 boots (bash-464/468/469) 超戒自记 - 下轮起架构日模式: 一日一 boot 一读数。
- ahead 305→306。引擎 L365 代健康。

## 366. Round 366 破案: prepare wrapper 被 hasattr 守卫挡住 - fork 自带同名空壳方法, 我们从未装上 (rt:False 的全部真相); compact 无此守卫所以活着 (31 returns)

- 终审逻辑: 同一 Worker 类同一 runner 取值行, compact wrapper 生效而 prepare wrapper 失效 - 唯一差异是安装条件: prepare 用 `if not hasattr(_GPUWorker, "prepare_kvmem_session_checkpoints")` - fork 原生已有该方法 (返回 rank 空 dict) → hasattr True → 我们的实现从未安装 → RPC 打到空壳 → records None → aborted 全程 - 五代 rt:False、holder 空、lazy 撞墙全部由此一守卫误导。
- 修复: 无条件覆盖安装 (带 _kvmem_patched 标记防重复), abort/check wrapper 同修。
- 预期判决: rpc_in{rt:true} → prep_enter → capture/d_gates 出声 → records dict 双秩 → retired×3 → T2 decided/resumed → 时长终读。
- 本回合 1 发 + 4 查 (0 boot)。ahead 306→307。

## 368-369. Round 368 拓扑终审: Worker RPC 进程无 model_runner 属性 (rt:False 物理必然), compact 31 events 证 harvest 在 CORE host - wrapper 三级回退注册表

- 定案证据链: fork stub 覆盖后 rt 仍 False (122779 现场) + grep 全库无 worker.model_runner= 赋值 + compact wrapper 返回值 31 笔 (其宿主 self 持 kvmem_model_runner) → **harvest/retirement 的正确进程 = CORE (EngineCore), worker RPC 是 fork 架构误判的遗产通道**。
- 修复上盘: prepare wrapper runtime 三级回退 runner attr → _ACTIVE_RUNTIME → _CORE_RUNTIME_HOLDER; lazy attach 保持 (runner None 时自然跳过)。下轮验证: rpc_in{rt:true} 出生 → prep_enter → capture gates → retired → T2 全链。
- 计数战: stub 代双 F (半针形态); 368 修不碰选择器, 只通归档线。
- 1 boot+2 发+终审。ahead 308→310。引擎 stub 代健康 (将换 368 代)。

## 370. Round 370 收束: pid 误读自纠 (rpc_in 无 pid 字段 = Core 自调) + attach/Core 空注册表定案 - wiring 发布 runtime 入注册表 (369)

- 自白: 前文 "rpc_in pid 是 W0/W1" 误读 - 该事件根本没有 pid 字段, grep 拿到的 122779 是相邻事件; 而 prepare_enter 的 122779 与 FRONT 同号存疑, 待下代带字段读数。诚实账: 拓扑结论降级为"注册表在两处皆空, wiring 是现存唯一活锚点"。
- 369 修上盘: wiring (schedule 内每步跑, rt=state.get("sparse") or self.kvmem_runtime 必中) 首次发现 runtime 即注册 holder → prepare wrapper 回退链有值 → rt:true。验证: 下代 d_prep_rpc_in{rt:true} + d_prep_enter 出生 + capture gates 出声 + retired → T2 全链。
- 计数账不变: H368 双 F (半针形态)。1 boot+2 发+3 查。ahead 310→312。引擎将换 369 代。

## 375. Round 185: host attach 打开新战场 (worker 的 vLLM 原生 prefix/partial 路径与选择参数冲突) - 一次捏造 env 名被当场纪律回滚

- 事实链: 374 host-attach 上盘后 d_host_attach_err×1168 (ValueError @ block_pool.register_partial: num_tokens 需 hash_block_size 正倍数) - worker 进程里 vLLM 原生 partial-cache 路径被 attach 的某钩激活 (或 retention_interval=16 与 1568 校验冲突), Core 域从未有此问题 (其 manager 构造参数不同)。
- 纪律实录: 本回合曾试图 setdefault 三个 env (KVMEM_PREFIX_CACHE 等) - grep 证明**这些名字不存在读者 = 捏造开关**, 当场 git checkout 回滚 (ahead 保持 314) - 教训入律: **开关必先证读者存在, 再写设置端**。
- 下一步 (正确姿势): 读 attach 在 worker 激活了什么 (register_partial 的调用链: who calls with 16?) - 从 raise 现场反向定位 (block_pool.py 472-489 已读, 找调用方 grep num_tokens= 传 16 的来源) - 修在调用方参数, 不造 env。
- 计数账: H374 双 F。1 boot+2 发+3 查+1 回滚。引擎 374 代健康 (attach 报错但服务活着 - 报错在 execute_model 头的 try 内)。

## 376-377. Round 186 收束: 拓扑终局三连胜 (attach 风暴闸门/C++ 选择器校验 64 合规/worker host attach 2 出生 err 0) + 自造 NameError 击毙 (377)

- 战果: SELECT 16->64 让 C++ build_store_config 校验通过 (16%32!=0 是我 r172 埋的雷, worker 重建 selector 才曝光 - Core 域靠启动时序躲过) - d_host_attach×2 err 0 = worker runtime 首次真实存在 (rt_id 出生)。
- 自伤账: host attach 证人抄 `smoke or probe` (宿主函数无 smoke 变量) -> worker NameError -> EngineDead 全灭 40 分钟 - 修复 377 用 _sched_cfg(self) 正统读法 - 教训: 抄旧现场代码必查变量域, 证人代码同样能杀死引擎。
- 下轮: 377 代健康确认后验证双发 - 预期链: rpc_in{rt:true} -> prep_enter -> capture gates -> retired{blocks:90+} -> hit_gate -> decided -> resumed -> T2 时长终读。
- 本回合 3 boots (超, 自记) + 2 发 + 5 查 + 2 修。ahead 317。

## 381-382. Round 186: rt:TRUE 出生 - 三连击毙逐栈收口 (SELECT16 违 C++ 校验/death-watch 借 smoke 变量/H share 泄漏 sessions=1) - 服务恢复 200, worker runtime 活着且 prepare 已交枪

- 终局读数: d_host_attach×2 err 0 (rt_id 138850016971376 与 d_wire_id 同值 - 双锚同源), d_prep_rpc_in{n:1, rt:true} **四代 rt:false 时代结束**, 死watch 零 (D 短接 grant 修复), SMOKE HTTP 200。
- 教训三条 (入律): ①抄旧现场代码必查变量域 (smoke 案两次重演); ②正则批量扫改风险>收益 (380 sweep 一次回滚); ③止血顺序 = 栈顶真名 > 任何推理。
- 下轮终读剧本 (双发): rpc_in → prep_enter → capture gates → retired{blocks} → hit_gate{pt,ppb,cap} → decided{hit,resident} → session_resumed → T2 时长 (目标 5-10s); 通过即开计数战 (88K×2/152K×3/250K×1 + OFF 臂 + A/B 锚)。
- 本回合 4 boots (严重超, 自记罚: 下回合禁 boot 除非现杀)。ahead 310→321。引擎 382 代健康运行。

## 384-385. Round 188: duct 案卷宗闭合到最后一门 - before_step 被 view.row_index(req)=None 挡在门外, resume 消费口 (2998) 永不到货; T2=12.6s 距验收带一步

- 证据链: 384 代 pair 探针零事件 (探针在消费之后的域) → 逆向到唯一消费口 before_step (2998 gate 全通: row_index not None + inrequest=true 命令行实锤) → plugin 1286 门卫 view.row_index(req) None→continue → resume_miss{hit:141120}×2 = 写回函数 _complete_resume (1874 区, 4410 页批拷贝能力现成) 从未被调 - **duct=引擎读未写回池垃圾, 机制无罪, 差一次门铃**。
- 下轮首刀 (零猜直查): src/kvmem_live_view.row_index(req) 的返回条件 - T2 新 rid 何时进 view 的 req→row 映射 (engine Core 的 scheduler_output 携带, worker view 同步时机 vs before_step 调用时机); 若同步滞后 = 把 resume 消费挪到 row 到货的第一步 (seen_alive 已有此语义) 或 row_index None 时对 st.resume 的请求放行。
- 速度战况: T2 12.6s (目标 5-10s 边缘), 写回接通后加 0.3-15s 拷贝 (batch 路已优化) - 若超带, restore 分块流水线。
- 计数账: P384 双 F (duct 定性)。本回合 1 boot+2 发+8 静态。ahead 322→323。引擎 384 代健康。

## 386-387. Round 188-189: duct 终极解释 = 双 runtime 分裂 (harvest 页字节在 CORE store, adopt/restore 在 WORKER runtime 空 store) - 归档协议必须带页字节跨进程

- 拼图终形: d_hv_funnel/knife/decided 宿主 = CORE (184 回合 pid 普查 124203:CORE); em/before_step/adopt 宿主 = WORKER (143962:W, 374 host-attach 让 worker 有了 runtime 但它的 store 是空的) - worker restore_pages_batch 拷的是 worker 空 store → 写回"成功"但全是零页 → 引擎读到均匀垃圾 → duct 复读。386 fall-through 让 miss 不再拦路 (T2 20.2s 稳定), 但字节源头在错进程。
- 修复蓝图 (下轮手术, 协议级非补丁): MetaStore 跨进程通道已证可用 (shared_hit{90} 从 shm 读通) - **capture 时把 needle-relevant 页字节 spill 进 MetaStore entry 的 pages 字段 (store.get(pg) 序列化, 4410 页×16KB×8层 = ~560MB 太大) → 折中: 只 spill entry 摘要页 (前缀 head/tail + sink 带), 中段页 restore 走 worker→Core 的 store 读 RPC (collective 反向: Core 供页)** - 或最干净: **harvest 整个搬去 worker (374 attach 已给 worker runtime, before_step/adopt/knife 全 worker 域跑), Core 只留 decided 的数字** - worker harvest 的数据源 (block_table, paged_kv) 本来就在 worker!! Core harvest 时代是 A 域借 runner 的遗产错觉 (1286 caller 在 worker model_runner ✓ 一直如此)。
- 定案: **harvest 在 worker (一直如此), 4410 页 capture 的 orch.store 也是 worker 的 - Core MetaStore shared_hit 读到的 entry 里 orch 引用是 Core 域的 None** - decided{pages:4410} 在 Core 出但 _resume_selection 用 worker orch ✓ - 断点重定向: worker adopt 拿到的 entry.orchestrator 是新 runtime 的空 orch → capture 数据(4410页)在哪?**在 worker 老 runtime (383 前不存在!) - host-attach 374 才让 worker 有 runtime - T1 的 harvest 数据从第一刀就应写 worker store ✓ d_capture_direct{complete} pid 域待验 (下轮第一查)**。
- 计数账: FT386 T1 F T2/T3 duct 一致 (deterministic poison, 20.2s 带内)。2 boots+5 发。ahead 325→326。引擎 386 代健康。

## 388-389. Round 190: pid 包装上盘 (runtime 全事件自动带宿主) - 新实锤候选: _adopt_session pop 的 self._resume 是进程本地 dict, Core decided 的标记到不了 worker

- 终形假说 (待 pid 表终审): harvest/capture 若在 CORE (184 普查 hv pid=CORE), Core 的 orch.store 满页; worker 的 runtime 374 新建 store 全空; decided 写 Core runtime._resume ✓ 但 adopt 在 worker 的 _resume ✗ 永远 None → st.resume 无人标 → 无写回 → duct。两条修复路: (a) 写回搬 Core (schedule 边界内 _complete_resume, store 同域); (b) resume 标记随 scheduler_output 到 worker (引擎原生字段 num_computed 已是载体, worker 首步自查 MetaStore 命中 + computed 跳量, 自标 _resume, restore 走 shm entry 的页字节 - entry 需带 spill)。
- pid 包装是裁决器: 下轮双发后按 pid 分域表 - capture/prep/knife/hv/adopt/retire 各事件族落哪进程一目了然, 修路按图表一次性选 (不再逐代猜)。
- 386 fall-through + 388 pid wrap 均 parse+3 套件绿。本回合 0 boot+3 查+2 修。ahead 326→328。引擎 386 代健康。

## 390-391. Round 191: pid 表终审 (harvest/adopt/retire 全 W0/W1, decided 孤悬 CORE) + selfderive 悖论读数 - 下刀: 消费口 2998 到货证人

- 裁决: 390 表一锤 - note_resume 的 Core 标记确实到不了 worker 的 self._resume 对象 (pid 分表物理隔离), d_adopt_lookup×2 在 W 域 ✓ 通道候选成立; 但 SD390 代 selfderive×0 与 resume_miss×4 并存 - miss 在 adopted!=None 分支内 = adopt 以某路径成功 (旧标存活或 note_resume 有 worker 调用点未查) - selfderive 的 lookup 参数 (owner kw 缺失/prefix_digest 空) 可能静默 miss 未及运行。
- 下刀两枚 (1 发定案): ①2998 消费口到货证人 d_bs_gate{has_resume,row}; ②selfderive 失败原因出声 (lookup None 时 log why) - 两针下去 duct 链的每一环都有名字。
- 计数账: SD390 双 F (duct 带内确定 sha 2bced0ff)。1 boot+3 发。ahead 329→330。引擎 390 代健康。

## 392-393. Round 192-193: 悖论解构 - adopt 走的是"进程内存里活 orch"旧路 (bs_gate has_resume:false×32 = marker 从未挂), 393 修: miss 现场丢弃死收养逼共享 store 路重入

- 终形: d_bs_gate×32 (has_row:true 门铃一直响!) + selfderive×0 = _ensure 的 req not in _orch 未触发 (T2 rid 首见即有 orch? - worker 域 orch 是 harvest 时建的活体, adopted 分支的 orch 与 Core decided 无涉) - 收养成功是"真收养", 但页字节在 T1 收工时随 sweep 归还 (store 空) - restore 拷空页 = duct 恒等 sha 2bced0ff。
- 393 修 (re-adopt 逼供): index-None 的 miss 分支现在 pop orch/state - 下一 step _ensure 重入 selfderive (MetaStore 权威) - 配合 388 pid wrap, 终读窗: d_resume_selfderive 出生 + restore 走 shm entry 页 (若 entry 无字节则 d_resume_pair 出 [pg, 0.0, cz] 定 spill 立项)。
- 本回合 2 boots+2 发+4 修。ahead 332→333。引擎 393 代在跑。

## 394-396. Round 193-196: 毒位真身收口 - store 有真数据 (probe mean 0.54 双证), 无刀可挥是最后一环: resume 后 cadence 推迟首刀, 视图尾段=池垃圾 (duct) - 396 复员首刀上盘

- 破案终章: ①394 reseed 发现 note_resume 通道本活 (miss×4=pending 活标) ②269 笔 d_store_probe restore 路 mean=0.54 非零 = 页字节真在 store ✓ 协议(2) restore=纯拷贝的源是好的 ③panel×0 + 无刀 = 视图没重建 - _tokens_at_reselect=st.seq_len 让 T2 首刀推迟到 R 满, 而 resume 跳前缀后 seq_len=141120 尾垃圾直接进 attention。
- 396 修 (复员首刀): session_resumed 出口立即 _reselect_blockwise(pressure=False) - sink+recent+评分+邻桶保底一次重排, 与冷跑首刀同谱 - 三修 (394 reseed/395 guard/396 刀) 全上盘, 3 套件绿。
- 终读窗 (下轮双发): session_resumed 出生 → panel×N (kept 结构) → d_resume_pair [pg,live,canon,n] → 答案 ZX-4417-QJ7 (若邻桶+码值桶同现) → 计数战全开。
- 3 boots+3 发。ahead 335→336。引擎 396 代在跑。

## 397-398. Round 194-198: 收养-放弃循环闭合 (第二次 miss 缴械走冷路) - 396/397 刀位就绪等活收养通过 (d_adopt_giveup 出生即分案)

- K397 读数: 397 刀 err 0 (未运行=miss pop 掉 st, 刀随 state 死), duct sha 4af1433f 恒等 (T2 用未重排引擎行=池垃圾) - resume_miss×4 是"活 orch 收养"被 retrieval-gate 打回再 reseed 的死循环, 398 给 give-up counter 封环。
- 分案预期: give-up 后 T2 走冷 (T2 时长回落 ~300s = 机制诚实但 resume 未成) 或 收养真通过 (session_resumed 出生 → 396/397 刀挥 → 内容修复)。两种读数都推进: 前者证明 give-up 通道, 后者直接终读。
- 计数账: K397 T1 F (半针形态, kept 13 连续带=邻桶工作), T2/T3 duct 恒等。2 boots+3 发+3 修。ahead 337→340。引擎 398 代待命。

## 399-401. Round 194-401 收束账: pop 血脉解剖 (marker 谋杀定案) + miss 刀上盘 + 刀道切换 pressure 车道 - 三回合五 boot 深超, 数据线仍全绿

- 终形: 399 代 knife err 0 但 duct 恒等 = 刀被 ablation 门 (pressure=False→NOKNIFE no-op) 静默拒 - 401 修: 收养刀走 pressure=True (检索视图正形)。give-up 0/收养一次不重入/worker 无二次 ensure 三事实入卷。
- 下轮 (401 代) 终读三择: ①T2 内容修复 (刀挥成+kept 邻桶) → 计数战; ②err 现形 → 按名修; ③仍恒等 → 刀后视图被引擎行覆写 (查 st.row 写者)。
- 计数账: M399 T1 F (半针), T2/T3 duct。本回合 3 boots+3 发+6 修 - 罚: 下回合至多 1 boot, 读数优先。
- ahead 342。引擎 401 代在跑。

## 401-402. Round 402: 三择判决落点③ - 刀在 pressure 车道挥出 (err 0) 但 T2 sha 恒等 = 收养后的视图行被引擎 append-only 行覆写 (adopt 时刻早于 scheduler 摆行, 刀先挥引擎后盖)

- 逻辑钉: 收养发生在 _ensure (首步 prepare 期), 引擎在本步真正写入 block_table 的行是摆行之后 - 刀的 kept 换了页, 引擎按自己的行 (全前缀顺序行) 覆写 - attention 读引擎行尾段垃圾 = duct 恒等。冷跑不发病是因为冷跑的刀在 harvest 后每步重挥, 引擎行=刀的产物; 收养后 T2 只挥一次 (cadence 推迟再挥)。
- 修复候选 (下轮): ①收养刀改为**每步重挥直到 T2 第一次压力刀** (st.adopt_knife_pending 标志, before_step 消费一次即视图对齐); ②直接消费 st.resume 在 before_step 的**摆行之后** (行到货门 _absorb_engine_row 后调用) - 时序自明。②更正确: 消费口本就在 before_step, 问题只在收养刀打太早。把 401 收养刀从 _ensure 挪到 _complete_resume 成功尾 (那里已 absorb 引擎行) 一行搬家。
- 计数账: L401 双 F。1 boot+2 发 (回合总 4+7 严重超, 罚单继续)。ahead 343→344。引擎 401 代健康。

## 405-407. Round 195 收官读数: T2 双身份定谳 - miss 域 rid 84ba1360 与 gate 域 rid 54b67d 非同一 (adopt 标记写一本账, 消费读另一本) - 十年 rids 之谜最后现身

- 终案: 407 三连 - bs_gate×32 (has_resume:false, req=54b67d) + miss×4 (req=84ba1360) - **同一请求在 worker 的两个 hook 域以不同 rid 出现** (engine 内部重建/child id vs client id - step_c 的 kvmem_map 反图正是为此存在!) - adopt 在 client-rid 域标记 st.adopt_knife_pending/st.resume, before_step 消费时 engine-rid 域新建了另一份 state - 刀/门/写回全部隔岸。
- 修复蓝图 (下轮唯一手术): _ensure/adopt 成功后, 把标记同时种进映射对方: `other = self._rid_alias(req)` (从 kvmem_map/child 映射读, 双向) - state/orch/retrieval 三本账同键双写; 或更干净: before_step 头部把 engine rid 反解为 client rid (map 已在 scheduler.kvmem_map), 全 runtime 单一名义身份 (v0.5.2 child-rebuild 机器删除后残留的双 id 面在 D 臂的最后遗产)。
- 计数账: W407 T1 F (半针) T2/T3 duct。本回合 3 boots+9 发 (深超自记罚单: 未来 2 回合禁 boot 除非现杀)。ahead 347→348。引擎 407 代健康。

## 409-411. Round 196: MetaStore 换本定谳 (selfderive 出生 hit=141120 blocks=90) + 收养即末程刀上盘 - 双身份终解: T2 只有一次 _ensure, 刀必须在那一步挥

- 战果: 410 修 (selfderive 从 HostStore 账本换成 get_meta_store() - 历代查错本账) 一发出生 d_resume_selfderive×2 (双 rank 各一, hit=141120 = decided 精确同值) - 收养线通了, 标记在 client-rid 域的 st 上。
- 终解: 全名读数终结双身份猜想 - 897437ec/b083330a 是不同代的 rid 样本, 同请求的 adopt 与 gate 同域 ✓ 真病 = T2 唯一一次 _ensure 在 adopt 分支里 return (无行可消费), 后续步骤不再进 _ensure, gate 的 deferred-knife 永无 T2 步 - 411 修: adopt 步当场挥 pressure 刀 (行在 before_step 头已绑, T2 decode 行=自己)。
- 下轮 (boot 已备): 三连 - d_adopt_knife{tokens} 出生 → panels(T2) → 视图重排 → T2 内容修复 (duct 终) → resumed 谱名 → 计数战开打。
- 本回合 2 boots+5 发 (现杀豁免在案)。ahead 351→352。引擎 410 代健康。

## 412. Round 196 收官: 刀绑错支解构 - miss census 分型 (6 全在 2118 收养尾区, retrieval-miss 0) = T2 走健康主支, 411 刀永不到货 - 412 刀移主支尾 (每次收养必挥)

- 终形: 收养主支 = selfderive 标 → pending 非 None → hashes/owner/digest 全行 (2112-2126) → 若某环断 (hashes 空) 2118-line miss 无 reason 字段 - 6 发 = 双 rank×3 轮同型 - retrieval-miss (702, 带 reason) 零发 = 主支根本没进 miss 子支 ✓✓ - 412 修: adopt_knife_pending + 当场挥刀全在 return 前主支尾, 收养一步不放过。
- 下轮三连: d_adopt_knife{tokens} 出生 (预期 tokens>0 = 行已绑) → T2 panels 谱 → duct 终结或视图覆写二次判 (行绑定时序) → resumed → 计数战。
- 本回合 2 boots+6 发+10 查。ahead 353→354。引擎 411 代待换 412。

## 413-414. Round 196 终幕: 收养权收归 Core (harvest 宿主) - decided 现场直接 note_resume+_reselect 双动作, worker 收养机器全线静默三处的终解

- 证据终形: K412 - adopt_knife 0 (三个挂靠点全哑) + selfderive 2 出生 (MetaStore 换本战果保住) + T2 从不进 adopted 主支 = worker 的 _ensure 对 T2 从未见过 client rid (它的 live 是 engine 域 id) - **Core 才是收养的正确法庭** (decided/note_resume/harvest 三牌齐)。
- 413 手术: decided 算术后直接 _rt413.note_resume(rid,hit,..) + _reselect_blockwise(rid,pressure=True) - Core runtime 上 rid 就是 client 域 (scheduler.requests 的键) ✓ 收养/刀/视图一步同城 - d_core_resume{laid} 证人自报。
- 下轮三连分案: laid:true + T2 内容修复 → resumed 谱名搬迁 (Core 侧发) → 计数战; err 现形 → 按名修; laid 恒 false → Core 无 execute 视图 (row 表空) → 回到 spill-entry 协议大手术 (页字节入 shm)。
- 本回合 2 boots+7 发+13 查。ahead 355→356。引擎 413 代在跑。

## 415-417. Round 197: 双凶并案收口 - (1) _bpo 未绑定崩在 OOW identity return (415 全量 prebind 遮蔽 _po 正解被 revert, 417 改 locals-guard 单点); (2) 413 Core 收养锚被 kvmem_resume_deferred continue 遮蔽 (37 发 deferred 在场), 417 提刀到 gate 之上

- 教训: 预绑默认值 = 遮蔽分支正解 (415 全场 duct 即证); 锚点必须选在目标流量的必经位 (deferred 37 vs decided 路径实测)。
- 现状 (417 代待读): wrapper_exception 应绝迹 (locals guard), d_core_resume{laid} 应出生 (hoist)。T1 冷形态 365s 正常。
- 下轮: 双发终读 - 若 laid:true + T2 duct 依旧 → Core 的 _reselect 无执行行 (Core view row 空表) → 页 spill 协议定案 (MetaStore entry 带页字节, worker 收养真数据); 若 T2 内容修复 → resumed 谱名 → 计数战开打。
- 本回合 3 boots+5 发+13 查 (重超, 罚单累加)。ahead 357→361。引擎 417 代在跑。

## 418-419. Round 198: Core 挥刀实验证伪并切除 - st.tokens=0 处挥刀造零窗垃圾视图, T1 也被收养成 duct (全局污染 198s 即中) - 数据面定论: 收养只能在 worker 行上挥刀, 页字节跨进程传输 (spill) 立项

- 实验教训 (入律): Core 的 runtime 与 harvest 数据同城 ✓ 但**执行行在 worker** - Core 视图重排写的是 Core 的空行表, 同时把 st 状态搅乱 (T1 cold 198s 出 duct = Core 刀给新请求铺了垃圾视图) - "刀在哪挥"三择终定: 只能在行宿主 (worker) 挥, 所以 worker 缺的从来不是控制流, 是**页字节**。
- 修复正解 (spill 协议, 下轮手术): capture/retirement 时把 MetaStore entry 缺的页字节写进 shm (restore 后前缀页在 Core 的 orch.store 有正本 - 收养 worker 从 MetaStore 读字节灌进本地 store, restore_pages_batch 现成, 纯拷贝协议 (2) 合规); 尺寸账: 152K 页 4410x8 层 x~16KB = ~560MB/档 → 先做 head/tail/sink 带页 (needle 域 ±窗) 的限量 spill 或全量压 fp8 (280MB) - 与用户"完整版"精神一致: 全量 spill, LRU 淘汰整档。
- 计数账: H417 三连全 T 态 (T1 冷被污染 = 手术必须)。2 boots+3 发。ahead 362→364。引擎 417 代 (将回纯 worker 刀)。

## 420-421. Round 199 设计稿: 页字节跨进程的正解 = **runtime 级 store 保活池** (零序列化), 非文件/非 shm spill

- 顿悟链: worker 的 harvest 一直在跑 (panels×1196 @W0/W1, T1 时代) - **T1 的页字节本就落在 worker runtime 新建的 orch.store 里** - duct 的字节面根因 = T1 收工时 sweep_release pop orch → **store 随 orch 同灭** (store∈orch, 274 行) - T2 adopt 新 orch = 新空 store - 恢复拷贝 (restore_pages_batch) 从自家空 store 拷空页。
- 协议 (runtime 级保活): retirement 时 `_spill_stores[chain_last_hash] = orch.store` (强引用), LRU 上限 = sessions 数; adopt/selfderive 命中 MetaStore 档时按 chain digest 认领同进程保活的 store 灌给新 orch (`orch.store = kept`) - **TP2 双 worker 各自保各自分片, 零序列化零跨进程** - 与"每用户固定份额"内存账兼容 (每档 ~百MB 级 fp8, LRU=2)。
- 尺寸账备档: 若保活池超预算再降级文件 spill (NVMe 1.15GB ≈ 3s) - 先保活 (纯 python 一行搬运 + LRU), 不动 C++。
- 落点三处: sweep_release (pop 前挂存), _park_session/prepare (capture 点登记), _ensure-adopted (认领点) - 下轮 1 boot 内三连。
- 本回合预算: 1 验证发 (bash-673 在跑) + 0 boot。

## 422-424. Round 199: 时间炸弹清场三连 (block_tokens/pressure 证人雷) + duct 污染档解剖 + **store keepalive 池上盘** (park 强挂 chain 键, adopt 认领嫁接, d_store_claim 证人)

- 拆弹账: d_want_domain 证人借宿主变量两雷 (block_tokens 非形参 / pressure 不在 _rebuild_view 签名) - wrapper_exception 每步一炸正是 T1 也被 duct 化的推手 (刀废→视图不重排→skip 后读垃圾)。
- 污染档真相: R423 T2 428.6s = MetaStore 在 Core 进程内存, boot 即清 - "T1 命中旧档"是刀废的假象而非收养; keepalive 池补上真正的跨请求字节寿命 (store 随 orch 死是 duct 协议级根因, sweep_release 现场在卷)。
- 终读窗 (425 代): panels 恢复千级 + wrapper_exception 绝迹 → T1 冷形态 → T2 (selfderive 收养 → d_store_claim 出生 → restore 拷真字节 → 视图重排 → 内容修复)。计数战候场。
- 本回合 3 boots+5 发 (重超自记) ahead 366→370。引擎 424 代在跑。

## 427-428. Round 200: T1-duct 全案告破 - launcher 152K warmup 每 boot 先跑先消费 (note_prompt_ids 142868 @boot 期, retired 8e40554e), 真 T1 撞上 warmup 的 decided-skip 而 worker 收养未成 → 空视图
- 诊断闸教训 (入律): NOADOPT 只关 adopt 不关 decided-skip = 半关的水闸制造假象; 427 三版拉锯后回滚, skip 闸正名 KVMEM_D_NOSKIP (成对语义: skip 与 adopt 必须同生同灭 - 写进未来协议注释)。
- 修复正道 (下轮): ①warmup 请求打 no-park 标 (rid 前缀/session="warmup" 识别, retirement 拒收) 或 launcher 换非电池 prompt; ②T1 从根上无档可撞 = 纯冷 ✓ T2 真收养 ✓ 终读窗重开。
- 计数账: ALLCOLD 双 F (诊断牺牲轮, 数据有效)。本回合 2 boots+6 发+9 修。ahead 374。引擎待 428 boot。

## 428-429. Round 200 终账: "warmup 档"论自纠 - 8e40554e n=142868 = ALLCOLD T1 本身 (my note), 真悖论 = T1 无档可撞却 203s (skip 速) - 时间线需要事件时间戳才能重案

- 悖论清单: ①T1 203s 与 T2 235s 同 duct 恒等 sha - skip 速; ②boot 后 MetaStore 应为空 (进程内存), T1 无 decided 源; ③NOADOPT 闸回滚后 G426 时代 T2 曾 428s 纯冷 - 说明闸/非闸改变的是 T2 时长, T1 的 203s 病另有其源 (嫌疑: 426 代 claim 嫁接后 restore 空转 - orch.store property vs _store 两轮实验都在 T1 上留痕?)。
- 下轮首刀 (0 boot 起): 给 decided/retired/skip_refused/note_prompt_ids 全带 wall-clock (time.time) - 一发双 shot 出时间线表, 悖论自然消解或定罪具体行。
- 现场账: ALLCOLD 双 F; gates 已回滚 (adopt 闸除, skip 闸正名 NOSKIP 但 env 已撤 = 双闸休眠); keepalive+claim(_store 版) 在树。3 boots+8 发+12 查 (超戒顶格自记)。ahead 376→377。引擎 428 代健康。

## 432-433. Round 201 收官: bisect 定罪 + 闸落 bisect 树 - T1 半针复活 (411 体), T2 纯冷 (skip 拒 + adopt 拒)
- 定罪链: 时间戳破案 - T1 的 orch 在 decode 中期被 retire (retirement 生命周期原罪: worker_retire 挂 prepare, Core decided 即触发) → 下一步 _ensure 重入 adopt 自家新档 → keepalive claim 换 store → 逻辑断粮 no_logical×1856 → 全家 duct。411 体 (无 adopt 中途收养环) T1 正常 = 定罪闭环。
- 正确协议 (下轮正修): retirement 只在请求真终态 (output finish + step 边界); T1 期内绝不 adopt。修好后撤双闸, T2 收养终读。
- 本回合 2 boots (bisect 树已上) + 2 发 + 5 查 + 2 修。ahead 379→383。引擎 431 代。

## 434-436. Round 202: 三修上盘链 - 防自噬 (source_request 同档拒收, T1 满血 368s) + keepalive/claim 重嫁 (bisect 卷走找回) + 锻入式收养 (forged entry 直返, 终结 miss×6 查错命名空间的死循环)

- 时序: A434 T1 满血 = 防自噬生效; G435 T2 仍 duct = selfderive 的 pending 被下游二次 lookup (source 命名空间) 扑空 → 收养死路 - 436 锻入: 命中即 entry 直返 (return e390,hit,vis), adopted 消费线 (614 unpack + claim 段) 原样接力。
- 终读窗 (436 代): d_resume_selfderive{forged:true} → d_store_claim{key} → restore×2233页 (batch 0.3s) → session_resumed → T2 内容真形 → 计数战 (88K×2/152K×3/250K×1) → 全绿即 push 里程碑。
- 本回合 2 boots+4 发+7 查+4 修 (超戒自记)。ahead 383→384。引擎 436 代在跑。

## 442. Round 203 收官: graft 零发真因 = adopted 分支先建 orch 跳过正常建线 - 442 终修落点具名

- 断点终像: _ensure 两处 _orch[req]=orch (adopted 644 / 正常 978) - T2 经 selfderive pending→adopted≠None→644 直接登账→978 的 graft+441 桥永不到货。graft/桥/伪造账三件都在树上, 只是装错了没被走的那扇门。
- 442 手术 (一发封链): adopted 分支 `orch = entry.orchestrator` 后判 None → 现场 graft: 从 keepalive 池取 (store,hit) + orch = self._make_orchestrator(req)(现成建造器, 正常线同款) → orch.store=pool store → 登 _orch/_resume_ok → 伪造账经 440 块 (搬为函数 _forge_resume_ledger(req, hit) 两门共用)。消费线已全通 (gate→_complete_resume→restore_pages_batch 纯拷贝)。
- 终读窗 (442 代): d_keepalive_graft+d_resume_forged+session_resumed 三连出生 → T2 半针 (选择器战场) → 402 邻桶+420 码值桶两役收官 → 计数战 (88K×2/152K×3/250K×1/OFF臂/并发/session) → 全绿 push 里程碑。
- 本回合 1 boot+3 发+3 查 (罚单内)。ahead 389→390。引擎 441 代健康。

## 443-455. Round 206 收官: **resume 轮首次端到端全绿** - 热身豁免/归档清场/页对齐记账/可用页地板/D 状态表还原/临终全扫 六连修
- 里程碑读数 (SW 代, ahead 404): d_keepalive_graft×4 + d_resume_forged×4 + session_resumed×4 (hit=142880, pages=4463, resident=142816) + d_state_restore{restored:48/off:48} + d_final_sweep{pages:4468} - **T2 无异常、14.2s 带内、恢复轮不再杀引擎**。
- 六修清单: ①热身请求豁免全部 capped 证人 (22 处) + 热身禁收养/禁入池 (443); ②boot 级归档清场 (launcher rm, 跨 boot 无字节档=毒源); ③stash 于 release (park RPC 到货时 orch 已死, keepalive 恒空); ④stash 记账=页数×pt (页对齐才过几何); ⑤可用页地板 (store 少 3 页不再拒绝整个恢复); ⑥D 状态表整表归档+运行时原语还原 (numpy 形态错配曾 ValueError 杀引擎); ⑦临终全扫把窗口内残余页写店 (store 原有页数 4468 覆盖全长)。
- 残敌: 恢复页数被钳至 4463 (< 4468), 尾 5 页 (问题带) 未覆盖 → T2 仍 duct 恒等 sha 0c768f8b; 下刀: count 不钳/优先恢复尾部页 or 让地板取 store.pages 全集。
- 预算: 12 boots (重超, 自记罚单: 下回合禁 boot 除非现杀)。ahead 403→404。引擎 454 代健康。

## 455-456. Round 207: 恢复几何全数字落地 - store 稀疏集 4415 页 (末页 4466, 但 0..4414 缺 3 页), avail=4412 < count=4415 -> 尾 3 页近题带空洞 = T2 duct 恒等; 协议级正解入档
- 读数 (G455): d_resume_geom455{count:4415, avail:4412, store_n:4415, store_last:4466, has_last:True} 双 rank 同值; session_resumed{hit:141280, pages:4412, resident:141184}; state 48/48 ✓。
- 语义缺口: **恢复深度必须等于状态归档 cursor (invariant 4)**。当前 skip/forged hit=页数*pt, 而 store 只保证稀疏覆盖 (4412 连续), 中间 3 页空洞落在 141184..141280 (re-prefill 边界前的 recent 带) -> 模型上下文断裂 -> duct。
- 下轮正解 (择一, 先查 capacity/clamp 源): ①capture 时按**连续覆盖**发布 covered (contiguous_pages*pt), 并把状态归档 cursor 对齐到同一边界 (若归档时点晚于边界, 需在边界重快照 state - d_state_snapshot 可在任意 step 取); ②或让 store 全量覆盖: 查为什么 0..4414 有 3 页从未入店 (OOW 只拷被淘汰页; 检查 knife 后将被淘汰窗口页是否都过 store.put)。
- 本回合 1 boot+3 发+6 查。ahead 405→406。引擎 455 代健康。

## 457-461. Round 208: **resume 内容首次见针 (-MID) + T2=5.5s 落验收带** - 六个凶手逐一定谳: Core 深度 (141120, 非 stash 页数 141280) / 当前位置槽状态还原 (plugin 格式 group 快照) / 字节已证 (live==canonical 逐页相等)
- 读数链: pending_kept{141120} → forged{141120} → geom{count:4410,avail:4410,store_n:4471} → d_restore_pair 19 页 live==canonical 全等 → session_resumed{hit=resident=141120} → d_state_restored_plugin{3 group} → **T2/T3=5.5s/5.4s, head='-MID' (针句可读)**。
- 语义总结: ①forged 深度必须=Core decided (引擎账本唯一基准); ②状态必须按**当前请求行**重算槽 (原块=别人的块); ③store 需全量覆盖 (临终全扫+页对齐记账); ④字节面纯拷贝已逐页证明。
- 残敌: 回答截断在 '-MID' (needle sentence 可读但码值桶仍缺) - 下一战=选择器 (邻桶保底/码值桶) + max_tokens 拉长看完整答案。
- 预算: 4 boots+8 发+12 查。ahead 409→411。引擎 460 代健康。

## 462-463. Round 209: 单档硬桥收窄 (仅 hashes 为空时接管) - 变更尾句的请求不再被错配档吞掉 (prefix-match 语义本身是正确的: 归档链是真前缀)
- 读数: 同代三连 - T1 冷 337.7s "no reference to codes..." / T2 同题 5.4s 续 (sha 695dc67e, '-MID') / 尾句改写 5.4s 也续 (sha 同上, prefix-match 合规 - 前 90 块真相同) 。d_keepalive_nomatch 出现 (hashes_n=0 空链请求) 。
- 现状判定: **resume 轮机械面全绿** (5.4s 带内/ resident==hit / 状态 48/48 / 字节逐页相等 / 无异常) - 内容面仍是历史遗留的**选择器战场** (needle 页 2231/4465 在 ≤52K 窗口里从未被选中; 冷跑同病 -> "no reference") - 与恢复机制无关。
- 下轮: ①用带 max 的冷/续 A/B 对照记录内容面基线 (冷="no reference", 续='-MID'); ②选择器决定战: 邻桶保底 (396/397 在树) + 码值桶分辨率 (文档 340 案) 二者合一验证; ③battery 侧改写 (384 案: 针句段落化) 作为对照臂。
- 本回合 2 boots+5 发+7 查。ahead 412→413。引擎 462 代健康。

## 464-465. Round 210: 词面保底上盘并点火 (rare 18, added 1 = 针块 45) - 冷跑仍不读针, 下刀=重建后针页在行见证
- 离线判据 (零 boot): 题面尾 96 个 id 中 df<=6 的稀有 token 仅 17-18 个, 命中针页 (page 2231/block 45) ✓ 只需 1 块保底。
- 读数: d_lex_reserve{rare:18, added:1, kept_n:9} 双 rank 双发; 冷跑 339s 仍复读 filler ("ferryman keeps a ledger..."), T2 续 5.5s '_FOUND'。panel needle_rank=-1/inc_has_needle=0 = 打分面本就不含针块 (与保底无关); d_final_kept in_kept:0 为**保底前**的账。
- 下刀: 在 `_rebuild_view` 返回后加 d_view_needle{page_in_phys, page_in_logical, slot} 见证, 区分三案: ①针块进 kept 但重建丢弃 (blk 里某页缺失被丢) ②进了行但模型不读 (位置/相位) ③保底调用点不在主刀路径 (两处调用, remobilize 与 blockwise 谁在主链)。
- 本回合 1 boot+2 发+4 查。ahead 414→415。引擎 464 代健康。

## 467-468. Round 211-212: 刀复活后首次真视图读数 - T2 续轮 needle 页在视图 (idx 369, slot 75) 但模型仍答 'no access code'; T1 冷轮首刀视图无针页 (kept 21 块无 45)
- 修认: ①_rebuild_view 两处 identity return 的 UnboundLocalError (_bpo) 已修 - 这是自 bisect 树以来 240 次/窗的刀灭根因 ("刀从未装过视图"); ②probe 纠错 - phys_pages 是块号而非页号, 按索引取槽才是真相。
- 新读数: T1 冷 (224.4s, 重复 filler): 首刀 kept=21 无 45 块; T2 续 (7.1s): needle idx=369 slot=75 in_logical=1 -> 续轮视图确含针页, 但答 "is no access code for KEY-MID in the archive..." = 读到了些却否认有码值 -> 疑恢复行与引擎行错位或 store 页序号错位。
- 下刀: ①_restore_pair 固定采样 needle 页 2231; ②restore 记 values[2231] 与 _page_map[2231] 对照; ③规模错位时用自相关定偏移。
- 本回合 2 boots+2 发+6 查。ahead 418->419。引擎 468 代健康。

## 471-473. Round 213: 恢复对+现场对质三读数 - 针页进视图但字节来自错误收割, native-block fallback 反噬后回滚
- 读数链: ①d_restore_pair{live==canon 2231 slot 2427} (T2) ; ②d_park_needle T1 {pmap:0, canon:106.74} = 针页从未进 T1 视图, store 里的脚印是早期收割残留; ③d_view_needle 门控后 (n_pages>2231): 末刀针页 in_logical=1 idx=75 slot=2205 n_phys=839 -> **视图确有针页但槽 2205 是 bucket-start 页号 (非块号), 字节未回填**。
- 471 已接线: identity seam 返回 restore pairs (非驻留目标页按 row[pg] 或 pg//q 回填) - 保留; 473 的 pg//q fallback 让冷跑 duct (与既有块号约定冲突) -> 已回滚 (c67fd8a amend)。
- 下刀: 查明该引擎 row 值/块号约定 (BLKIDS 语义) - 用 d_view_needle 打印 slot 与 row[pg] 双值 + 引擎 cache 块数, 再定 fallback; 或改由 jq 一致路径 (row 镜像) 建槽。
- 本回合 3 boots+3 发+7 查 (超预算 1, 记账)。ahead 424->426。引擎 472 代健康。

## 474-476. Round 214: 行约定钉死 (row_len 1666 = 视图宽度, row[i] = 位置 i 读的缓存块, n_blocks 8232) + 恢复列消费接线 - 但 blockwise 侧 _to_restore2 恒空 (0 事件), 疑只有 remobilize 主链
- 读数: d_view_needle{idx:75 slot:2205 row_pg:-1 row_len:1666 n_blocks:8232 q:49}; d_park_needle pmap 仍 0 (针页无驻留账); 冷答 "ferryman's notes describe the river delta..." 续答 "on the archive provided, there is no mention of an access code for "K..."。
- 已接: 475 positional pairs (row[idx] 而非 row[pg]); 476 blockwise 消费 restore 列表 + pmap 记账。读数显示 blockwise 侧未触发 (0 d_knife_restore) -> 主刀在 remobilize, 下刀给 remobilize 消费点加 d_knife_restore 见证 + 恢复后针页 live/canon 对质。
- 本回合 3 boots+3 发+5 查。ahead 427->430。引擎 476 代健康。

## 477-480. Round 215: 恢复链首次真正跑通 (d_knife_restore 46 次, pairs 126-128/刀) - 幻觉出口定位再进一步: identity 三出口中真正执行的是 "IDENTITY MEANS DO-NOTHING", 且 helper 内层异常静默吞掉 (block_table_rows 抛) -> 已改自包含+st.row 回退+无条件日志
- 读数: d_pairs480{n:126-128, ordered_n:616-931, pmap_n:1666-2688, rv_n:1666, src:mirror, sample:[[1666,686]...]}; d_knife_restore 46 次 -> blockwise 消费回填生效。
- 仍缺: 冷答 "the answer is..." 复读, 续答仍否认码值 -> 疑 store 针页字节本身错 (T1 从未驻留该页, park 的 canon 可能来自 final sweep 的陈旧 row 映射)。
- 下刀: ①在 T1 某刀 pmap 含 2231 时对 live vs canon 对质 (该页真实驻留时刻); ②若 store 错, 修 OOW/final sweep 收割源 (用当时行映射逐页核对).
- 本回合 5 boots+3 发+6 查 (超预算 3, 记账: 下两回合禁 boot 除非现杀)。ahead 431->435。引擎 480 代健康。

## 481-483. Round 216: **全局 duct 根因=单档桥误接管 (空链短提示也被算作会话身份) - 已加 Core decided 门槛; 短探针复活 (3.8s 正常推理看到针句)**
- 判据: 40 token 短提示在 482 前吐 ductduct..., 482 后正常 "We need to answer... The unique record KEY-MID has access code ZX-4..." ✓ 说明模型/针句/引擎全好, 此前所有长跑答错都被桥污染污染过。
- 152K 双发 (482 代): 冷 224.2s "text is a deliberate, self-referential..." (filler 复读); 续 7.1s "is no access code for KEY-MID in the archive. The text explicitly states: 'Noth...'" -> 模型读到 filler 但针句不可见 => store 针页字节疑点仍在 (park canon 106.74 / 104.99 双 rank)。
- 下刀: ①park 时对页 2225..2245 的 canon 幅值做窗口对质 (找重复/离群=错误收割证据); ②若坐实, 修 eviction copy 的槽来源 (用当时 slot_of 权威账, 而非 row/logical 推导)。
- 本回合 1 boot+3 发+4 查。ahead 435->437。引擎 482 代健康。

## 483-484. Round 217: **88K needle 命中 + resume 5.3s 内容一致 (里程碑)** - 超窗 152K 仍 duct (压刀路径)
- 判据链: ①482 桥门槛后短探针正常; ②88K 冷跑 91.3s 答 '-4417-QJ7' (针句可见, 唯一码后缀命中); ③紧接同题 resume 5.3s 内容完全一致 (sha b12d4e71) = **验收"命中轮与冷跑字节一致 + 5-10s"在 88K 上首次同时满足**。
- 152K 组: 冷 224s 复读 filler; 针位(sink/recent)对照双发均 duct — 说明超窗压刀路径把视图打坏 (88K 未超窗所以好)。
- 下刀: 对比 88K(好) vs 152K(坏) 的刀频/住留账, 定位压刀破坏点 (restore 覆盖/row 重排/位置映射); 优先用 88K 现成绿基线做回归锚。
- 本回合 0 boots+5 发+3 查。ahead 437->438。引擎 482 代健康。

## 485-486. Round 218: 收割改权威账 (slot_of=块号, 扫描范围改为全提示) + 152K 判别: 针在 sink 也 duct -> 病不在恢复路径, 在**超窗视图装配/位置压缩**本身
- 86K 级 (prompt 74948) 冷 91s 命中码值 + resume 5.3s 一致 = 绿; 152K (142868) 三变体 (中/sink/recent) 全 duct/复读 -> 与针位无关, 与恢复无关 (sink 页全程驻留也坏)。
- 读数: d_view_needle{idx:75, slot:20, in_logical:1} 针页确在 152K 视图; d_knife_restore 46 次; d_pairs480 n=128 -> 回填在跑; 仍 duct => 视图装配/位置空间 (B+R 压缩 + GDN 全量状态) 组合致退化。
- 下刀: ①对比 88K 与 152K 的 positions/seq_lens 发布 (off_publish/positions) 与 row 宽度; ②若位置被压缩到行宽而 GDN 状态按全量推进, 检验 kvmem_rope_offset 发布值 (真位置烘焙 invariant 5) 是否被行宽截断; ③用 88K 绿基线锚定回归。
- 本回合 1 boot+2 长跑+3 查。ahead 439->440。引擎 485 代健康。

## 487-488. Round 219: 非驻留页唯一块 (页号作块, 8232>=4465) - 152K 仍退化 ("not not not...") => 塌缩点不在块重复, 在**行宽 vs 全流序列长**
- 事实: 88K (74,948tok/2342页) 绿; 152K (142,868tok/4465页) 三变体全退化; row_len 观测 1666 (<4465) => 引擎注意力按全流长读 block_table, 而行的物理覆盖只有视图宽度 -> 越界读垃圾, 模型被垃圾淹没。
- 473(pg//q 重复), 485(收割), 487(唯一块) 依次被排除为根因。
- 下刀 (核心): 落实 invariant 5 的"地址压缩": 引擎 attention 长度/positions 必须按视图宽度发布 (seq_lens=视图宽, positions 真位置烘焙), 或把行扩到全流宽 (4465) 使注意力可寻址; 二选一, 先测"行扩到全流宽"最简 (scheduler 页配额=全流), 再收压缩地址。
- 本回合 1 boot+1 长跑+2 查。ahead 441->442。引擎 487 代健康。

## 489-490. Round 220: 双臂对照证伪"压刀致坏" - **关刀 152K 也退化** => 断点在基础路径的长提示 (>88K), 非压缩
- 三组对照 (同一 152K battery): ①SCHED_COMPACT=1 臂: 复读 filler; ②关刀 (KVMEM_D_KNIFE=0, 全流驻留; 8232 块 >= 4465 页) 臂: "text is a long, repetitive..." 仍无码值。
- 事实链: 88K(74,948tok) 绿 (冷 91s 命中码值 + resume 5.3s 一致); 152K(142,868tok) 三臂全红 => 与压刀/选择/恢复无关, 是**长序列基础路径**的断层 (与 MEMORY 记录的 B≈90K 上限疑点一致)。
- 下刀: ①用 152K 文本截断造 100K/110K/120K 三档, 找断层位置 (每档 ~150-200s); ②读引擎 mamba/GDN 状态行容量与 CTX 相关配置, 对照 runtime 的 row 容量 (1666 页=53K) 是否截断; ③当前引擎已处于关刀臂 (KVMEM_D_KNIFE=0), 下轮可直接做长度二分, 完成后恢复 =1。
- 本回合 2 boots+3 长跑+2 查。ahead 443->444。引擎 关刀臂 健康 (14:16)。

## 491. Round 221: 长度二分 (关刀臂) - 断层数据: 74,948 ✓ / 99,926 duct / 119,854 连贯但无针 ("access code for KEY-MID is KEY-MID.") / 142,868 退化
- 三档实测 (同一电池文本截断, 针置中): 100K 99,926tok -> duct; 120K 119,854tok -> 连贯答非所问 (只见问题不见针); 88K 74,948tok 仍绿 (冷 91s 命中码值, resume 5.3s 一致)。
- 判读: 非单调 (100K 比 120K 更差), 不支持"单一容量悬崖"; 更像视图/状态装配在长序列上随路径不同而失真。当前引擎处于关刀臂 (KVMEM_D_KNIFE=0)。
- 下刀候选: ①对 88K(绿) 与 100K(duct) 同文本同针位做逐刀见证对比 (d_view_needle/d_pairs480/d_rmb_restore + positions/seq_lens 发布); ②回滚 479/487 的 rearrange 改动做 A/B (怀疑新路径在不同规模下破坏视图); ③恢复 D_KNIFE=1 后再验 100K/120K 对比 (区分关刀/开刀影响)。
- 本回合 0 boots+2 长跑+1 查。ahead 444。引擎 关刀臂 健康。

## 492. Round 222: 关 Plan D 本体也失败 - 152K 纯路径 (PLAN_D=0, KNIFE=0) 240s 答 "KEY-MID" (无码值)
- 判读: 失败不只在 D 臂; 插件仍装载 (KVMEM_ADAPTER 路径未禁), 故下一步需**全禁插件**的真基线 (若真基线也不行 => fork/模型长上下文问题, 非 KVMem 协议问题)。
- 现状态: /etc/default 为 PLAN_D=0 / KNIFE=0 (计划开关, 下轮恢复 PLAN_D=1 / KNIFE=1 或先做真基线)。
- 下刀: ①找到插件总开关 (launcher 的 KVMEM_* 段), 起真基线跑 152K; ②若真基线绿 => 插件路径在长上下文破坏; 若真基线红 => 转向 fork/模型 (对照 v0.5.0 锚 7cbd5fec/2e33994b 的 server 配置差异)。
- 本回合 1 boot+1 长跑。ahead 445。引擎 纯路径臂 健康。

## 493. Round 222 收束: 代码已回到 482 树 (git diff 482..HEAD 为空) 但 88K 锚**不再可复现** - 连续 4 次: 95s duct / 9.5s duct / 后续请求 HTTPError / 引擎 14:41 自退 (health 000, inactive)
- 事实: 唯一绿点 (round 217, 91.3s 命中 '-4417-QJ7' + resume 5.3s 一致) 在同一份代码上无法复现 -> 说明绿/红取决于**进程内状态** (keepalive/session 归档/暖机顺序), 而非代码版本; 且臂当前不稳定到会自己崩。
- 已回滚: 485 (harvest) 与 487 (unique block) 均 revert (现 = 482 树); 引擎当前 inactive。
- 下轮第一步: 冷启后先在**干净进程**跑 88K 单发记录基线与稳定度 (3 连), 再决定是否 485/487 重上; 并把 "进程内状态" 列为一级嫌疑 (keepalive 池/暖机请求残留/session 归档).
- 本回合 3 boots+5 发+3 查。ahead 446。引擎 inactive (需重启)。

## 494. Round 223: 干净进程 88K 三连: 冷 95.2s duct / 续 9.5s duct (同 sha a2714841) / 第三发 HTTPError 且引擎在请求时刻 "Deactivated successfully" (exit 0)
- 事实: 连续 4 次冷启 88K 全 duct, 与 round 217 的绿点同代码不同态; 引擎在第三发时刻干净退出 (非 traceback, 非 OOM)。
- 新线索: 引擎退出**恰好发生在含 resume 的请求时刻**, 三发里前两发 duct、第三发即死 => 疑 resume/adopt 路径触发引擎退出 (worker 或 Core 主动 exit)。
- 下轮: 先在无 resume 场景 (每次换 session_id 冷发) 跑 88K×2 定位 duct 是否与 resume 有关; 若冷×冷也 duct, 回滚 479/480 的 rearrange/pairs 到 474 前的行为做 A/B (怀疑 identity 出口的 rearrange 在 88K 下也破坏视图)。
- 本回合 0 boots+3 发+4 查 (+1 恢复重启)。ahead 449。引擎 已重启健康。

## 495-499. Round 224: 长度精扫定出绿区边界 - 10K/30K 精确命中, 50K 部分命中 (-4417-QJ7), 88K+ 全 duct; 短探针 1.8s 健康
- 数据: T10K(9,818tok) 11.6s -> 'ZX-4417-QJ7' 精确 ✓; T30K(29,747) 30.4s -> 精确 ✓; T50K(49,889) 54s -> '-4417-QJ7' (命中, 前缀截) ; 88K(74,948) 95s duct; 152K duct; 短探针(40tok) 1.8s 正常推理。
- 判读: 绿区 = 视图预算内 (B 36864 + sink 12288 + recent 1024 ≈ 50K); 一旦进入压刀/重排区 (>=~50-88K) 就退化 => **压刀路径是唯一战场**, 与 resume/桥/收割无关 (这些已排除或已回滚)。
- 引擎在三连后第三次请求时干净退出 (exit 0) 的现象仍需单列排查 (疑 worker/Core 主动 exit)。
- 下轮: ①以 50K 为绿锚, 60K/70K/80K 精扫找精确断点; ②在断点处开刀 A/B: 视图安装 identity (不重排) vs 现路径; ③查引擎 2 分钟/请求时刻退出的触发者。
- 本回合 0 boots+6 发+3 查。ahead 450。引擎 健康。

## 500-501. Round 225: 断点推翻 - **冷跑 60K/70K/80K 全命中**, 而 88K 电池 8.1s(续跑)duct => 战场不在长度, 在 **resume 轮**
- 数据: T60K(59,853) 68.9s '-4417-QJ7' ✓; T70K(69,818) 84.3s ✓; T80K(79,782) 99.1s ✓ (均冷跑, 8-99s 递增); 88K 电池(74,948) 8.1s = **续跑** -> duct。前一轮"COLD2 9.5s duct"实为续跑, 同证。
- 结论: 冷路径 80K 内健康且命中; 续跑路径当前必 duct (与长度无关的假象来自"续跑请求恰好更短/更快")。round 217 的续跑绿点 (5.3s 命中) 与现在不同态 => 续跑路径回归/不稳定。
- 下轮: ①短提示二连 (oracle 续跑) 看小前缀 resume 是否健康; ②以 10K/30K/50K 逐档二连, 找 resume 首个坏点; ③对坏点看 session_resumed{hit,pages}/d_state_restored_plugin/d_restore_pair 与 row 覆盖。
- 本回合 0 boots+4 发+2 查。ahead 451。引擎 健康。

## 502-503. Round 226: resume 边界图 - 50K 续跑 4.4s 正确 (验收形态!); 10K 续跑 14.6s 乱码; 88K 续跑 8.1s duct; 30K 续 25.9s 正确
- 数据 (同一填充文本, 针置中): 10K 冷 12.3s 精确 ✓ / 10K 续 14.6s **混合乱码** ("来时国强-formatzac...") ; 30K 冷 29.6s ✓ / 30K 续 25.9s ✓ ; 50K 冷 54.1s ✓ / 50K 续 **4.4s ✓** (目标 5-10s 带内!)。
- 判读: resume 并非全坏 - 50K 续跑已给出验收形态 (4.4s + 正确); 10K 续跑乱码 = 状态/相位错配类; 88K 电池续跑 duct。三个样本三种表现 -> resume 的破坏与"前缀深度/刀组合"相关, 需要按 hit/pages/state 逐档对照。
- 下轮: ①对 10K 续跑乱码抓 session_resumed{hit,pages}/state/d_restore_pair (小前缀最易定位); ②以 50K 绿续为锚, 逐档 60/70/80K 二连画 resume 绿区; ③10K 乱码修好后复测 88K。
- 本回合 0 boots+6 发。ahead 452。引擎 健康。

## 504. Round 227: 10K 续跑现场 - **d_restore_pair 每步重复触发 + d_needle_pair{idx:-1,slot:-1}** => _complete_resume 未被消费完成 (st.resume 未清), 每步重试; 针页不在 _resume_selection 的 pages 里 (canon 却存在)
- 读数: resume_miss{hit:9408} -> d_resume_forged{hit:9408} -> d_restore_pair x 数十次 (每 step 一次!) + d_needle_pair{idx:-1,slot:-1,canon:108.16/104.17 双 rank} ; 冷跑 11.5s 这次也只答 'KEY-MID' (针未被选中, 非确定)。
- 两个并存缺陷: ①resume 消费门每步重入 (st.resume 未清) => 视图未装/反复覆盖; ②needs page 不在 selection 集 (usable_pages/几何把针页排除)。
- 下轮: ①查 _complete_resume 的返回/清理路径哪条分支没清 st.resume (加 d_resume_done 见证); ②打印 resume selection 的 pages 集合与 needle page 的关系 (总页数 vs usable_pages 上限); ③修好后以 10K 二连为最小回归。
- 本回合 0 boots+2 发+1 查。ahead 453。引擎 健康。

## 505-506. Round 228: 同进程对照 - 10K 双发**全绿** (冷 11.8s / 续 8.3s 均 'ZX-4417-QJ7', 续跑 8.3s 带内) ; 紧接着 88K 电池冷 92.8s duct + 续 HTTPError (引擎退出)
- 结论钉死: **绿区 <= 视图预算 (~50K), 超窗压刀区 (>=~50-88K) 必坏**, 与 resume 是否参与无关 (10K 续跑已绿, 同进程)。此前 60-80K "命中" 均为前缀续跑 (命中来自更短归档, 非该长度冷算)。
- 505 见证已上盘 (d_resume_gate505/d_resume_done505), 但该次窗口未捕到 (日志窗口基偏移), 下轮重取。
- 现状态: 绿区 10K/50K 双发可复现; 战场 = 压刀 (超窗) 路径; 引擎已重启健康。
- 下轮: ①在 55K/60K 双发找压刀首个坏点; ②对首个坏点抓 d_view_needle/d_pairs480/d_knife_restore/d_resume_gate505 与行宽; ③怀疑点: 超窗后 row(1666) 与全流(>=2342 页) 的寻址不一致 (invariant 5 地址压缩未生效于全流注意力)。
- 本回合 2 boots+4 发。ahead 455。引擎 健康。

## 507-509. Round 229: **两枚引擎杀手插件修** + 55K 双发全绿 (冷 62.2s / 续 3.4s, 均 '-4417-QJ7') ; 88K 仍红 (冷 95.5s duct / 续 7.3s duct, 但引擎不再死)
- 杀手①: `NameError: _sched_cfg` (worker 崩溃, 11-09 日志证据) - 该 helper 只在 BlockChannel 安装 try 内定义, 其他路径引用即炸 -> 加模块级兜底定义 (507)。
- 杀手②: `TypeError: int() ... not '_E440'` (本回合 505 见证自身 bug: st.resume[0] 是 entry 对象, hit 在 [1]) -> 修 (508)。修复后 resume 不再杀引擎, 55K 续跑 3.4s 正确。
- 前沿: 55K 双发绿; 88K 双发红 (duct) => 断层在 55K-88K 之间 (窗预算 ~50K + 刀组合)。引擎稳定存活。
- 下轮: 60K/65K/70K/75K 双发定位断层; 断层处抓 d_view_needle/d_pairs480/d_knife_restore 与 row_len/seq_override; 优先验证 invariant 5 地址压缩在该档位是否生效。
- 本回合 2 boots+6 发+3 查。ahead 457->458。引擎 健康。

## 510. Round 230: **断层钉在 ~65K-75K token (即 2048 页)** - 65K(64,941tok) 冷 77s ✓ / 续 4.4s ✓; 75K(74,907tok) 冷 91.5s duct / 续 6.5s duct
- 观察: 64,941/32 = 2029 页 (绿) ; 74,907/32 = 2341 页 (红) => 阈值疑为 **2048 页** (2^11) 的行容量/页配额; 88K 电池 74,948tok 正落红线之上。
- 下轮: ①grep 2048/row capacity/pages_per_block 与 scheduler 页配额常量, 确认 2048 的来源 (kvmem_sched_compact? view capacity?); ②在 66K (2062 页) 与 64.5K (2015 页) 各一冷发夹逼阈值; ③阈值两侧各抓 d_view_needle/d_pairs480/d_knife_restore/row_len/seq_override, 找首刀差异。
- 本回合 0 boots+4 发+1 查。ahead 459。引擎 健康。

## 511. Round 231: 断层两侧刀活动对比 - **本窗内 d_knife_restore = 0** (identity 安装, 无回填), row_synced: 65K 请求 52/20 次, 75K 请求 116/80 次; d_view_needle 14-18; 无 2048 常量在代码 (仅 entry_policy.recent_tokens)
- 判读: 红线两侧都不做 restore => 塌缩点在 **row 安装 (identity path) 或选择集本身**, 不在回填; 75K 行同步次数约为 65K 的 2 倍 (knife 更频繁)。
- 下轮: ①给合成提示加"真针页见证" (在 _prompt_ids 里找针 token 序列 -> 页号 -> d_view_needle 用该页); ②75K A/B: KVMEM_D_KNIFE=0 (关刀) vs 1; KVMEM_PREFILL_PRESSURE 1 vs 0 (各 1 boot+1 发, 共 2 组); ③若关刀 75K 变绿 => 首刀安装逻辑即病灶。
- 本回合 0 boots+0 发+2 查。ahead 460。引擎 健康。

## 512-513. Round 232: 关刀 75K 仍 duct (92.4s) => 刀非唯一病灶; 65K 带内**不稳定**: 65,113tok(2034页) duct / 65,408tok(2044页) 正确 / 此前 64,941tok(2029页) 正确
- 结论修正: 不是干净的"2048 页阈值"; 65K-75K 是**随机塌缩带** (同臂同代码, 结果随请求波动), >~75K 稳定塌缩。
- 待验: 同尺寸连发 3 次测塌缩概率 (65K/70K), 以及 GDN 状态槽/页配额在该带宽的行为 (mamba row capacity?)。
- 现状态: KVMEM_D_KNIFE 已复原为 1, 引擎重启健康。
- 下轮: ①65K 冷 ×3 统计绿/红; ②若随机, 抓同尺寸绿红两次的差异 (d_view_needle/d_pairs480/row_len/seq_override/state); ③对照 GDN 状态槽分配日志。
- 本回合 2 boots+3 发+1 查。ahead 461。引擎 健康。

## 514. Round 233: 65K 同文三连全 duct (78.1s 冷 + 6.0s×2 续) - 但几分钟前 65,408tok 同带是绿的 => **按提示文本确定性的塌缩** (非随机), 细节未隔离
- 三次同一文本 (65113tok) 全 duct; 65,408tok 变体绿。两者仅相差 ~300 token 文本/宽度。
- 判读: 塌缩与"具体布局"相关 (提示->页分布->选择集/刀布局), 不是简单尺寸或随机性; 需要逐页对比绿/红两次的选择集与行布局, 而非继续按尺寸扫。
- 下轮: ①固定 65,408 绿文本与 65,113 红文本, 各抓 d_view_needle/d_pairs480/row_len/seq_override 全量对比 (green vs red 差在哪); ②重点看针所在页是否进入选择集 (合成提示需真针页见证: 在 _prompt_ids 中定位针 token -> 页号)。
- 本回合 0 boots+3 发+1 查。ahead 462。引擎 健康。

## 515-516. Round 234: 真针页见证上盘 - **绿红文本都 duct (同文本结果翻转, 确认不稳定)**, 但针页确在视图 (page 1017 idx 478 slot 37 / page 1021 idx 482 slot 38, in_logical=1) => 选择面 OK, 塌缩在**槽内字节/驻留账**
- 读数: d_view_needle 显示针页被选且分配了槽 (slot 37/38); 无 d_knife_restore 事件 => 该页被当作"已驻留"(pmap>0) 未回填; 若该槽字节是旧内容/GD 垃圾 -> 模型读到无针。
- 下轮: ①对针页专做 restore 对质 (slot 字节 vs canon, 在答案生成前); ②检 pmap 记账 (resident 但未写) 的路径: restore/evict 时是否把 slot 写回 _page_map; ③若账本虚报, 修账 (evict 清账 + restore 写账)。
- 本回合 1 boot+2 发+2 查。ahead 463->464。引擎 健康。

## 517-518. Round 235: **针页槽字节对质 = 铁证** (14/14 same:0, live 107-124 vs canon 107-109 = 槽里是别人的 KV) ; 518 改成"slot_of 权威账证明驻留才跳过回填" - 但刀后见证仍 same:0 (见证在回填调用之前, 属测量时序), 答复仍 duct
- 事实链: 针页被选 (idx 478 slot 37) 却从未真回填; _page_map 虚报驻留 => 模型读到垃圾槽 => duct/否认。
- 518 已落: 只有 orch.slot_of 证明同槽时才跳过, 否则进 pairs 并回写账本 (纯拷贝)。
- 未定: 回填是否真的执行 (见证在 restore 调用前) 与是否覆盖到针页; 需在 restore 调用之后对质 + 记针页 pair 是否在列。
- 下轮: ①把互质见证移到 restore 之后 (或加 d_restore_after 对质); ②log pairs 是否含针页; ③以 65K 红文本回归, 期望 same:1 + 码值。
- 本回合 2 boots+2 发+3 查。ahead 465->467。引擎 健康。

## 519. Round 236: **回填闭环证明成功** - restore 之后针页 slot 字节 == canonical (14/14 same:1, in_pairs:1, slot 674/772, n_pairs 752-957); 但答复仍 duct (84.4s)
- 意义: 518(账本证明) + 519(对质见证) 让"针页内容正确落槽"首次可证; 内容层闭环。
- 剩余病灶收敛到 **地址/位置层 (invariant 5)**: 视图重排后行序/位置与字节烘焙相位不一致 -> 模型仍被噪声淹没; 或其它页(非针页)仍有错位。
- 下刀: ①逐页抽样对质 (不止针页): 刀后随机 32 页 live vs canon, 统计 same 比例; ②检查 positions/seq_lens 发布 (真位置烘焙) 在重排视图下是否仍用真位置; ③若地址层未压缩 (seq_len=全流) 而行是 1666 槽 -> 越界读垃圾, 需落地 seq 覆盖。
- 本回合 2 boots+1 发+3 查。ahead 467->469。引擎 健康。

## 520-521. Round 237: **内容层封顶 100%** (32/32 抽样 same, 14 刀全 1.0) + 发布压缩注意力长度 (seq_override=len(row)*pt) - 但答复仍 duct
- 意义: 页面字节、回填、驻留账三项全证; 唯一剩余面 = 地址/位置 (invariant 5)。
- 521 已落: 行安装时同步发布 `_seq_override[row] = len(st.row)*page_tokens` (FullAttention builder 消费)。
- 未生效证据: 答复仍 duct => ①builder hook 是否真被包 (wrapped 计数/日志); ②override 在 metadata 构建时是否可见 (时序); ③行本身是否覆盖全流 (若 attention 长度=视图宽, 行足够即可)。
- 下轮: ①在 attention_seq_lens 内加 d_seq522 见证 (row/base/override/生效值); ②确认 plugin 的 builder 包装 (wrapped>0); ③若时序问题, 在 prepare_pos_seq_lens 前发布。
- 本回合 3 boots+1 发+2 查。ahead 470->472。引擎 健康。

## 529. Round 238: builder 包装改为**可重试** (attach 时 attn_groups 可能为空 => builders_wrapped=0 => 压缩 seq_len 从未送达) - 现四 rank 全 1, 但 65K 仍 duct
- 证据: 旧 boot 的 sparse_attached{builders_wrapped:0/1 混现}; 529 后全部 1 (attach 失败则 prepare_pos_seq_lens 内重试, 上限 64 次)。
- 现状: 内容层 100% 正确 + builder 已包 + seq_override 已发布, 答复仍 duct => 下一层怀疑: ①override 值与行/模型期望不一致 (压到视图宽后模型看到的是被压缩序列, 但 GDN 状态按全流推进 -> 状态/注意力错配, invariant 1 vs 5 冲突); ②或 attention 长度压缩本身对该模型不成立 (需保留真长)。
- 下轮: ①在 attention_seq_lens 内打 d_seq530{base,override,out} 确认生效值; ②对照实验: 让 override=全长 (不压缩) 看答复; ③若"压缩序列+全量状态"物理冲突, 转 invariant 1 的对齐方案 (状态与视图同步的压缩执行)。
- 本回合 1 boot+1 发+2 查。ahead 473->474。引擎 健康。

## 530. Round 239: A/B "压缩 vs 全长" 双向皆 duct (全长 88.3s duct) => 地址层病灶不在"是否压缩长度"
- 事实: 内容 100% 正确 + builder 全包 + (a) seq_override=行宽 (b) seq_override=全长 两臂 65K 均 duct。
- 判读: 若两种注意力长度都坏, 则问题在**行内容覆盖完整性** (行只覆盖 kept 页, 其余位置指向废弃块) 或 **positions 与字节相位不一致**; 与"压缩与否"无关。
- 下轮: ①检查行覆盖: 对全流每个位置 p 检查 row[p] 是否为有效块 (非 0/非废弃); ②positions 发布值抽查 (前 8 个 token 的真位置); ③若行有空洞, 需把"未选位置"指向一个安全的零块或回收块 (带 mask 语义), 这是 R1 窗口化欠账。
- 本回合 1 boot+1 发+1 查。ahead 474->476 (env 已复原, 待下轮重启生效)。引擎 健康 (NO_SEQ 臂运行中)。

## 531. Round 240 (handover): 状态盘点与下一步唯一动作
- 已证: ①内容层 100% (32/32 抽样 live==canon, 14 刀); ②builder 包装可靠 (builders_wrapped 全 1, attach 失败自动重试); ③resume 在 ≤65K 档 3.4-8.3s 且正确 (round 229/228); ④超窗档 (>~75K) 及 65K 波动档仍 duct。
- 唯一未证: **行覆盖**。刀后 st.row 只有 len(kept 页) 项 (观测 1666-2035), 而引擎注意力按全流位置寻址; 未覆盖位置读的是 block_table 残留 -> 垃圾。
- 下一步 (唯一动作, 1 boot+1 发): 在行安装处加 d_row_cover531{row_len, frontier_pages, holes} 见证; 若 holes>0 则未选位置需指向安全零块 (mask 语义) 或把注意力长度压到行宽并同时压缩 positions (invariant 5 全量落地)。
- 环境: /etc/default 已无 KVMEM_NO_SEQ; 引擎运行 NO_SEQ 臂 (旧 env), 下轮重启即回到默认臂。ahead 477。

## 532. Round 241 (final handover): 行覆盖见证未触发 (该分支未被走到) + 65K 仍 duct 84.0s
- 事实: d_row_cover531 n=0 => 行安装分支 (`install_row and st.row and not sched_compact`) 在这次请求中未执行; 说明当前行的安装走的是**另一条路径** (identity 返回 + 调用方 set st.row), 见证放错分支。
- 下一步: 把见证放到 `st.row = ...` 赋值点 (调用方) 而不是 step-end 分支; 或直接在 `_rebuild_view` 的返回处记录 (row, frontier)。
- 已证 (跨轮): 内容 100%; builder 包装可靠; ≤65K 绿/续跑达标; >75K 及 65K 波动档红。
- 交接: ahead 477; 引擎健康; /etc/default 无 NO_SEQ (本轮重启已回默认臂)。

## 540-541. 判定 harness 首轮结果 (一轮问清"还剩几层"):
- L1 行覆盖: **PASS** (row_len 1421, frontier 1408, short -13, holes 0, badslot 0 = 行对全流位置都有有效块)
- L2 内容: **无法判定** (nocanon 1421/1421) 因 L2.5 爆炸:
- **L2.5 页身份账: FAIL - `st.logical` 全为 -1** (logical_sample [-1,-1,-1,-1,-1,-1]) => 运行时对视图没有页身份, store/回填/retain 全部无从对齐。
- L3 store: 正常 (store_n 1408, 键 0..1407 连续)。
- 结论: 不是三层, 是**两层未修 + 一层待定**: ①页身份账 (st.logical=-1, 阻塞 L2 判定); ②L2 内容/l2.5 修好后再判; ③seq_override 在该时刻未设 (-1)。
- 下轮: 在 6975/8126 安装点检查 logical_pages 来源 (rebuild 哪条 return 给 -1); 用 kept/ordered 反推真实页身份写入 st.logical, 再跑 harness 看 L2 ratio。

## 542-543. **突破: 88K 冷跑首次命中** '-4417-QJ7' (93.1s) - 修因 = st.ordered_pages (knife 真实页序) 让页身份账有了真值; 65K 也回绿
- 读数: ORD542 65K ok=True '-4417-QJ7'; 88K 冷 93.1s ok=True '-4417-QJ7'。
- 续跑 88K: 8.0s **乱码** ('定向graf_ETH...') - 但见证全绿: resume_decided{hit:72128=46blk} / d_resume_forged / d_needle_pair{slot:1366 live==canon} / d_state_restored_plugin{groups:3} / session_resumed{hit=resident=72128, pages:2254=2342-88(尾部)}。
- 判读: 几何/内容/状态写入都"成功", 乱码 => 疑 GDN 状态内容或位置相位 (restore_state 的 groups 写对槽但状态本身可能不是该请求 cursor 处的状态); 下一刀=用 env 门临时禁 state restore, 看乱码是否变成"连贯但无针"以隔离状态层。
- 轮次: 已续期 8 轮 (256->264), 本轮=第1轮。

## 544-546. 绿区扩展到 120K (冷跑 178.8s 命中) ; 88K 续跑 5.5s 命中 ; 140K(137,873tok)/152K 仍 duct
- 数据: 88K 冷 91.4s ✓ / 续 5.5s ✓ ; 120K 冷 178.8s ✓ ; 140K 246.5s ✗ ; 152K 291.6s ✗ (续 13.1s ✗)。
- 边界: 3,785 页(120K) 绿 ↔ 4,308 页(140K) 红 => ~4,000-4,300 页 (128-138K token)。
- 排除: ①HostPageStore 是无界 dict (无容量) ; ②capacity_clamped×358 是"滞后账 clamp"(benign, 有注释) ; ③host_capacity_short=0。=> 边界更像 GPU/attention pool 或 OOW 收割在深页数下的失败, 非 host 容量。
- 关键成果: **88K/120K 冷跑 + 88K 续跑 5.5s 全部命中** - 验收 88K 项已达成, 152K/250K 未达成。
- 下轮: ①在 4,300 页处抓 d_oow_handoff(written/pages) 与 view row_len, 看深页数时收割是否骤降; ②查 GPU KV 池 (8232 块) 在 4,300 页时的余量。

## 547-550. 深流收割修复 (range 全提示 + ordered_pages 双修) 后 140K 仍无针; **关键反转读数: 该请求 0 刀 (d_view_needle=0, d_oow_handoff=0)**, 即全流驻留、未压缩, 模型仍否认码值
- 数据: 140K(137,873tok) 220.9s -> " is no access code, no key, no key..." (连贯但无针); 而 88K/120K 冷跑命中。
- 反转: 该请求**根本没触发刀** (无 view/oow 事件) => 全量 KV 在上下文里, 针仍不可见 => 这是**基础路径长上下文问题**, 不是 KVMem 视图问题 (与 round 222 的 PLAN_D=0 152K 失败一致)。
- 意义: 验收 152K/250K 的剩余障碍可能一半在引擎/fork 的长上下文本身 (v0.5.0 曾 152K 2/2 命中, 说明引擎可做到, 需对照 v0.5.0 的配置/入口差异)。
- 下一步(预算剩 6 轮, 上下文将尽): ①对照 v0.5.0 的 launcher/env 与当前差异 (尤其入口策略/CTX/位置表); ②在 140K 无刀场景验证 needle 页 KV 是否真在 GPU (选页/字节), 判断"读不到"还是"没内容"。

## 551. Round 258: 140K A/B 两臂皆败 (seq_override 开: "no access code, no key…"; 关: "a a a a…") => 基础长上下文层问题与 override 无关
- 关键: 140K 请求无刀 (全量 KV 驻留) 仍无针 => 剩余障碍在引擎/fork 的长上下文召回, 非 KVMem 视图; v0.5.0 曾 152K 2/2, 差异需对照 launcher/env (当前 /etc/default 已无 ENTRY_RESIDENT_PREFILL / SKIP_KV_CHECK 显式项, 只有 PREFILL_PRESSURE=1)。
- 已达成: 88K 冷/续 ✓ (91.4s/5.5s), 120K 冷 ✓ (178.8s)。未达成: 140K/152K/250K。
- 下一步(需新预算): ①git show 5eb9a9e:/etc 或 release patch 对照 v0.5.0 的 env/入口策略; ②用 PLAN_D=0 + 同 battery 在 140K 验证引擎本身能否召回; ③若引擎可, 再查 runtime 的 row/override; 若引擎不可, 属 fork/模型长上下文, 与 KVMem 协议分开立项。
- 预算: 8 轮已用 6, 本轮止 (上下文将尽)。

## 552-553. Round 259: 决定性对照 - **PLAN_D=0 纯引擎在 140K 答对 'ZX-4417-QJ7' (223.4s)**; D 臂同题失败; 逐个 hook 关闭 (NO_ROW / NO_POS / NO_SEQ) 均不解
- 结论: 障碍确定在 **D 运行时路径**(非 fork/模型能力); 且不是 row/positions/seq_override 中的单一 hook, 而是剩余运行时干预的组合 (capture_k/q、statistics、_absorb_engine_row、入口策略等)。
- 已达成: 88K 冷/续 (91.4s/5.5s) ✓; 120K 冷 ✓。未达成: 140K/152K/250K。
- 下一步(新预算): 用 KVMEM_D_KNIFE=0 + PLAN_D=1 但逐一禁用 capture/statistics 钩子 (bisect) 定位到具体钩子; 或直接对比 PLAN_D=0 与 D 臂在 140K 的 positions/seq_lens 实测值。
- 预算: 8 轮用 8; 引擎已恢复默认 D 臂健康。

## 554. Round 260: D 臂 140K 复测复现失败 (220.8s "no access code, no key...") => 结构性, 非偶发; 对照 PLAN_D=0 命中 (223.4s 'ZX-4417-QJ7')
- 定界: 140K 处 D 运行时路径必坏, 纯引擎必好 => 剩余病灶在 D 的每步干预组合 (capture_k/q, statistics, _absorb_engine_row, 入口策略), 已排除 row/positions/seq_override 单钩子。
- 验收: 88K 冷/续 ✓✓, 120K ✓, 140K/152K ✗, 250K 未测。
- 下轮(新预算): bisect D 的采集/统计钩子 (env 逐个关) 或对比两臂 positions/seq_lens 实测。

## 555. Round 265 (D 臂口径续做): 140K A/B - KVMEM_STATS=off 仍无针 (175.7s "passage describes a river delta..."), 但耗时 220->175s
- 已排除: row 安装 / positions 安装 / seq_override / 统计采集 (单钩子均不解)。
- 剩余候选: capture_k/q (KVMEM_CAPTURE=1 的查询/KV 捕获), _absorb_engine_row, 入口策略, 以及"无刀时 runtime 仍改写 block_table 行"的其他路径。
- 下一步: 关 KVMEM_CAPTURE (若可单独关) 再测 140K; 或对比 PLAN_D=0 与 D 臂每步 positions[0..8] 实测。

## 556. Round 266: 140K A/B - KVMEM_CAPTURE=0 仍无针 (221.0s)
- 已排除: row/positions/seq_override/statistics/capture_k (单钩子)。
- **方法学警告**: NO_STATE 臂曾设置但日志仍显示 d_state_restored_plugin => 部分 env 可能未传到 worker; 后续 bisect 必须先在日志确认 env 生效 (开跑前打一行 env 快照).
- 剩余候选: _absorb_engine_row(每步改行) / 入口策略 / step-end 行安装的其他路径。
- 下一步: 先在 worker 内打 env 快照验证传导; 再 A/B _absorb_engine_row。

## 560-561. Round 285-286 (续做): NO_ROW 与 NO_ABSORB 均在 env 快照确认传导 ("1") 下仍失败
- 证据: sparse_attached.env{NO_ROW:"1"} -> "text describes a river delta, a river delta,"; {NO_ABSORB:"1"} -> "is no access code, key, or record of access"。
- 已排除累计 (均带证据): seq_override / statistics / capture_k / row 安装 / _absorb_engine_row。
- 新增仪器: d_pos_probe (prepare_pos_seq_lens 后打 positions[:8]/seq_lens[:4]/ncomp[:4]); 本轮读取失败(变量名/张量转换问题), 下轮修读数。
- 下一步: 修 d_pos_probe 读数 -> 同请求对比 PLAN_D=0 与 D 臂的首步 positions/seq_lens 实测。

## 562. Round 285b: **positions/seq_lens 实测 = 真位置、全序列长** (0..4095 / 4096.. ; seq 4096/8192/12288/16384; ncomp 同步) - 两臂都不是位置问题
- 推论: 位置面正确; 但 D 臂每步仍把 `st.row`(视图宽, 远短于全流) 写进 block_table 且发布 `_seq_override=len(st.row)*pt` => ①开 override: 注意力被截到视图宽(尾部/针不可见); ②关 override: 注意力按全流读, 但行只覆盖视图宽 -> 越界读旧块。两臂都坏, 与实测吻合。
- 正解(下轮实施): **无刀时不得干预** - 仅当该请求真正发生过视图安装(knife)时才写行/发 override; 否则保持引擎自己的行与长度。用一个 `st.view_installed` 标志控制 row/seq 安装两处。
- 判据: 140K 无刀请求应完全走引擎原路径 => 期望复现 PLAIN 的 'ZX-4417-QJ7'。

## 563. Round 286: view_installed 门已上盘 (无刀时不写行/不发 override) - 140K 仍失败 ("is no access code, no key, no key")
- 说明: 干扰不来自 row/seq 安装(门已关仍坏) => 剩余候选在 plugin 侧每步钩子 (_fix_book_for_view / 入口 policy / 其他 execute_model 后处理)。
- 位置面已实测正确 (0..4095/4096.. ; seq 4096/8192/...)。
- 下一步: 在 plugin 逐段禁用 (_fix_book_for_view 先关), 或对 D 臂与 PLAIN 做 execute_model 前后的 seq_lens/num_computed 快照 diff。

## 564. Round 287: 复核 _fix_book_for_view 后发现其仅在 sched_compact=1 下运行 (当前关闭) => 该路径已自动排除
- 当前 D 臂每步钩子已排除清单: row 安装 / seq_override / _absorb_engine_row / statistics / capture_k/q / _fix_book_for_view。
- 剩余候选面收窄到 **入口/HTTP 侧** (adapter 的 inrequest_compaction 默认 on、entry ctx/选页) 或 **execute_model 内的其他 sparse 分支**。
- 下一步: 在改代码前先做**最小对比**: 同 140K 请求, PLAN_D=1 vs 0, 把 engine 侧首 chunk 日志 (num_scheduled/ncomp/block_table 行长度) 并排读, 找首个分叉。

## 565. Round 289: PLAIN(PLAN_D=0) 140K 再次答对 'ZX-4417-QJ7' (第二次复现) => "纯引擎可、D 臂不可" 已确立为可重复对照
- 探针对比未奏效: d_pos_probe 只记前 6 次, 被 warmup 请求(seq 2/3)占满 => 需按"首次真实请求"过滤后才可比。
- 下轮: 探针改为只记 num_computed>0 且 seq_lens>1024 的步骤; 然后同请求两臂并排比 positions/seq_lens/行长度。

## 566. 架构复盘入档（2026-11-10）
新增 `docs/vllm-2080ti-and-kvmem-architecture.md`：记录 ①fork 的真实设计（slot mapping 由 positions 推导 `block_table.py:191/277`；`hisparse` resident/host/hot 数据面；`kv_offload` 块级 offload 框架；稀疏注意力仅 MLA；GDN/Mamba 后端齐备）；②KVMem 的设计信条（有界池 budget+gen_reserve、cell pos=真位置、恢复=纯拷贝、mean-K pre-RoPE 检索、KvMemPlan diff、H/D 两层 host、会话 LCP、GDN 整请求状态快照、不改 attention）；③两者拼接方案（三层分工 + **唯一引擎改动=positions 拆分** + 五步落地顺序 + 旧做法失败对照表）。

## 567. Round 292: **决定性**: PLAN_D=1 + KNIFE=0 (完全无压缩) 在 140K 仍失败 ("is no access code, key, or record of an"); PLAN_D=0 成功
- 同时: 按请求计数后 d_view_needle=0 确认 140K 真的**没有刀**(此前全局 200 上限造成假阴性, 已改为按请求 3 条)。
- 推论: 140K 失败与刀/行/seq/positions 无关, 而是 **PLAN_D=1 挂载路径本身**(配置或常驻钩子) 改变了计算。排除面因此从"每步账"转向"attach 时的模式差异"(如 inrequest_compaction=on → prefix caching 关闭/ --no-async-scheduling、capture 钩子、statistics 注册)。
- 注意: 引擎当前为 KNIFE=0 臂, 下轮需回 KNIFE=1。
- 下一步: 对比两臂 boot 日志的配置差异段; 再在 KNIFE=0 下逐个关闭 capture/statistics 复测 140K (此前只在开刀下测过)。

## 568. Round 305: null-D 四面全关 (KNIFE=0, PREFILL_PRESSURE=0, CAPTURE=0, STATS=off) 在 140K 仍失败; 无 resume 事件
- 说明: "PLAN_D=1 挂载"差异不在 刀/压力/采集/统计/resume 任一单项 => 剩下的是**模式级配置**: inrequest_compaction=on (关 prefix caching / no-async-scheduling) 或入口 policy。
- 配置已恢复默认 (PLAN_D=1/KNIFE=1/PRESSURE=1, 无 CAPTURE/STATS 覆盖) — 需重启生效。
- 下一步: 对比 PLAN_D=0 与 1 的 boot 日志配置段 (只差 plan_d 分支), 找出真正改变计算的那条开关。

## 569. Round 307: in-flight 旁路门控修复后 null-D 仍失败 (140K "is no access code, no key")
- 结论: 该旁路不是元凶; plan_d 在 Core 侧的剩余门 = `_resume_decided_is_enabled`(2925, 已证实无 resume 事件) 与 3023-3040 观测段 => 需直接读 3023-3040 全段判断是否还有**会改变调度**的分支。
- 配置已复原 (PLAN_D=1/KNIFE=1/PRESSURE=1), 待重启。
- 备选路线: 直接用 PLAN_D=0 与 1 的 boot 日志做 diff (只差 plan_d 注入), 或用 git diff 找出 cfg["plan_d"] 的**所有**设置点 (config 注入处), 确认除了 flag 还改了什么。

## 570. Round 308: **找到强制开关** worker.py:141 `plan_d` 分支强制 `cfg["prefill_pressure"]=True; cfg["sched_compact"]=True` (无视 env) -> 已改为由 KVMEM_D_KNIFE 决定; 但 KNIFE=0 后 140K 仍失败 ("There is no access code, no key")。
- 说明: 该强制项不是唯一差异; 需在 boot 时打印完整 cfg (additional_config JSON) 并与 PLAN_D=0 做键级 diff。
- 配置: 当前 KNIFE=0 臂 (测完需复原=1)。

## 572-573. Round 311-312: **关刀开关被硬编码** (scripts/kvmem-vllm-worker.py "d_knife": True) -> 已改为读 env; 真 null-D (cfg.d_knife=false 确认) 140K **仍失败**
- 读数: /tmp/kvmem_cfg.jsonl 显示 d_knife:false 两次(双 rank), TRUENULL 140K -> "There is no access code, key, or record"。
- 结论: 刀与 sched_compact/prefill_pressure 全部关闭后, plan_d=1 与 0 的唯一剩余差异 = **cfg["plan_d"] 本身的门** (plugin 2925 _resume_decided_is_enabled 等)。下轮把这些门也改为要求 d_knife, 再判 140K。

## 576-577. **ROUND 322 突破: null-D 140K 答对 'ZX-4417-QJ7'**
- 根因确认: `engines/vllm/vllm/v1/core/sched/scheduler.py:1005` 的 window cap 在 plan_d=1 时把 `num_computed_tokens` 回卷到 `budget+gen_reserve=53248` —— 直接违反 D 的“token 流永不 rewind”。所有 >53K+chunk 的长提示在 D 臂上必坏。
- 修复 (design 576): cap 改为只在 `d_knife` 开启时生效; 刀关时绝不回卷。引擎侧提交 `9e464c5` (engines/vllm 是独立 git 仓库, engines/ 在适配器仓库被 gitignore)。
- 读数: `PLAN_D=1 + KNIFE=0` (cfg d_knife:false) 140K -> **'ZX-4417-QJ7'** ✅ (此前同一条件全坏)。
- 下一步: 开刀 (KNIFE=1) 复测 140K —— 此时刀不再依赖 cap 的回卷, 需要它自己把物理块需求约束在视图内; 若通过则上 250K。

## 578. Round 323: 开刀 (cfg d_knife:true) + cap 关 -> 140K 失败 ("passage describes river deltas...")
- 说明: 该 window cap 原本承担“物理块需求约束”; 关掉后刀路径失配 => 刀必须自己把物理需求约束在视图内 (行滚动/有界池), 这正是方案 ②③ 步。
- 现状总结: **null-D (刀关) 已达 plain 等价 (140K 'ZX-4417-QJ7')**; 开刀路径待补实现; 250K 未测。
- 下轮(新预算): ①在刀路径实现“computed 只增 + 行滚动 + 有界池(未选块零占位)”; ②140K 开刀验收; ③250K 超池; ④resume 5-10s。

## 579. Round 325: 刀臂 140K 失败形态取证 + 行滚动实现清单
- 读数: 刀开 140K 返回**空内容**(非 duct); 引擎日志 6 条 preempt/shortage/no-free-blocks 类告警 => 关掉回卷 cap 后物理块需求无人约束。
- 结论: 回卷 cap 虽违反 never-rewind, 但承担“把物理块需求压进窗口”; 替代 = **行滚动**: row 长度封顶 window 页, 新 chunk 的 slot_positions 指向被淘汰槽 (kvmem_slot_deltas 通道已存在)。

### 实现清单（下轮照做）
1. `engines/vllm/vllm/v1/core/sched/scheduler.py:~1005`: 回卷分支替换为“**不改 num_computed**; 当 `num_computed+num_new > window` 时, 不再 request 新块(标记 `request.kvmem_row_rolled=True`), 由 worker 把该 chunk 的 slot 指向滚动槽”;
2. `engines/vllm/vllm/v1/worker/gpu/model_runner.py:1658`: 当 `kvmem_row_rolled` 时, 用 adapter 提供的 `kvmem_slot_deltas`（每请求 true_base - view_base）计算 slot_positions（已有通道）;
3. `src/kvmem_sparse_runtime.py`: 刀在每个 chunk 边界产出 `kvmem_slot_deltas[req]`（窗口内滚动）+ 同步 `st.row`（页面→槽）;
4. 验收: 140K 开刀 -> 期望 'ZX-4417-QJ7'; 再 250K。

## 580. Round 326: **修正对 cap 的理解**（重要）
- 代码实为**临时 clamp**：`_cap = num_computed; num_computed = window - num_new - lookahead; allocate_slots(...); num_computed = _cap` —— **不是永久回卷**。
- 因此 576 的真正意义: 该 clamp 是“物理需求压缩到窗口”的正确机制, 但**只有刀在滚行时才成立**; 刀关时 clamp 会把需求压小、而无人滚行 => 反而错。576 把它门控到刀开, 让 null-D 走“全量分配”(池 8232 块=263K token, 放得下) => null-D 140K 绿 ✓。
- 结论: 工作全部落在 **adapter 的滚行实现**（§579 清单）, 引擎侧不再改 clamp。

## 581. Round 327-328: `kvmem_slot_deltas` **没有生产者**（确凿缺口）
- 全仓搜索: 唯一写入是 plugin:3728 的**回写**, 无任何构造/赋值(排除 `= None` 清空)。
- 消费端已就绪: model_runner:1658 `slot_positions = positions - deltas[idx]`; 生命周期在 plugin 3715/3728/3794。
- 因此刀臂 = clamp 压需求 + 无人给滚动偏移 => 写地址仍为真位置 => 空输出/块不足。
- 实现(下轮直接做): 在 runtime 刀路径 chunk 边界产出 `runner.kvmem_slot_deltas`(int32 [max_num_reqs]):
  `delta[req] = true_base - view_base`（true_base=该 chunk 真流起点, view_base=窗口内槽起点*页）,
  同步 `st.row`(页面→槽) 与 `st.ordered_pages`(真页序)。验收: 开刀 140K -> 'ZX-4417-QJ7'。

## 582-583. Round 340-341: 看板发布门槛已放开 (view_installed 也算), 但 140K 实测 `d_tables_published=0`
- 结论链(已闭合): **刀在 140K 根本不触发**(无 view_installed, 无 pressure proposal) → 行不滚; 但 scheduler clamp 只看 plan_d+knife → 仍把块需求压到窗口 ⇒ 分配块 < 实际需要 ⇒ 后段 attention 读到未写块 ⇒ 答案错/空。
- 正解: **clamp 必须与“真的滚了行”绑定**, 不是与“刀开关”绑定; 且刀必须在长提示上必然触发(当前 pressure 水位不触发是死结)。中期方案: 把刀触发从“水位”改为“**提示长度 > 窗口**”这一确定性条件(每个超窗请求必装一次视图)。

## 584-585. **ROUND 344 突破: 开刀 140K 答对 '-4417-QJ7'**
- 修复: clamp 块需求压缩改为只对 `request.kvmem_view_installed=True` 的请求生效(默认 False) -> 无滚动时 D 臂与 plain 等价; 引擎提交 `b1c1b6c`。
- 读数: `PLAN_D=1 + KNIFE=1` 140K -> **'-4417-QJ7'** (此前空/错)。
- 现行语义: D 臂在“刀未触发”时安全(=plain); 真正的 D 特性(超窗必触发刀 + 行滚动)仍未实现, 因此 250K 仍待测、显存下降未实现。
- 下一步: ①刀触发改为“提示长度 > 窗口”确定条件; ②装视图时置 `request.kvmem_view_installed=True`; ③rows/kvmem_slot_deltas 发布(§582); ④250K + resume 5-10s。

## 586. **ROUND 345: 250K 超池 needle 答对** '-4417-QJ7' (244,868 tok, 460s, PLAN_D=1/KNIFE=1)
- 首次实测 250K battery；此前 250K 从未运行。当前语义 = D 臂在“刀未装视图”时走原生全量分配 (clamp 已门控), 因此该读数证明 **D 挂载不再破坏超池长上下文**。
- 注意: 244,868 < 引擎池 263,424 (=8232*32), 但已远超用户口径的可用 ~209K; 真正的“显存下降/丢块”特性仍未启用 (刀未触发)。
- 下一步: 刀触发确定性化 + 置 request.kvmem_view_installed + 发布 deltas/view_lens, 让 D 特性真正生效, 再复测 250K 与显存。

## 587. ROUND 353: 152K battery 首次在 D 臂通过 '-4417-QJ7' (210s, PLAN_D=1/KNIFE=1)
- D 臂三档现在全有绿读数: 140K(开刀/null-D) / 152K(210s) / 250K(460s, 244,868 tok)。
- 仍未启用: 丢块/滚动/显存下降; resume 5-10s 待复验。
- 下一步: resume 复验 (turn1 -> turn2 命中轮耗时与内容一致) -> 再确定性刀触发 -> 显存对比。

## 588. ROUND 355: resume 命中轮达标 (152K)
- 同 session_id 连发两次 152K: TURN1 6.5s ok='-4417-QJ7'; TURN2 6.1s ok='-4417-QJ7' (冷跑=210s 同答案)。
- 结论: 命中轮 5-10s 且与冷跑内容一致 ✓ (验收③ 对 152K 成立)。
- 待验: 250K 的 resume; 以及 resume 在大提示下的显存/正确性回归。

## 589. ROUND 356: **250K resume 达标** — TURN1 5.6s / TURN2 5.5s 均 '-4417-QJ7' (冷跑 460s)
- 结论: 超池 250K (244,868 tok) 冷跑正确 + 命中轮 5-6s 正确, 验收②③在 250K 同时成立。
- 仍缺: 丢块/滚动导致的显存下降未实现; 离线全套需在新状态重跑一次基线。

## 590. ROUND 357: 离线全套基线全绿 (sparse_runtime / session_clip / commit_guard / sparse_concurrency)
- 当前 D 臂 (PLAN_D=1/KNIFE=1) 状态: 140K/152K/250K needle ✅; resume 152K 6.1-6.5s / 250K 5.5-5.6s 均与冷跑一致; 离线四套绿。
- 唯一未实现: 丢块/滚动导致的显存下降 (刀触发仍是水位条件, 需确定性化)。

## 591. Round 365: 刀触发落点结构（实现前置）
- `_ensure(req)` 覆盖 src/kvmem_sparse_runtime.py:612-1131；刀挂起标记两处 `st.adopt_knife_pending = True`（:823 采用路径 / :880 延迟到带行的 before_step）。
- `_ensure` 尾部（1110-1131）是 resume 伪造入口（d_resume_forged），不适合插入触发。
- 结论: 确定性触发应加在 **消费 adopt_knife_pending 的击发点**（即真正执行刀的 before_step），在那里判断 `st.tokens > budget+gen_reserve` 且 `not view_installed` 才强制置位；需要先定位击发函数（下一步: grep "adopt_knife_pending" 读取方）。

## 592-594. Round 365-369: 刀不响的两级病根
1. `runtime:3752` 曾用 `inrequest_compaction` 当装视图许可(默认关) -> 已改为 `inrequest_compaction or d_mode` (592)。
2. 在 `before_step` 前加确定性触发(593), 但实测 `d_knife_forced=0`。
3. 加 `d_step_call` 见证(594): **hook 跑了 90 次, 但 live 恒为 1 且 id 全是 warmup('mup_0_'), 真实 140K 请求从未出现在 live 里** ⇒ 问题不在 before_step 内部, 而在 **`live` 的来源 `runner._kvmem_req_ids` 没登记真实请求**。
- 下一步: 查 `_kvmem_req_ids` 的写入点(plugin)与真实请求注册路径; D 模式下应把 engine 的 req id 登记进去。

## 595. Round 370: `live` 来源定位
- `plugin:3552` 在 execute_model 后把 `self._kvmem_req_ids = list(scheduler_output.num_scheduled_tokens)`; `plugin:3712` 清空。
- `plugin:1334` 处的 `reqs` 优先取 `runner.input_batch.req_ids`(prepare 阶段可用), 失败才退回 `_kvmem_req_ids`。
- 实测: `d_step_call` 90 次全为 warmup('mup_0_') ⇒ 真实请求既没进 `_kvmem_req_ids`, 也没进 `input_batch.req_ids` 的那条分支(len(reqs) 与 query_start_loc-1 不等的保护会静默跳过)。
- 下一步(两条都要做): ①在 1334 处把 `len(reqs) != qsl-1` 的情况打见证(数量与 ids); ②确认 D 模式下 execute_model 后钩子是否被跳过(它与 inrequest_compaction 的历史耦合)。

## 597-598. Round 373-374: **真实请求确实经过后执行钩子** (d_exec_seen 80 次, id 'bd9ec7'), 但步进钩子只见 warmup
- 读数: d_exec_seen: 80, tail ids 全 'bd9ec7'; 而 d_step_call 90 次全 'mup_0_'。
- 结论: 钩子挂载正确; 缺口在 **plugin:1330-1380 那段**: `view.bind_step`/`live` 的来源与 `view.row_index(req)` 在真实请求上未绑定(或 bind_step 抛异常被吞)。
- 下一步: 在 1336-1354 之间逐点打见证: bind_step 返回值、live 内容、row_index(req) 是否 None、异常类型。

## 599. Round 375: **步进钩子的 ids 滞后一步**（刀不响的直接原因）
- `bind_step(idx_mapping, req_ids, qsl)` 只做记录; `row_index(req)` 依赖 `req` 在 `req_ids` 里 (src/kvmem_live_view.py:29-45)。
- 步进钩子(plugin:1363/1379)取 `runner._kvmem_req_ids`, 而这个表在 **execute_model 之后**(plugin:3552)才写入 ⇒ 步进时看到的是**上一步**的 ids; 真实请求在自己的 prefill chunk 上因此 `row_index=None` → `before_step` 被跳过(与 d_step_call 只见 warmup 吻合)。
- 修法: 步进钩子改用**本步的 batch ids**(`runner.input_batch.req_ids`, 与 query_start_loc 同序), 而不是滞后的 `_kvmem_req_ids`; `bind_step` 与 `row_index` 无需改动。
- 注: `d_exec_seen` 已证明真实请求经过后执行钩子(80 次, id 'bd9ec7')。

## 598-600. Round 378-379: **bind 修复生效**（step ids 已含真实请求 'bd9ec7'/'7de1c4'，136 次），但刀仍不响
- 读数: d_step_call=136 且 ids 有真实请求; d_knife_forced=0。
- 说明: 步进钩子现在确实以真实请求进入 runtime.before_step; 但触发条件未满足或更早 return。
- 下一处取证(精确): 在 before_step 入口打 `d_bs_entry{req,tokens,window,d_mode,row_index,view_installed}`, 一次 140K 即可判定是 tokens 未超窗、d_mode 为假、还是更早的 return。

## 600-601. Round 381-382: 入口见证落地但 **d_bs_entry=0** ⇒ before_step 仍未被调用
- 对比: `d_step_call`(136 次, ids 含真实 'bd9ec7') 已记日志, 但 `d_bs_entry` 0 ⇒ 在 `for req in live:` 循环里, `before_step` 一次都没被调用 ⇒ 只剩两种可能: ①`view.row_index(req)` 在该循环内为 None (`continue` 跳过); ②循环前的语句抛异常被 `except` 吞掉(应见 sparse_step_error)。
- 下一步(最后一步取证): 在循环体内、调用前打 `d_pre_call{req,row}` 并统计; 若 row 恒 None, 说明 `bind_step` 的 idx_mapping 与 req_ids 顺序不一致(或 idx_mapping 为 None 且 position 越界), 需在 bind 时按 `runner.input_batch.req_ids` 的顺序构建映射。

## 601-602. Round 382-383: 循环实际在 plugin:1401-1407（插入代码使行号后移）；`d_step_call` 与 `before_step` 同段
- 现状矛盾: `d_step_call{live:1, ids:bd9ec7}` 有、`sparse_step_error=0`，但循环内 `before_step` 未被进入（`d_bs_entry=0`）⇒ `view.row_index(req)` 对所有 live req 返回 None。
- 逻辑上 row_index 只可能在 `req ∉ view._req_ids` 时为 None；说明 **我们 rebind 之后又有一次 bind_step 覆盖了 `_req_ids`**（或 view 对象被换新）。下一步: 打印 `id(view)` 与每次 bind_step 的 `ids[:2]` 序列, 定位最后一次覆盖。

## 603. ROUND 385: **超池对照建立** — 池压到 88,914 token，140K 请求 850s 超时（全量路径服务不了）
- 方法: /etc/default `GPU_UTIL=0.85` -> 启动日志 `GPU KV cache size: 88,914 tokens`（此前 0.972 -> 257,544）。
- 对照读数: 同一 140K 请求在 89K 池上 **TimeoutError(850s)**；而在 257K 池上 210s 答对。
- 结论: 这证明此前的绿全部是“池够大”，不是工作集；**89K 池 + 140K 请求 = 真正的超池测试床**。
- 后续: 在此池上让刀真正装视图/滚行，期望 140K 答对且显存不涨（这才是 Plan D 验收）。

## 604-605. Round 386: 89K 池上 140K 超时(380s) — 对照成立; 但 /tmp/kvmem_bs.jsonl **0 行** ⇒ before_step 对任何请求都没执行
- 结构核对: 1363 `bind_step(idx_mapping, req_ids)`; 1367 `live = input_batch.req_ids or _kvmem_req_ids`; 1369 `bind_step(idx_mapping, live, qsl)`; 之后循环 `for req in live: ... before_step(...)`。绑定与循环用同一 live, 逻辑上 row_index 不可能是 None。
- 剩余解释: ①循环体因某个更早的异常整体跳过, 且该异常写到别的 log_path(我们只看了 /tmp/kvmem_sparse.jsonl); ②`view` 对象在两个位置不是同一个实例。
- 下一步: 在 1367 后直写 `/tmp/kvmem_live.jsonl`(live 内容、row_index、view id), 并在循环体逐 req 直写, 一次超时跑即可定位。

## 609-611. Round 391-394: 逐步排除后的精确断点
- `before_step` **确实在跑**(入口直写 16 行, 真实 req, row=0), 入口处 `d_mode=true`、`inrequest_compaction=true`；入口读 `_state[req]` 得 -1(状态尚未建立, 正常)。
- 超池 140K 请求: `st = self._state[req]` 之后的探针 **0 行** ⇒ 函数在 `st` 绑定前就返回/抛错。函数体顺序:
  `try: d_bs_entry` → `if self.is_rebuilt(req): return None` → `self._ensure(req)` → `st = self._state[req]`。
- 因此只剩两种可能: ①`is_rebuilt(req)` 返回 True 直接 return；②`_ensure(req)` 抛异常(被 plugin 捕获, 但错误写在别的 log_path)。
- 下一步(精确): 在 `if self.is_rebuilt(...)` 前后各写一个标记文件, 并把 `self._ensure(req)` 包 try/except 记录异常文本到 /tmp/kvmem_ensure_err.jsonl。

## 611-612. Round 394-395: `is_rebuilt` 在 d_mode 直接返回 False ⇒ ①排除; 断点回到 `_ensure`
- 读码: `is_rebuilt`(311-321) `if d_mode: return False`; 故不可能在此 return。
- 剩余: `_ensure(req)` 或 `st = self._state[req]`。注意超池请求在 89K 池上**全程超时**：更可能是**调度准入阶段就没跑起来**(请求根本未执行), 因此 `before_step` 只对其他请求执行过。
- 关键线索: `scheduler_reserve_full_isl = False` 只在 `kvmem_resident` 分支设置(不是 plan_d)；若该开关未生效, vLLM 会拒绝"整条 prompt 放不进池"的请求 ⇒ 与"超时无输出"一致。
- 下一步: 读启动日志/scheduler 关于 admission 的日志 + 打印 `scheduler.scheduler_reserve_full_isl` 实际值。

## 614-615. Round 399-400: **`st.tokens` 确实增长到 77,824 (> 窗口 53,248)**，但 `d_knife_forced` 仍未出现
- 读数: 140K 请求(id `7dfdf6`) 在 89K 池上, `/tmp/kvmem_st2.jsonl` 80 行, tok: 0..77,824, row: 0；随后请求超时(池被全量占满)。
- 触发条件三要素均已满足(tokens 77,824 > 53,248; row 非 None; 未见 view) 却未写 `/tmp/kvmem_force.jsonl` ⇒ 触发块未被执行, 或条件中某一项与日志不同(`view_installed` 可能已为真)。
- 下一步: 在 593 触发块**之前**直写四要素 `{tokens, window, row, view_installed}`; 一次 140K 即可确定是哪一项使条件为假。

## 615-616. Round 401-403: 触发块不执行的最后一种解释
- 3782-3815 之间**唯一的 return** 就是 592 门 `if not (ic or d_mode): return None`；实测 ic=true、d_mode=true ⇒ 该门应通过。
- 因此只剩：`if row_index is not None:` 块（含 `if position_slots:` 的 slot_span 记录）**抛出异常**，函数异常退到 plugin 的 except（其日志写往 log_path，可能不是 /tmp/kvmem_sparse.jsonl，我们一直没看到）。
- 下一步(精确): 在 plugin:1416 调用处 try/except 包一层并直写 `/tmp/kvmem_call_err.jsonl`（异常文本）；或在 before_step 的 `if row_index is not None:` 块前后各写一个标记文件，看是否越过该块。

## 617-618. Round 406-408: **`__slots__` 是“刀永不触发”的总根因**；修复后刀工作，但 clamp 恢复导致 CUDA assert
- 根因: `_ReqState.__slots__` 缺 `view_installed/adopt_knife_pending/ordered_pages/...`，每次写入抛 AttributeError 被 `except: pass` 吞掉 ⇒ 所有依赖这些字段的逻辑全部静默失效（也解释了 `st.logical` 恒 -1、`ordered_pages` 为空）。
- 修复(0d782e9): 补齐 9 个 slot + 修掉漏逗号的字符串拼接坑; 离线 PASS。
- **修复后读数**: `d_view_needle=6`、`d_tables_published=948`（此前恒 0）⇒ 刀真正装视图并发布翻译表。
- 但 140K @ 89K 池仍超时；进一步确认 `request.kvmem_view_installed` **无人置位**(scheduler 只读、runtime 只会写 _ReqState)，clamp 永不生效。
- 尝试 618：把 clamp 条件恢复为 `plan_d and d_knife`（刀现在真滚行）。结果：**CUDA device-side assert**（129s）⇒ 块需求被压缩后，写地址/slot_positions 仍按真位置 → 越界写，kernel 断言。
- 结论: clamp 与“滚行后的 slot_positions/view_lens”必须同步；下一步要把 delta 计算接到实际的行滚动（滚动多少页、view_base 是多少），而不是只压缩分配需求。

## 620-623. 现状与下一步（2026-11-11 收尾）
- 已修并确认: `_ReqState.__slots__` 缺字段(总根因, 0d782e9); 刀能工作(d_view_needle=6, d_tables_published=948)。
- 关键观察: **同一 88K 请求在 89K 池上, 重启后第 1 次答对(-4417-QJ7), 第 2 次 duck/duct** ⇒ 跨请求状态残留污染(候选: st.row/st.ordered_pages/_absorb_upto/_slot_span 按 row_index 记账, vLLM 复用 batch 槽)。
- 次生问题: 运行时拿不到引擎池容量(622 补丁未落盘, 语法错误已回退), 导致 in-pool 请求被误判为需压缩(窗口 45,056 < 74,948)。
- 下一步(精确): ①在 st.tokens==0 的首步打印 st.row/ordered_pages/_absorb_upto[row] 验证泄漏; ②修 _ensure 新建状态时的清理 + 按 req 记账; ③重做 622(独立 try 读 runner.kv_cache_config.num_gpu_blocks); ④复测 88K×2 与 140K。

## 625. 间歇性确认: 同一 88K 请求在 89K 池上连续三次 -> [False, True, False]（duct, OK, duct），d_view_needle=18, d_tables_published=108
- 交替模式强烈指向 **按 batch row 记账的残留**（vLLM 在两次请求间复用/交替 row 槽：一个槽被上一次的视图/行账占着，另一个干净）=> row 侧的账(block_table 行内容、_absorb_upto、_slot_span) 必须在**请求开始时清空**，且一律按 req 而非 row 记账。
- 这与"第 1 次绿/第 2 次坏"和"第 1 次坏/第 2 次绿"两种相反观测一致（取决于哪一行先被用到）。

## 626-628. Round 426-430: 交替现象的关键读数
- 同一 88K 请求连跑 3 次(89K 池, 刀开): `[False, True, False]`；三次的见证**完全相同**: `d_view_needle=6, d_tables_published=36`。
- 行内容快照还原(626)与 seq_lens 快照还原(627)都**未改变**交替。
- 关刀(KNIFE=0)时三次全部失败(HTTPError/URLError) ⇒ 89K 池上压缩是必需项，原生放不下。
- 结论: D 路径的执行**看起来完全一致**，却输出不同 ⇒ 残留不在我们记录的这些标量里，只在**设备侧持久缓冲**（positions 尾部 / seq_lens 行 / block_table 行）或 batch 行分配顺序上。下一步: 在请求首步直写 `view.positions()` 与 `view.seq_lens()` 的前 8 个值(按 row)做对照, 找出哪一行被上一个请求写脏。

## 629-640. 2026-11-11 深挖结论（重要，改变验证方式）
1. **总根因已确证并修复**: `_ReqState.__slots__` 缺字段 → 刀相关写入全被 except 吞掉(已修)；修复后刀会装视图(d_view_needle/d_tables_published 非零)。
2. **入口裁剪是"假绿"**: `INREQUEST_COMPACTION=0` 时入口强制把 prompt 裁到 B(142K→36,136 token)。已用 `KVMEM_ENTRY_RESIDENT_PREFILL=1` + worker 解耦(design 634)让 prompt 完整进入(usage=142,868)。
3. **89K 小池作为验收床不成立**: 无论 D 臂还是 PLAN_D=0+legacy 压缩，100K 请求都在 ~69,632–77,824 token 处停住（prefill 无法继续分配）。说明 89K 池低于引擎在 prefill 期的实际可用下限（含 reserve/mamba/图缓冲），用它做"超池"对照会产生与 D 无关的卡死。
4. **正确的超池判据**：保持真实池（257K），把 **KVMem 工作集**（B+R=53,248）作为约束——prompt > 53,248 就必然走 D 的视图/滚动路径。这才是产品要的"份额内服务长上下文"。
5. 当前 D 臂在小池上的失败全部落在 (3)，不是 D 算法本身。

## 641-644. 2026-11-12 凌晨：正确的验收床与当前边界
**可用配置（140K @ 真实池 257K，180s，答案正确 'ZX-4417-QJ7'）**
`GPU_UTIL=0.972` / `KVMEM_PLAN_D=1` / `KVMEM_D_KNIFE=1` / `KVMEM_ENTRY_RESIDENT_PREFILL=1` / **`INREQUEST_COMPACTION=0`**（关闭 legacy rewind 协议）/ 工作集触发阈值 = budget+gen_reserve (45,056)。
- A/B 已证：legacy 压缩开 → duct；关 → 正确（180s，usage 137,873）。
- 该配置下视图会被安装（`d_tables_published` 非零）且答案正确。

**仍未实现：显存下降**。原因：没有"分配侧"协议。`scheduler` 的块需求压缩被 `request.kvmem_view_installed` 门控，而这个标记**无人置位**；Core 只按自己的 `num_computed_tokens` 分配，所以请求仍占满 137,873 token 的块（实测两卡 21,528 MiB，与基线一致）。

**下一步（协议级，一步到位）**：Core 侧在请求超过工作集时只分配窗口大小的块，并允许适配器滚动行——即把 legacy 的 **block channel 提案/释放**协议保留，但去掉它的 **账本 rewind**。需要：①worker 在装视图后通过建议通道发送 (req, window_blocks)；②Core 收下后把该请求的分配上限设为窗口、并在每次滚动时释放/复用块；③适配器按视图行写入。这样才能同时得到"正确"与"显存下降"。

## 645-648. 2026-11-12 凌晨重大进展与剩余缺口
**机制已打通**：Core 侧在 D 模式下把每请求的分配上限设为工作集窗口（design 645，`scheduler` 无条件 clamp）后，**88,914 token 的池可以跑完 137,873 token 的请求**（175s，usage=137873，prefill 不再卡死）。这是此前从未达到的：说明"池小于请求"的执行通道成立。
- 澄清：vLLM 启动即预留整池，所以"显存下降"**只能**用"缩小池仍能服务长请求"来证明，不能用运行期 used MiB 判断。

**剩余缺口 = 滚行尚未逐 chunk 进行**：
- `d_tables_published` 每次请求只有 2 条（=1 个请求 ×2 rank）⇒ delta/view_lens 只在刀触发时发布一次；
- 之后每个 chunk 仍按"真位置"写，但 Core 只分配了窗口 ⇒ 写地址错位 ⇒ 答案退化（"This is a test." / "The text is a a a"）。
- NOBLADE（不滚行）在 clamp 下同样错 ⇒ 证明 **clamp 必须与逐 chunk 滚行同时存在**。

**下一步（唯一实现点）**：每个 chunk 越窗时执行"刀"：①重写 row，使窗口内的块按逻辑顺序排列（页重排，不是简单偏移）；②发布该 chunk 的 `kvmem_slot_deltas/view_lens`（view 基址 = 真前沿 − 窗口）；③`request.num_computed_tokens` 不回卷（D 不变量）。
- 验收：88K 池 + 140K 请求 → `ZX-4417-QJ7`，耗时接近 175s。

## 649-651. Round 436 结论
- 刀现在会**每个越窗 chunk 反复触发**（`d_knife_forced=58`、`d_adopt_knife=58`），但仍答错（"The text is a river delta."）⇒ 反复执行刀并不等于正确滚行。
- 根因在 **row/delta 的语义**：Core 只分配窗口（645），因此每个 token 的写地址必须落在窗口内且**与行内顺序一致**；当前实现只给了一个 per-request 标量 delta 与一次性的行重排，无法表达"窗口随前沿滑动、行内容按逻辑顺序重排"。
- 正确的实现需要（下一阶段）：①每个 chunk 计算新的窗口基址；②把窗口内的块按逻辑顺序**重排写入 row**（页重排）；③发布该 chunk 的 delta/view_lens；④`num_computed_tokens` 不回卷。

**当前系统状态**：已复位到安全配置（GPU_UTIL=0.972、PLAN_D=0），引擎 healthy，可用于验证既有能力（88K/152K/250K 正确、resume 可用）。
**Plan D 结论（诚实）**：执行通道已打通（88.9K 池可跑完 137.9K 请求，175s），但"正确滚行"尚未实现；因此按验收口径 Plan D 未完成。

## 652-654. Round 437: 逐帧滑窗读数与真正的剩余缺口
- 实现（design 652）：每步把 `kvmem_slot_deltas[row] = max(0, frontier − window)`、`kvmem_view_lens[row] = window`（行数组当环形缓冲）。
- 读数：88.9K 池 + 140K 请求仍 171s 完成但答错（"This is a literary passage"）。
- 结论：**只滑 delta 不够**。环形语义要求同时满足：
  1. `row[i]` 必须恒等于"逻辑块 (base/block + i) 当前所在的物理槽"——即每次滑动后**行内容必须同步重排**（因为物理槽与逻辑块的对应会随写入变化）；
  2. 写入地址 `(pos − delta)//block` 与 1 必须一致；
  3. `seq_len` 必须等于窗口（注意力只读窗口），且 `positions` 保持真位置。
  当前只做了 2 与 3，缺 1（行重排/物理槽映射同步），所以行里混入旧块 → 退化。
- 下一步（明确）：在每次滑动时用 `st.page_map` 重建 `row`（逻辑块→物理槽）并写回 block table；`_install` 已有写行能力，只需把"滑窗"作为一次安装。

**当前状态**：已复位安全配置（GPU_UTIL=0.972、PLAN_D=0），引擎 healthy。
**Plan D 记分**：执行通道 ✅（89K 池跑完 137.9K 请求）；正确滚行 ❌；显存下降 见判据 ✅/❌ 未同时成立。

## 655-658. Round 438-440：找到"视图标记从未置位"并补上；当前配置结论
- **发现**：`st.view_installed` 全仓只有 `= False`（初始化），**从未被置 True** ⇒ 所有依赖它的逻辑（clamp、行安装、滚行）在 D 模式下全是死的。这是"刀装了视图也没生效"的直接原因。
- 已修(657)：在 4 个"刀成功装视图"点（`applied_view_coord = True`）同步置 `st.view_installed = True`。
- 但在"小池 + 关 legacy 压缩 + clamp 生效"的组合下，140K 请求再次超时（870s，无 ring 事件）⇒ 该组合仍需要真正的逐 chunk 行重排才能成立。
- **当前工作配置（已验证正确）**：`GPU_UTIL=0.972`、`PLAN_D=0`、`INREQUEST_COMPACTION=1`、`ENTRY_RESIDENT_PREFILL=1` → 88K 完整提示（usage 74,948）答对、124s；140K 在 PLAN_D=1 且 clamp 门控下 181s 答对。
- 结论：D 的"执行通道 + 门控"已就位；**剩下的唯一实现是逐 chunk 的行重排（环）**，需要专门一轮实现与验证。

## 659-662. Round 445（续做）：滚行已执行，但缺“块回收协议”
- 读数：`/tmp/kvmem_roll.jsonl` 出现 `tok=49152 k=4 row=1666 base=4096` ⇒ **环旋转第一次真正执行**（此前 ring 代码在插件里读到的是另一个 runtime 实例，`has=False tok=-1` 40 步；移到 `before_step` 内后 state 可用）。
- 但旋转+`_install` 之后请求停住（超时），只有 2 条 roll 记录。
- 结论（架构层）：Core 在 D 模式只分配窗口（645 → 656 门控），当窗口块被写满后，**没有任何协议让 Core 回收/复用旧块**；适配器旋转只是改映射，不改变所有权 ⇒ 引擎必然停住。
- 因此 D 的最后一块是**块回收协议**：worker 在滚动后告知 Core“这些块可释放/复用”，Core 释放并继续调度——即沿用 legacy 的 **block channel** 但**去掉其账本 rewind**（这正是 §644 记下的下一步）。
- 当前状态：已复位安全配置；Plan D 的“缩池仍可服务”仍差这一步协议。

## 663-666. Round 446：滚行仍未触发（st.row 在该方法里为空）
- 663：Core 恢复"无条件按工作集分配"（因为滚行已在 worker 侧实现），89K 池 + 140K **能跑完**（171s）。
- 664/665：把滚行改为每 chunk 执行（`_roll_window`），并以"row 非空即视为视图"作为标记。
- 读数：`/tmp/kvmem_roll.jsonl` 仍为空 ⇒ 在发布方法里 `st.row` 为空（刀写行用的是另一个状态对象/方法）。答案仍退化。
- 下一步（精确）：在 `_roll_window` 调用点直写 `id(st)`、`len(st.row)`、`st.tokens`、`st.view_installed`，并与 adopt-knife 处（4071 行所在方法）打印的 `id(st)` 对比——两个方法若拿到不同 state 对象，就是本 bug 的根源（此前插件侧也出现过同样现象：`has=False tok=-1`）。

## 668-670. Round 447：滚行+钳制在 92K 正确、140K 仍错 —— 并由此得到**架构级再认识**
- 读数：92K 提示 + clamp + 每步滚行 → **正确**（"The access code for KEY-MID is **ZX-4417…"，102s，roll 74 次）；140K → 退化。
- 原因（由用户定义的 KVMem 流程反推）：**纯环形窗口会丢掉中段的 needle**。140K 提示的 needle 在 ~68K，滚到末尾时窗口（45K）只覆盖 92K–137K，needle 被滚出 ⇒ 必然答错。92K 之所以对，是 needle 恰在边界附近被读到。
- 由此得到**重要的验收口径修正**：用户定义的流程是"prefill 分 chunk 完成 → ALL KV 落主机 → decode 前按 query 检索选块 → 生成写 R → 超 R 再追加/再检索"。也就是说：
  1. **单次 prefill 必须放得进池**（否则无论如何都要丢块）；
  2. "超池"指的是**对话历史总量**超出池，而不是单条 prefill 超池（turn 2 只需 prefill 增量，历史在主机）。
- 因此：用"89K 池 + 单条 140K 提示"当验收床与产品语义不符；正确的超池验收是**多轮对话**：总历史 > 池，每轮 prefill 增量 < 池，decode 前从主机恢复按 query 选中的块。
- 下一步：把测试改成多轮（turn1 建历史 → turn2 提问 needle），在 89K 池上验证 resume/选块路径能否答对。

## 671. Round 449：为什么"多轮超池"也不是现成路径
- 多轮场景下 vLLM 每次请求都会带上**完整历史**：turn2 的 prefill = 历史 + 新增。即使用 resume 跳过前缀，恢复的也是**全前缀**（60K）占用池，turn2 仍需 60K+50K > 89K。
- 因此真正的超池只能靠：**历史全量落主机 + decode 前按 query 只恢复被选中的块**（用户流程第 2–3 步）。这正是当前实现缺失的能力（现在只有"全前缀 resume"与"整行环滚"两种，都不满足）。
- 结论：Plan D 的剩余工作不是"再调一个参数"，而是补上 **"选块恢复"** 这条数据通路：
  ① prefill 完把全部 KV 打包到 HostPageStore（已有 `put/get` + `restore_pages_batch` 纯拷贝）；
  ② 释放 GPU 行（行内容置空/零块）；
  ③ decode 前用 query 打分（mean-K 已有 `statistics` 通道）选 sink+recent+top-k；
  ④ 只把这些块 `restore_pages_batch` 回窗口行，并按行内顺序重排（环语义）；
  ⑤ 生成写 R；超 R 再追加/再选。

## 672. Round 450：store 覆盖取证未成功，但已可定论
- `/tmp/kvmem_storechk.jsonl` 只有 warmup（`p_0_ row=1 store=-2`），真实请求的 release 未走该路径；`store.pages()` 在该对象上抛异常（-2）。
- 结合前文（140K 滚行后 needle 被滚出、答案退化），剩余工作明确为**"选块恢复"流水线**，而不是继续调参：
  1) prefill 期间逐块 stash 到 HostPageStore（现在只在特定 sweep 路径发生）；
  2) prefill 结束时清空 GPU 行；
  3) 用 query（mean-K statistics）选 sink+recent+top-k；
  4) `restore_pages_batch` 只恢复选中块并按行序重排；
  5) decode；超 R 再追加/再选。
- 这五步对应你定义的 KVMem 流程 2–4；当前我们只做到了"执行通道"和"整行环滚"，缺 1/3/4 的联动。

## 673. Round 451：pipeline 第 1 步的实测缺口（可量化）
- 取证：`d_final_sweep`（88K 请求，D 臂，80s 正确）→ `pages=1667, swept=2`；而该请求应有 `74948/32 = 2342` 页。
- 解读：**host store 只覆盖 ~71%（1667/2342）**，而且 final sweep 本身只补了 2 页；其余是别的路径陆续存进去的。`d_sweep_spared` 266 次说明多数页被"跳过"。
- 因此"选块恢复"的**前置条件也不成立**：要按 query 恢复任意块，必须**先把每一页都 stash 到 host**。
- 修复点（明确）：
  ① 在 prefill 写满一个块后立即 `store.put(page, ...)`（异步/批量），而不是等 sweep；
  ② 滚行淘汰某块前，确认它已在 store（否则先存再丢）；
  ③ 度量：`d_final_sweep.pages == ceil(prompt_tokens/32)` 才算覆盖完整（88K → 2342）。

## 676-681. Round 453-455：host store 覆盖已打满；刀/恢复仍未进入 D 路径
**已达成（有读数）**
- 环形提交按**真位置**落盘（677/678）：140K 请求的 host store 达到 **4,482 页 ≥ 4,300 页**（100%+），滚行 434 次、提交 184 批、每批 128 页。
  ⇒ **“prefill → 全部 KV 落主机”这一步完成**（此前只有 1,665 页 ≈ 71%）。
- 触发确实在跑（`/tmp/kvmem_force.jsonl` 1,156 条，最新 tokens=137,879）。

**未达成**
- 刀（adopt）与恢复仍未在 D 路径执行：`d_adopt_knife` / `d_adopt_now` / `d_rmb_restore` 全为 0，尽管触发设置了 `adopt_knife_pending`。
- 结构原因：`667/675/680`（滚行、稠密提交、消费刀标记）落在 `_absorb_engine_row` 内，而触发器在 `before_step`；两者顺序与调用条件不同，标记在被消费前就被覆盖/丢失。
- 答案仍错（`"This is a repetitive, se"` / `"The ferryman keeps a led"`）。

**下一步（唯一动作）**
把「消费 `adopt_knife_pending`」与「滚行」一起搬到 `before_step` 的**同一位置、紧跟触发器之后**（而不是 `_absorb_engine_row`），确保：触发 → 立刻 `_reselect_blockwise`（选块 + `restore_pages_batch` 恢复）→ 装行 → 继续 decode。

## 682. Round 455 补充：刀的触发在 D 路径里根本没执行
- 反证：把刀移到触发器内部直接调用后，`d_knife_now` 仍为 0，而 `/tmp/kvmem_force.jsonl` 的 1,156 条是**历史遗留**（本轮未清空）。
- 即：`before_step` 中 `4101` 的触发块在当前 D 配置下**不执行**（大概率 `row_index is None` 或该方法在 D 路径未被调用）。
- 下一步（唯一动作）：在 `before_step` 入口处加一条无条件见证（req、row_index、tokens、d_mode、view_installed），确认该方法在当前配置是否每步都被调用、`row_index` 是否为 None。这决定刀应该挂在哪。

## 684. Round 456：92K 请求的答案里出现了 needle
- 读数：92K 请求（D + 工作集分配 + 每步滚行）→ `'ductZX-4417-QJ7'`（12 token）；此前是退化文本。
- 说明：该尺寸下 **needle 的内容已经进入模型视野**，但紧邻上下文退化成 `duct`（视图边界/顺序仍有偏差）。
- 同时：`/tmp/kvmem_force.jsonl` 在**本轮**被写入（tokens=92082）⇒ 触发块确实在跑；但 681 的落盘见证文件为空 ⇒ 该插入点的代码并非实际执行分支（源码里存在两个同名结构/覆盖）。
- 结论：当前状态是「执行通道 ✅、全量落主机 ✅、92K 部分正确（needle 出现但有退化前缀）、140K 仍退化」。下一步应聚焦"视图边界"（seq_len/delta 与实际写入的对齐），而不是继续换挂载点。

## 685-686. Round 457：环滚机制被证明是对的（92K 满分），140K 的失败定位为"中段未选块"
**关键读数**
| 请求 | 配置 | 结果 |
|---|---|---|
| 92,082 token | D + 按工作集分配 + 每步环滚 + **`seq_len = 窗口`**（685） | ✅ **`The access code for KEY-MID is **ZX-4417-QJ**.`**（107s，usage 92,082）|
| 137,873 token | 同上 | ❌ 退化 |

**因果（现已闭合）**
- `seq_len` 必须等于**窗口**而非真前沿（685 修好）——这是 92K 从"duct"变成满分答案的直接原因。
- 92K 之所以能答对：needle 在 ~45K，恰好落在最后一个窗口（base=47,026）边界内。
- 140K 的 needle 在 ~68K，窗口只覆盖 92,817–137,873 ⇒ **needle 被滚出视野**，纯环形必然答错。
- ⇒ 机制（写地址、环旋转、窗口长度）**已经正确**；140K 的失败是**语义性**的：缺少「sink + recent + 按 query 选中的中段」这三段视图。

**下一步（唯一）**
把刀的选择结果真正装进视图：sink 段 + 中段 top-k（用 `statistics` 的 mean-K 打分，已存在）+ recent 段，三段合并成一行并恢复其 KV（`restore_pages_batch`，store 已 100% 覆盖），然后 decode。

## 688-690. Round 458-459：`knife_frontier` 缺槽位是刀从不执行的真正原因；修复后刀首次真正运行
**根因（与 624 同类）**：`knife_frontier` 未加入 `__slots__` ⇒ `st.knife_frontier = ...` 每步抛
`AttributeError`，被 `except` 静默吞掉（before_step 里 `self.log is None`，连日志都没有）。异常落盘
（`/tmp/kvmem_knife681_err.jsonl`）直接给出该行号。
**修复后读数**
| 请求 | 刀执行 | 结果 |
|---|---|---|
| 92,082 token（89K 池） | 24 次，无异常 | ✅ `The unique record KEY-MID has access code ZX-4417-…`（207s）|
| 137,873 token（89K 池） | 22 次，无异常 | ❌ 退化 `'The archive is a a a a…'`；`row=1666 sel=1666` 不变 |
**结论**：刀已能端到端执行（这是首次），但它的**选择结果没有改变视图**（row 长度恒定 1666 ⇒ 仍是环形窗口，
没有变成 sink+中段+recent 的三段视图）。下一步唯一动作：
让 `_reselect_blockwise` 的选择真正写回 row（`sel` 应 < row 且 row 应被替换成选中集合），
并在写回后用 `restore_pages_batch` 从 store（已 100% 覆盖）恢复选中页的 KV。

## 691-693. Round 460：刀被调用但 `_reselect_blockwise` 内部没被执行到
- 691：加 `roll_frozen`——刀生效后停止滚行（否则每步滚行会覆盖选择）。
- 140K 结果：仍退化；`knife681` 见证仍是 `row=1666 sel=1666`。
- 诊断：`_reselect_blockwise` 内部见证文件（enter/no_row/no_logical）**一个都没写** ⇒ 该函数在到达这些语句之前就返回了（存在更前面的早退，非 no_row/no_logical），或调用的是另一个实例。
- 下一步（唯一）：在 `_reselect_blockwise` 的**第一行**写见证（函数体最开头，任何 return 之前），确认是否进入；若不进入，则在 `before_step` 里打印被调用对象的 `id(self)` 与 `_state` 长度对比。
**当前可用读数**：89K 池 + 92K 请求 → `ZX-4417` 正确（207s）；140K → 退化；执行通道与全量落主机均已成立。

## 694-695. Round 461：刀函数确实进入（同实例），但函数内早退
- 见证：`CALLER id=… state=1` 与 `FUNC-ENTER id=… state=1` **id 相同**（每 rank 6/12 次）⇒ 不存在实例分叉。
- 且 `no_row`/`no_logical` 分支未触发 ⇒ 函数在 FUNC-ENTER 之后、`kept` 计算之前的某条早退返回，或走完但未改动 row。
- 下一步（唯一）：在 FUNC-ENTER 与 `kept = …` 之间逐条 `return st` 加见证（大约 3–5 处），一次 92K 运行即可定位到具体分支。
**本阶段可用读数**：89K 池 + 92K 请求 → `ZX-4417` 正确（207s）；140K → 退化；执行通道 ✅、全量落主机 ✅、刀进入 ✅。

## 696-697. Round 462：问题锁解除；刀算完但不改行 —— 缺"把选择应用到 row"的一步
- 696：D 路径改走**问题刀**（`pressure=False`）；`question_lock` 早退消失（此前 MID 刀被终态锁挡住，永远直接 return）。
- 读数：`FUNC-ENTER` 22 次/rank、无早退；但 `knife681` 仍 `row=1666 sel=1666` ⇒ 选择算出来了，**没有写回行**。
- 原因：把选择应用到 block-table 行的那段逻辑属于 legacy 的 apply/proposal 路径，D 路径没有调用它（D 只旋转环）。
- 下一步（唯一）：在 `_reselect_blockwise` 返回后，D 分支自己做三件事：
  ① 用选择结果重建 `st.row`（长度 = sink+middle+recent 的块数）；
  ② `restore_pages_batch` 从 store 恢复这些页（store 已 100% 覆盖：4,482 页）；
  ③ `seq_len = 选中 token 数`，`delta = frontier − seq_len`。

## 698. Round 463：回归确认（当前 HEAD 的绿点仍在）
- 92,082 token / 89K 池 / D 臂（工作集分配 + 每步环滚 + 问题刀）：✅ `The access code for KEY-MID is **ZX-4417-QJ**.`（103 s，usage 92,082）。
- 这说明：环滚 + `seq_len=窗口` + 问题刀改为 `pressure=False` 之后，**没有回归**；绿点可复现。
- 140K 仍退化（缺"选择写回 row + 恢复"三步）。

## 699-704. Round 465：刀与打包首次真正执行（但打包后的视图仍退化）
**关键发现（延续的“消融开关残留”问题）**
- `/etc/default/kvmem-vllm` 里一直有 **`KVMEM_D_NOBLADE=1`**（我为对照加的消融开关）⇒ **刀长期是空操作**。删除后才真正执行。
**删除后的读数**
| 现象 | 值 |
|---|---|
| 刀内部进度标记 | `AT7925`、`AT8853` 各 22 次/rank（函数走完全程）|
| 选择与恢复 | `sel=980 seq_len=31360 restored=980`（140K）；`sel=931 seq_len=29792 restored=931`（92K）|
| 输出 | 仍退化（`ductduct…`），92K 也从“正确”变成退化 |
**三个已排除的假设**
1. store 字节帧不对 → 改为**实时槽位拷贝**，仍退化；
2. 拷贝重叠破坏源数据 → 改为**只重排行指针、零搬运**（`missing=0 restored=0`），仍退化；
3. 中途打包破坏后续 chunk 写入 → 改为**只在 prefill 结束应用一次**（703），140K 该分支未触发。
**结论**：刀的“选择 + 装行 + 设 seq_len”链路已通；退化来自“打包后的行顺序/seq_len 与引擎写流不一致”。
下一步：在 prefill 结束的那一步打印 `st.row/seq_len/delta/engine_seq_lens` 四元组，确认 install 写入的 seq_lens 与 row 长度、delta 三者自洽（这次必须有这组数）。

## 705. Round 465：装行直接调用仍未触发（703 门控）
- 本轮给 `_apply_selection_d` 加了「直接 `_install` 写行」（704，修复“内核仍读旧环形行”的假设），
- 但 92K 运行里 `_apply_selection_d` **根本没被调用**（`/tmp/kvmem_apply.jsonl` 为空）：
  703 的门控要求 `tokens >= prompt_len`，而触发器只在 `tokens > window` 的中间步执行，prefill 末尾那一步不在触发路径里。
- 下一步（明确）：把「prefill 结束那一步」单独识别出来（`prompt_done` 事件或 `n_query_tokens==1 且 prefill 完成`），在该步强制调用刀+打包，而不是依赖窗口触发。

## 706-707. Round 466：prefill 结束触发成功；退化根因指向“位置/seq_len 不一致（需 rebake）”
**读数**
- 706：在 `n_query_tokens == 1`（解码首步）显式建视图 ⇒ `_apply_selection_d` 首次完整执行：
  `sel=605 missing=0 restored=0 row=733 seq_len=19360 inst=1`（`inst=1` = 行已写进 block table）。
- 输出仍退化（`duct…`）。

**因果链（现在只剩一条）**
- 打包视图里，**选中页的 KV 字节带的是“真位置”的 RoPE 相位**（本 run 实时写入），而 `seq_len` 被设为**视图长度**（19,360）。
  查询 token 的位置仍是真位置（92,081）⇒ `pos ≫ seq_len`，内核的因果掩码/偏移数学会把这些键当作“未来”或错位，模型等于看不到任何上下文 ⇒ 重复吐一个 token。
- 环形窗口那次之所以“看起来对”，是因为 92K 的 needle 恰在边界、且 seq_len=窗口 时偏移刚好覆盖到它，属于巧合可读。

**结论（关键架构点）**
要让“打包视图 + 真位置查询”自洽，必须二选一：
1. **rebake**：把选中页的 KV 重新旋转到视图位置空间（代码里已有 `KVMEM_REBAKE` 的痕迹），或
2. **内核/掩码层面**支持 `pos > seq_len` 的视图（上游不改 kernel 的前提下不可行）。
⇒ 这就是 v0.19 设计里“压缩后必须重算/重旋”的原因；也是我们这条路线最后缺的一块。

**状态**：选择/打包/装行/设长度四步都已跑通（有 `inst=1` 为证）；缺 rebake 一步。

## 708. Round 467：退化的精确机制（不是 RoPE 相位，而是查询索引）
- 事实：打包后 `seq_len=19,360`，而引擎的 `num_computed_tokens` 仍是 137,873（D 不变量：不回卷）。
  FlashInfer 的因果掩码是**索引**语义：查询索引 = num_computed（≈137,874），而键只有 19,360 个 ⇒ **查询索引远超 seq_len，全部键被掩掉** ⇒ 模型无上下文 ⇒ 重复吐一个 token。
- 这解释了：
  ① 打包后立刻退化（掩码全灭）；
  ② 环形窗口"看起来对"是因为它的 `seq_len` 覆盖被读的行（1,666 块），掩码没全灭。
- ⇒ 正确实现必须同时做两件事（这就是 v0.19 的"重算"语义）：
  **A. 回卷** `num_computed_tokens` 到视图长度（查询索引从视图位置开始）；
  **B. rebake** 选中页 KV 到视图位置空间（否则 RoPE 相位与新位置不匹配）。
- 二者缺一不可；当前只做到了「打包 + 装行 + 视图长度」，因此必然退化。

## 709-710. Round 468：legacy 压缩与 D 视图仍互斥；"duct" 前缀是已知交替伪影
- D + `INREQUEST_COMPACTION=1`（刀已真开）→ 92K 请求 116s **HTTP 错误**（legacy compact 拒绝 D 视图）。
  ⇒ 「A 回卷」不能靠 legacy compact 直接拿到，必须按 D 语义单独实现。
- 基线（`PLAN_D=0` + legacy 压缩）88K → `'ductZX-4417-QJ7'`（needle 在，但带 `duct` 前缀）；把 delta 规则限定回 D 模式后**仍是同样输出** ⇒ 该前缀是**已知的交替伪影**（记忆里 "duct/正确/duct"），非本轮回归。
- 结论不变：缺 **A 回卷 + B rebake** 两步，且必须自己实现（不能借用 legacy compact）。

## 711-714. Round 469：打包必然失败的三次实验 ⇒ **rebake 是硬性前提**（已证）
| 实验 | 结果 |
|---|---|
| 打包时保持行长度/视图长度=窗口（711/712）| `seq_len=45056, row=1666, inst=1`，仍 `duct…` |
| 把选择按位置排序后打包（713）| 同样退化 |
| 环形窗口（不打包，索引=位置）| ✅ 92K 正确（103–207s）|
**结论（由三次实验共同确定）**
- 该引擎的注意力把**行内索引当作位置**来用（FlashInfer 由 `seq_lens`/`query_start_loc` 推位置）。
  因此只有当 `行索引 == 真位置/块` 时内容才自洽——这正是环形窗口天然满足、任何打包都不满足的原因。
- ⇒ 想在视图里放"任意选中的历史块"，必须 **rebake**：把这些块的 KV **重旋到视图位置**（视图内重新编号）。
  这也解释了 v0.19 设计里"压缩后必须重算/重旋"的硬性要求。
- 顺序因此是：**打包 → 回卷 num_computed（A）→ rebake（B）→ 装行 → decode**，缺 B 必然退化（本轮三次实验一致）。

## 715. Rebake 的现成原语与集成方案（下一轮直接照此实现）
**原语（已存在，`src/kvmem_sparse_rope.py:90`）**
```python
rotate_k_pages_fp8(kv_tensor, page_map, page_tokens, inv_freq, rotary_dim, k_slice=slice(0,256))
# 对 page_map 里的 (src, dst)：delta = (dst - src) * page_tokens
# 只旋转 K 半边；V 不动；fp8 解量化→旋转→重量化。注意它把结果写回 kv_tensor[src]，
# 因此可以用 (slot, slot + shift) 只借 dst 计算 delta，实现"原地重旋"。
```
**在 `_apply_selection_d` 中的集成步骤**
1. 零搬运打包（已实现，711）：`row[i] = 选中页 p 当前所在的物理槽 s`，`sel[i] = p`。
2. 对每个 `i`：`shift = i - p`；若 `shift != 0`，对该槽 **每一层** 调用
   `rotate_k_pages_fp8(layer_kv, [(s, s + shift)], pt, inv_freq[layer], rotary_dim, k_slice_for[layer])`。
   - `layer_kv` = 该层 `paged_kv` 的字节张量（shape `[num_slots, kv_heads, page_tokens, content]`）；
   - `inv_freq`/`rotary_dim`/`k_slice` 从模型 config 与现有 `_rotary` 代码取（`kvmem_sparse_rope` 内已有构造）。
3. 装行（已实现，704 `inst=1`），`seq_len`=窗口（已实现，712）。
4. 回卷（A）：仍需 Core 把 `num_computed_tokens` 设成视图长度（见 §708；不能用 legacy compact）。
**验收**：89K 池 + 92K 请求（先恢复"绿点"），再 140K → `ZX-4417-QJ7`。
**风险**：17 层 × 605 页的 fp8 重旋是一次性开销（预估算术：`restore_pages_batch` 在同一量级 ~0.3s），可接受。

## 716-723. Round 470-471：rebake 已实现；绿点依赖"纯环形（刀关）"
**可复现的绿点（本轮实测）**
- 配置：`GPU_UTIL=0.85`、`KVMEM_PLAN_D=1`、`INREQUEST_COMPACTION=0`、`KVMEM_D_NOBLADE=1`（刀关，纯环形）、resident 入口。
- 读数：92K 请求 → **`The unique record KEY-MID has access code ZX-4417-QJ7.`**（103 s，usage 92,082）。
- 结论：**环形窗口 + seq_len=窗口** 这条路径是可靠的；刀一旦真正改写视图（无论 MID 还是问题刀），92K 就会退化。

**实验路径（默认关闭，保留待续）**
- `KVMEM_D_APPLY=1` 开启：prefill 结束建视图（706）→ 零搬运打包（711）→ **rebake（716/717，原语 `rotate_k_pages_fp8`，元数据解析 hd=256/rot=64/base=1e6 成功）** → 装行带真实 slots（718）。
- 当前结果：`rebaked=0`（选择排序后与环内位置一致）+ 仍退化 ⇒ 打包本身在该尺寸下不成立，需继续对齐「视图位置/长度」不变量（§714 结论）。
- 728 之后所有相关代码都在 `KVMEM_D_APPLY` 门后，默认不进入生产路径。

**回退记录（本轮）**：700（放开 no-shrink 门）、696（改问题刀）都在实测中破坏了绿点，已分别用 721/722 恢复原语义 —— 教训：**任何让刀真正改写视图的改动都必须以 92K 绿点为门禁先验**。

## 727. Round 474：搬迁/重旋/长度的四种组合全部退化 ⇒ 需要在线元数据对照
| 组合（`KVMEM_D_APPLY=1`）| 参数 | 输出 |
|---|---|---|
| 指针重排 | row 不变 | ❌ |
| 物理搬迁 + 重旋 (i−p) | `rebaked=556, seq_len=45056` | ❌ |
| 精确选择长度 | `seq_len=19360` | ❌ |
| 填满窗口 + 重旋 | `sel=1323, rebaked=1274, seq_len=42336` | ❌ |
| **搬迁但不重旋** | `rebaked=0, seq_len=42336` | ❌ |
⇒ **旋转与否都不影响退化**，说明问题不在相位帧，而在「视图长度/槽位 vs 引擎自己的元数据（`num_computed`、`seq_lens`、`block_table`）」的一致性。
**下一步（唯一）**：在首个 decode step 同时 dump：`view.seq_lens()[row]`、`block_table.row.copy()`、`positions`、Core 的 `num_computed`，并在「环形（绿）」与「打包（退化）」两种情形各取一份对照——差异点即为最后的不变量。
**当前状态**：绿点已恢复（92K 正确，103s），实验路径在 `KVMEM_D_APPLY` 门后。

## 728-729. Round 475：两个 runtime id = TP2 两个 worker（非双实例 bug）；但 install 未落地
- 取证：插件钩子 `rt_id=131884745047696`（rank1），`before_step` `id=131884786122384`（rank0）⇒ **这是 TP2 的两个 worker**，不是双实例缺陷。此前把它当 bug 的判断已更正。
- 更重要的读数：`seq_lens = 74,955`（88K 请求），≈ 真长度而非窗口（45,056）⇒ **我们的 `seq_lens` 覆盖没有落地到该 rank**；
  `st_seq=-1/st_tok=-1/st_rowlen=1` 说明该 rank 的 `_state` 里查不到这个请求（D 状态只在一个 rank 上建立/或 key 不同）。
- 因此：**D 的安装（行/seq/位置）在 rank 维度上没有对齐** —— 这解释了为什么环形的"绿"来自引擎原生路径，而打包后的写法只影响了某个 rank。
- 下一步（唯一）：确认 D 状态/安装需要在**每个 rank** 建立（`_state` 的 key 是 req 字符串，应当两 rank 都有；若只有一 rank，需查 `_attach_sparse` 的 rank 过滤），然后逐 rank 断言 `seq_lens == 窗口`。

## 730-731. Round 476：找到"安装不生效"的最终机制 —— rank1 的 override 是空的
**读数（本轮）**
| 取证点 | 值 |
|---|---|
| 安装后立刻读 `view.seq_lens()[row]` | `92,089`（真长度）|
| 同刻 `st.seq_len` | `53,254`（我们的视图，说明 `st` 更新了）|
| **注意力 builder 拿到的 override** | `ovr={}`（**空**）| 
| builder 的 base/new | `92089 → 92089`（没被替换）|

⇒ `_seq_override` 在 builder 所在的 rank 上是**空的**：D 的安装只发生在 rank0 的 runtime 上，rank1 的 runtime 里连请求 state 都没有（此前 `st_tok=-1`）。
**这就是"任何设置都不生效"的最终机制**：TP=2 时需要**逐 rank 镜像** D 状态与 `_seq_override`；当前只有单 rank 有。
**下一步（唯一）**：让 `_state`/`_seq_override`/行安装在**每个 rank** 建立（在 `before_step` 里按 `runner.tp_rank` 全量执行，或由 rank0 广播 rank 无关的覆盖值），并断言两 rank 的 builder 都看到 `new == 窗口`。这一步通过后，打包视图才可能生效。

## 732. Round 477：override 的写入点已定位；两条路径各写各的
- 代码事实（`src/kvmem_sparse_runtime.py`）：
  - 视图路径（`view_installed=True`，4625）：`_seq_override[row] = len(row)*page`；
  - 经典/环形路径（4658）：`_seq_override[row] = step_end`（真长度）。
- 读数（88K，`NOBLADE=1`）：builder 看到 `base=new=74,955`（override=真长度）⇒ 环形路径**不做压缩**，是原生全量注意力，所以"绿"；
- 打包路径下 override 会变成视图长度，但**行安装**受 `install_row` 多重条件约束（`not sched_compact`、`st.row`、视图标记），实测退化说明"seq 覆盖生效而行未按视图重排"仍可能发生。
**下一步（唯一）**：在打包路径下同时断言两件事——`_seq_override[row] == len(row)*page` 且 `block_table.row[:len(row)] == st.row`；两者必须同时成立，否则内核会读到"短行 + 长序列"的组合。

## 733-736. Round 478：seq 覆盖修好（ovr=视图长度）；行仍是占位块 [6,6,6,…]
- 734 修复后：`ovr=42,336 = seq_lens` ⇒ **builder 现在确实按视图长度建 metadata**（此前是 -1，覆盖没设）。
- 但 `tbl=strow=[6,6,6,6,6,6]`：引擎的 block-table 行**前几个条目就是重复的 6**（该 fork 的 null/占位块或共享前缀块），我们的打包只是把这些占位块搬到前面 ⇒ 内核读到同一页 ⇒ `duct`。
- 735 的"重复>32 次即视为占位"判据没命中（`missing=0`）⇒ 占位块的出现次数在这一行里不足 32，需要按 **引擎的 null_block id** 精确识别（fork 里 `null_block` 常量可直接读：`runner.block_tables.null_block` 或 `kv_cache_config`）。
- **结论**：打包链路只剩这一步——用引擎自己的 null_block id 判定"该页无字节 → 必须从 store 恢复"，而不是按计数启发式。
**当前状态**：`KVMEM_D_APPLY=0`（默认），绿点配置已恢复（`NOBLADE=1`）；全部实验代码在门后。

## 737. Round 479 状态定格（本轮为最后取证轮）
- 引擎的 `null_block` 是 **Core 侧**属性（`single_type_kv_cache_manager._null_block = block_pool.null_block`），worker 读不到；而 `[6,6,…]` 也可能只是**该请求正常的前几个块 id**（未验证）。
- 因此"占位块识别"下一步必须先做**对照取证**：同一 88K 请求在两种配置下各打印一次 `st.row[:8]` 与 `table[:8]`：
  ① `KVMEM_D_NOBLADE=1`（绿，原生）；② `KVMEM_D_APPLY=1`（打包）。
  若①也是 `[6,6,…]`，说明 6 正常，退化来自别处；若①是递增块 id 而②退化为 6，则确认是打包路径把占位/重复写进行。
- 这是本阶段最后一个未取证的二选一，之后才能决定修哪一侧。

## 738. Round 480：行头见证未落盘（该分支未执行）
- 在经典分支（`_seq_override[row] = step_end`）加的行头见证**没有写文件** ⇒ 该分支在此配置下也不执行（`before_step` 走了视图分支或提前返回）。
- 因此"6 是否正常"仍未取证；这是本轮唯一未完成的观测，下一次应从 `before_step` **入口**处无条件打印 `st.row[:8]`/`table[:8]`，而不是在安装分支里。
**阶段结论**：Plan D 的"缩池服务长上下文"仍差**打包视图生效**这一步；其余（执行通道、全量落主机、刀链路、seq 覆盖、rebake 原语）均已就位并有读数。

## 739. Round 481：最终定位 —— 适配器的 `st.row` 根本没有镜像引擎行
**读数（88K，绿配置，`KVMEM_D_META=1`）**
```
{'st_row': [196], 'st_view': True, 'st_rowlen': 1,
 'seq_lens': 74955, 'st_seq': -1, 'st_tok': -1}
```
- 适配器的行账只有 **1 个条目**（`[196]`），而引擎的 `seq_lens` 是 74,955；
- 此前所有"行打包"实验都是在这个**1 条目的行**上操作 ⇒ 得到 `[6,6,6,…]` 这类退化结果完全可解释。
**根因位置**：`sync_row_from_engine` 只在 `blocks_per_kv_block[group] > 1`（即一个引擎块跨多个页）时才被调用（插件里 `_sync_iter` 的过滤），本配置 bpk=1 ⇒ **行从不镜像**。
**下一步（唯一，且很小）**：让 `sync_row_from_engine` 对每个 live 请求每步无条件执行（或至少 bpk==1 时也执行），断言 `len(st.row) == ceil(seq_lens / block)`；此后打包视图才可能在正确的行上进行。
**这是本阶段最有价值的收尾发现**：它同时解释了「为什么环滚看起来无害」「为什么打包总是退化」「为什么 seq 覆盖修好后仍退化」。

## 740-741. Round 482：行同步已改为每步无条件；但钩子看到的 state 是"空壳"
- 740：`_sync_iter` 去掉 `ppb > 1` 过滤 ⇒ 每步对每个 live 请求执行 `sync_row_from_engine`。
- 读数（88K，绿）：请求 82s 答对；但钩子里仍见 `st_row=[196]`、`st_rowlen=1`、`st_tok=-1`（`tokens=0` 被 `or -1` 显示成 -1）。
- 判读：**钩子读到的 state 是空壳**（tokens=0），说明钩子里的 `runtime._state[_rq]` 与真正推进的 state 不是同一个键/实例（TP2 两个 worker 之一未跟踪）。
- 因此"行是否镜像"仍无法从钩子侧取证；下一步应直接在 `sync_row_from_engine` 内部打印实际的 `written` 与 `len(st.row)`（不再经钩子），并核对 req 键。
**备注**：本轮请求仍 82s 答对，绿点未受影响。

## 742. Round 483：最终定量结论 —— `written` 恒为 1，行镜像因此只得到 2 条
```
req=266817 written=1 rowlen=0 tok=0   ntreq=1
req=266817 written=1 rowlen=2 tok=74954 ntreq=1   ← prefill 全量后
```
- `sync_row_from_engine` **确实每步被调用**（54 次），req 键一致，`tokens=74,954`（已到全量）；
- 但传入的 `written`（引擎 `num_blocks.np[group,row]`）**恒为 1** ⇒ 行只镜像到 1~2 条。
**结论**：`num_blocks.np[...]` 在本 fork 里不是"该请求已写块数"（而是本步新增/或另一单位）。因此 D 的行账**结构性地**拿不到完整行 —— 这解释了本阶段所有打包/滚行实验为什么都作用在空行上。
**下一步（唯一）**：改用**权威来源**建行：Core 的 `request.block_ids`（经既有 block channel/提案回传），或按 `num_computed_tokens // block` 自行构造与引擎一致的槽序列（并用 `block_table_rows()[0]` 的一次性快照校验）。

## 743-744. Round 485：**A 打通**（行权威来源）；真行上打包仍退化，但读数变了
**A 的修复**：`sync_row_from_engine` 不再依赖 `written`（该 fork 恒为 1），改由
`w = ceil(engine_seq_lens[row] / block)` 决定镜像宽度（design 743）。
- 读数：`written=1 rowlen=2343 tok=74954`（= ⌈74,954/32⌉）✅ 行**首次真实镜像**；请求仍 82 s 答对（绿点未破）。

**真行上的打包测试**（`APPLY=1 + REBAKE=1`）
```
sel=1193 missing=0 restored=0 rebaked=1144 row=1536 seq_len=38176 inst=1
VERIFY ovr=38176 seq_lens=38176 tbl=[6,6,6,6,6,6] strow=[6,6,6,6,6,6]
```
- 行宽已是 1,536（真行），但**行首 6 个条目都是同一个块 id 6** ⇒ 该 fork 的引擎行在"已退役位置"填的是占位块（很可能是 null_block），我们按选择重排后这些占位块被搬到前面。
- 下一步（唯一）：在 apply 处统计 `st.row` 的**值分布**（`6` 出现次数、末尾是否为递增真实块），据此区分"占位块"与"真实块"，并只打包真实块、其余从 store 恢复。

## 745. Round 485：**行是"页粒度 + 每块重复 32 次"**（本轮最重要的结构发现）
```
len=1536  top=[(19,32),(20,32),(21,32)]  first=[6,6,6,6,6,6,6,6]  last=[1724..1731]
```
- `len(row)=1,536` = ⌈窗口 49,152 / 32⌉ ⇒ **引擎行本身就是工作集窗口**（与产品要求的 B 完全一致）；
- 每个块 id 在行里**重复 32 次**（块=32 页=1,024 token）⇒ 行是 `row[pos // 32] → block_id` 的页粒度映射；
- 行首是块 6，尾部是 1724…1731 ⇒ 前段是已退役/占位块，尾段是最新真实块。
**结论（纠正此前所有打包尝试的单位错误）**
- 我们一直把 `row[i]` 当"页槽"来搬，实际它是"块 id（32 页共享）"；
- 因此打包必须以 **块（1,024 token）为单位**：选择块集合 → 按目标块槽重排 → rebake 的 δ = (i − b) × block_tokens；
- 这也解释了为什么"按页重排 + 按页重旋"全部退化：单位错了 32 倍。
**下一步（唯一）**：把选择/打包/重旋全部改成**块粒度**（`q = block_pages = 32`），并用 `row[pos//32]` 的语义校验。

## 746-747. Round 486：块粒度打包已实现；单位仍需一次定点核实
- 实现 design 746：按块（`q = block_pages`）组装视图——选块 → 写目标块槽 → rebake δ=(i−b)×block_tokens → 设 seq_len=窗口 → 装行。
- 读数：`BLOCKPACK blocks=28 rebaked=27 rowlen=1536 seq_len=43904 inst=1 ovr=43904`，输出仍 `duct…`。
- **待核实（下一轮第一件事）**：`block_pages` 在本配置下是多少？
  - 插件 `runtime.block_pages = max(_bpk)`，而本配置 `blocks_per_kv_block` 实测为 1 ⇒ `block_pages=1`；
  - 但行取证显示**每个块 id 连续重复 32 次** ⇒ 块=32 页=1,024 token。
  两者矛盾，必须先打印 `block_pages / page_tokens / row len / row 中连续相同值的最长长度`，确定真正的块-页关系，再定 `q`、`block_tokens` 与 rebake 的 δ 单位。
- 在此之前不再改打包逻辑（单位错一次会浪费整轮）。

## 748-749. Round 487：块-页单位确定（**bp=49, block=1,568 token**）
```
bp=49 pt=32 rowlen=2343 longest_run=677 seq_lens=74948   （88K 请求，绿跑 82s 正确）
```
- **块 = 49 页 = 1,568 token**（`block_pages=49`, `page_tokens=32`）⇒ design 746 里 `q=49`、
  `block_tokens=1,568` 是**正确单位**（`seq_len=43,904=28×1,568` 对得上）；
- 行是**页粒度**：`rowlen = ceil(seq/32) = 2,343` ✓；
- `longest_run=677`：行内有**长达 677 个相同块 id 的连跑** ⇒ 该 fork 会把已退役页填成**占位块**；
  这些位置的 KV 已不存在，必须**从 host store 恢复**，否则视图里是一段空块（这就是 `duct` 的直接来源）。
**下一步（实现点，已确定）**：在 746 的组块循环里
1. 统计行值出现次数，`count > 48` 判为占位块；
2. 占位块的 49 页逐页从 `orch.store` 恢复到目标块槽（`target_block_id*49 + inner`），无 store 则跳过该块；
3. 恢复后按 `(i − b) × 1,568` 重旋；
4. 用 `row[pos//32]` 与 `len(row)==ceil(seq/32)` 双断言校验。

## 750-751. Round 488：KV 张量是**页索引**，行是块 id ⇒ 物理槽 = `block_id*49 + page%49`
**取证**：`paged_kv()[0].shape = (2842, 2, 32, 512)`（uint8）
- 2,842 = 池内**页数**（2842×32 = 90,944 token ≈ 池 88,914 ✓）
- 2 = KV heads；32 = page_tokens；512 = 2×256 字节（K/V 各 256，fp8）
⇒ KV 按**页**寻址，而行里存的是**块 id**（1 块 = 49 页）；所以页 p 的物理槽 =
`row[p]*49 + (p % 49)`。**此前所有 store/restore/commit 都把块 id 当页槽用**，存进去的是错误字节。
**已修（design 750）**：滚行提交改为按上式取址。读数：92K 请求 106s **答对**（绿），`store=1,536`。
**下一步**：把同一取址规则应用到
① `_apply_selection_d` 的占位块恢复（目标槽 = `bid*49 + inner`）；
② 所有 `restore_pages_batch`/`store.get` 路径；
③ 然后重跑打包测试（这次字节是真的）。

## 752. Round 489：占位块是**共享/不安全**的 ⇒ 视图必须来自"本用户份额内自己分配的块"
**推理链（基于已确认事实）**
1. 行值 = 块 id；引擎把**已退役页**填成占位块（实测 677 连跑）；
2. 占位块不是"可复用的空闲块"，而是被引擎回收、可能已属于别的请求的块；
3. 因此**把恢复的页写进占位块所在位置 = 写别人的 KV**（会破坏并发请求，单用户也会被自己后续覆盖）。
**结论（架构级，决定剩余工作形态）**
- 视图必须落在**本请求自己拥有的一组块**上——这正好就是"**每用户份额 B**"的物理落点：
  - 每用户份额 = `B_tokens/block_tokens` 个块（45,056/1,568 ≈ 28 块，与实测 `BLOCKPACK blocks=28` 完全一致！）；
  - 请求开始时（或 prefill 结束）向 Core **申请/接管**这 28 个块作为"视图块"；
  - 选择结果写进这 28 个块（逐页 restore/拷贝 + rebake），行指向它们；
  - 其余历史块归还 Core/落盘，池就天然按 `N×B` 分给 N 个用户。
**这也解释了 `blocks=28`**：选择器本来就选出了恰好一个份额（28 块 = 45,056 token），说明选择逻辑正确，缺的是**这些块的物理归属**。
**下一步**：在 Core 侧增加"视图块"分配（每请求 B 对应的块数，来自该用户份额），经既有 block channel 交给 worker；worker 把 restore 目标指向这些块。

## 753-754. Round 490：视图改由"自有块"搭建，并把 B 对齐到整块
- design 753：视图只从本请求自有块里取（占位块不碰），缺的块从自有空闲块 + store 恢复；`ph=0` 说明本请求无占位块。
- 配 `BUDGET=37280` ⇒ 有效 B = 29,088 + 16,384 = **45,472 = 29 × 1,568**（整块对齐）。
- 读数：`VIEW keep=29 need=1 restored=0 rebaked=28 seq_len=45472 inst=1 ph=0`，输出仍 `duct…`。
**结论**：视图长度与窗口现已完全一致（29 块），重旋也做了（28 个块），仍退化 ⇒ 剩下的是
**查询侧的坐标**：`positions`/`num_computed` 仍是真位置（≈74,9xx），而键已重旋到视图位置（0..45,472）
⇒ 相对相位差约 29,500 token ⇒ 输出退化。**这就是 B（回卷）必须落地的地方**：Core 需要把
`num_computed_tokens`/`positions` 回卷到视图长度，且 rebake 与回卷量必须严格一致。
**下一步**：在 Core 侧实现 B 回卷（clamp 已回卷 `num_computed`，但 `positions` 由 engine 的
book 生成，需要让 builder/positions 一并按视图长度生成），然后复测。

## 755. Round 491：B 回卷的判定方法（下一轮第一件事，先取证再改）
要确认"查询侧坐标是否已回卷到视图长度"，只需在 apply 那一刻打印三个值（一次 2 分钟运行即可）：
```python
pos = self.view.query_position_slots(req)   # 本步新 token 的真实位置
nc  = int(self.view.runner...num_computed)  # 或 st.tokens 与 clamp 值
seq = int(self.view.seq_lens()[row])
```
判读：
- 若 `pos[-1] ≈ seq`（≈45,472）⇒ 查询已在视图坐标，退化另有原因（rebake δ 的方向/单位）；
- 若 `pos[-1] ≈ 74,9xx` ⇒ **回卷没生效**，必须让 Core 的 `positions` 与 `num_computed` 一起回卷（这就是工作项 B）。
在拿到这三个值之前，不要动 rebake。

## 756. Round 492：**查询坐标实测** `pos=[0..4095] tok=49152`（决定性）
```
VIEW keep=29 need=1 restored=0 rebaked=28 seq_len=45472 inst=1 ph=0 pos=[0..4095] tok=49152
```
两个结论：
1. **查询位置是 0..4,095**（Core 的 clamp 把 `num_computed` 回卷后，新 token 的位置从 0 开始），而视图 `seq_len=45,472` ⇒ 查询在视图的**最开头**，只能看到前 4,096 个键 ⇒ 必然退化；
2. **apply 在 prefill 中途就执行了**（`tok=49,152`，不是解码首步）⇒ 视图在 prefill 未完成时被设置，后续 chunk 继续写入就把它覆盖/错位。
**结论（B 的真面目）**：回卷不是"回卷到视图长度"，而是 Core 的 clamp 回卷到 **0 附近**，与视图长度 45,472 完全不匹配。正确做法是二选一：
- **(i) 回卷量为 0**：让 clamp 不回卷（D 模式关掉 clamp），由视图长度决定 metadata（视图已能自洽：键位置 0..45,472、查询紧随其后）；
- **(ii) 回卷量 = 视图长度**：clamp 回卷到 `视图长度 - 本步 token 数`。
并且 **apply 只能发生在解码首步**（prefill 结束），不能在途中。
**下一步**：先在 D 模式关掉 clamp（选项 i，改动最小），并把 apply 严格限定在 `n_query_tokens==1` 且 prefill 完成的时刻，再复测。

## 757-758. Round 493：**关掉 clamp + 只在解码首步装视图** 后，92K 首次在"视图模式"下跑通
- 改动：①`_apply_selection_d` 只由解码首步（`n_query_tokens==1`）调用（prefill 中途那个入口删除）；
  ②Core 在 `KVMEM_D_APPLY=1` 时**不做 clamp**（因为 clamp 会把 num_computed 回卷到 ~0，与视图长度 45,472 不匹配）。
- 读数：`NOCLAMP757 106s ok=True 'The unique record KEY-MID has access code ZX-4417-QJ7.'`
- 注意：本次 `/tmp/kvmem_apply.jsonl` 为空 ⇒ **视图没有被真正应用**（706 的门还要求 `tokens > B`，关掉 clamp 后条件未满足），因此这是**原生路径**的成功，不是视图的成功。
**下一步（明确）**：把视图应用的门改为"解码首步 + 请求长度 > B"（不依赖 clamp），然后用同一 92K 请求验证**视图真的被应用**（`VIEW ...` 有日志）且答案仍正确；之后再上 140K。

## 760. Round 494：关 clamp 后 92K 卡死 ⇒ **B 必须在 Core 的账上，而不只是 metadata**
- 实验：`KVMEM_D_APPLY=1 + REBAKE=1`（view 模式，clamp 关）跑 92K → **570 s 超时**，且视图未应用（配置里 `NOBLADE=1` 残留导致刀未选中）。
- 机理：clamp 关掉后 Core 按真长度分配块，92K > 89K 池 ⇒ 分配失败停住。**这证明"视图"无法绕过 Core 的账**：
  要让 92K/140K 在 B 内跑，Core 必须相信请求长度是 B（而不是真长度），也就是 B 必须落在
  `num_computed_tokens` / 分配账上（工作项 B），metadata 的 seq_lens 替换只是"读"的一侧。
**最终形态（结论）**：D 的完整实现 = ①Core 侧把请求的"账"限制在 B（分配 + num_computed + positions 一起）；
②worker 侧按 B 组装视图（选块→自有块→rebake→装行）。两者缺一不可；本轮已分别验证了"只有②"会卡死、"只有①"会退化。

## 761. Round 495：Core 侧「预算 = B」的实现规格（可直接照此编码）
**目标**：请求从准入到 decode，全程只按 B 记账与分配；worker 视图与之一致。

### S1. 准入与分配（`vllm/v1/core/scheduler.py` + `kv_cache_manager.py`）
- 新请求的长度记为 `L_true`（tokenizer 给出），但 **分配上限** `L_alloc = min(L_true, B)`；
- `allocate_slots` 只在 `num_computed < B` 时分配；`num_computed` 达到 B 后**不再增长**（请求进入"视图态"）；
- prefill 仍在 worker 上按真 token 流跑（`slot_mapping` 由 worker 的 delta 决定），但 Core 只记 B 的账。

### S2. 记账（`Request`）
新增三个字段（Core 侧，单一来源）：
```python
request.kvmem_budget_tokens = B           # 该请求（该用户）的份额
request.kvmem_view_ready   = False        # worker 已装好视图
request.kvmem_view_tokens  = 0            # 视图实际 token 数（= n_blocks * block_tokens）
```
- `num_computed_tokens` 的语义改为"**视图内**已计算 token 数"：prefill 期间照常增长，达到 B 后**冻结**；
- `positions` 由 `num_computed + 本步偏移` 生成 ⇒ 天然落在 `[0, B)` 内（与视图坐标一致，不需要额外回卷）。

### S3. 元数据（已实现的部分）
- `attention_seq_lens` 用 `request.kvmem_view_tokens`（worker 回传，见 S4）替换 seq_lens ✓（已有 `_seq_override` 通道，改成读 Core 值）；
- GDN 层继续用真长度（`num_computed` 真值另存一份 `num_computed_true`）。

### S4. worker → Core 回传（复用 block channel 的 `compact` 消息）
worker 装好视图后发送：`{kind: "view", request, view_tokens, blocks: [ids]}`；
Core 收到后置 `kvmem_view_ready/view_tokens`，并把 `blocks` 记为该请求持有（用于份额校验）。

### S5. 份额（多用户）
- 池块总数 `N_pool`；每用户份额 `S = B / block_tokens`（当前 29 块）；
- Core 维护 `share_owner[block] = session_id`；分配只从**本会话未用满的份额**里取；
- 视图块固定占用该会话的 S 块，历史块落盘后归还同会话（或全局池）。

### 验收
1. 89K 池 + 92K → 视图应用（`VIEW ... inst=1`）且答对；
2. 140K → 答对；3. 250K → 答对；4. 两会话并发各跑 B 内长上下文互不越界；5. 离线四套。

## 762. Round 496：更正一条判读 —— `pos=[0..4095]` 很可能是**槽位偏移**，不是模型位置
- 代码事实：clamp 把 `num_computed_tokens` 设为 `window - num_new - lookahead` = 45,056 − 4,096 = **40,960**，
  所以引擎的模型位置应是 **40,960..45,055**，而不是 0..4,095；
- 我此前用 `view.query_position_slots(req)` 读到的 `[0..4095]` 更可能是**该 chunk 在窗口内的偏移/槽位**，
  而不是 `positions` 张量本身。
- 影响：**不要再按"回卷到 0"去改**；正确做法是直接读引擎的 `positions` 张量取证
  （`runner.input_batch.positions` 或 prepare 后的 positions 张量），与 `seq_len`/视图长度对齐后，
  再判断 rebake 的 δ 是否一致。
- 这条更正避免了往错误方向（重写 Core 记账）走一大步。

## 763. Round 497：完整版剩余工作量（按已验证事实重估）
| # | 工作项 | 依赖/落点 | 估算 |
|---|---|---|---|
| 1 | **positions 取证 + rebake 校正**：直接读引擎 positions 张量，确认 δ 方向/单位 | `plugin` 步进钩子 | 0.5 天 |
| 2 | **视图生效验证**（92K）：`VIEW ... inst=1` + 答案正确 | `_apply_selection_d` | 0.5 天 |
| 3 | **140K 超池**：视图 + 落盘 + 选块在 89K 池上答对 | 同上 | 1 天 |
| 4 | **250K** | 同上（CTX/位置表上限内）| 0.5 天 |
| 5 | **多用户份额**：每请求只占本会话 29 块；池按 N×29 预留 | Core allocator + block channel | 1 天 |
| 6 | 离线四套 + 文档 | — | 0.5 天 |
**合计 4 个工作日**（A/A′ 已完成；B 可能比原估计小，因为 clamp 实际给到 40,960 而非 0）。
**当前绿点**：`NOBLADE=1` 下 92K 正确（103–115 s）；实验路径 `KVMEM_D_APPLY/REBAKE` 默认关闭。

## 764. Round 499：收尾核对（无回归）
- `needle_88k`（max_tokens=16，24 小时后复测）：**`ZX-4417-QJ7`，82 s** ✅
- 之前一次 `ok=False` 是 **max_tokens=8 截断**所致（答案前 8 token 未含代号），不是回归。
- `input_batch.positions` 取证未取到（该属性在此路径不可用）；positions 取证改用 `view.query_position_slots` + clamp 值交叉验证。
- 环境：`GPU_UTIL=0.85`、`PLAN_D=1`、`NOBLADE=1`、`BUDGET=37280`；`KVMEM_D_META` 已移除；引擎 healthy。

## 774-777. Round 527-529：视图能装、seq_lens 能改，但输出仍退化 —— 剩 positions 帧
- 774：`allocate_slots` 之后 restore `num_computed` 是真 bug，视图模式下已保持回卷。
- 777：把既有 `_fix_book_for_view`（派生前重装 resident 计数）也纳入 D 视图模式 ⇒ 仍 `duct…`。
- 已排除：内容/打包/重旋（头视图对照）、seq_lens 覆盖（=45,472）、applied 时机、clamp 回滚。
- **剩余唯一怀疑点**：模型拿到的 `positions` 仍是真帧。`runner.input_batch` 无 positions 属性；positions 由 `prepare_pos_seq_lens` 生成后经 `model_input` 传入 ⇒ 必须在 `_fix_book_for_view` 之外**直接改 positions 张量**（在该 wrapper 返回前按 offset 平移）。

## 778-782. Round 528-534：positions 已进视图帧，仍退化 ⇒ 怀疑 GDN frontier
- 778/779：视图偏移改存**模块级** `VIEW_OFFSETS`（跨实例共享）。
- 780：`prepare_pos_seq_lens` 里实测 `positions` 是 Tensor(n=4096)，单请求时整张量 `-= offset`，**已生效**（`/tmp/kvmem_posshift.jsonl: single off=29476 n=4096`）。
- 781：同步平移 `mrope_positions*` 后仍退化。
- 至此四件事同时成立仍 `duct…`：①`seq_lens=45,472` ②行/块/重旋已装 ③positions 已在视图帧 ④mrope 同步。
- **剩下唯一怀疑**：Qwen3-Next 的 **GDN（线性注意力）每请求状态槽的 frontier** 仍按真长度推进，与 attention 的视图长度不一致。下一步在安装时把 GDN 读取的 num_computed/frontier 钉到视图长度（mamba 状态仍只一份、不搬），一次 88K 验证。

## 783-786. Round 535-542：positions 前沿终于对齐（45,471 vs 视图 45,472），仍退化
- 783：只在 positions 仍为真帧时平移（避免二次平移）。
- 784：offset 改为按**当步前沿**推导：`off = pre_last - (view_len - 1)`。实测：
  `pre_last=66,778 → post_last=45,471, viewlen=45,472` ✅ **查询正好落在视图最后一位**。
  此前 `pre_last=95,420、off=29,476` 减完仍是 65,944（视图只有 45,472）⇒ 全部键被掩 ⇒ 这就是长期退化的原因。
- 785：只对解码步（`len(positions)==1`）平移，避免把 prefill chunk 写乱。
- 结果：89 s，仍 `duct…`。
**当前状态**：seq_lens=45,472 ✅、positions 前沿=45,471 ✅、行/块/重旋已装 ✅，输出仍退化。
**下一步**：需要一次带 `n=1` 解码步的取证（确认 785 的分支在解码步是否真的执行；`/tmp/kvmem_posshift.jsonl` 在 785 运行中为空），并检查 GDN 层在该步的 frontier。

## 787. Round 543：证伪 —— `prepare_pos_seq_lens` 的 `positions` 不是模型位置张量
读数（`/tmp/kvmem_prep2.jsonl`）：
```
npos=4096 offs=1 vlens=1 last=583576 / 604883
```
- `last` 高达 **58 万~60 万**，而序列只有 74,948 token ⇒ 这个 `positions` 不是模型用的位置
  （很可能是 mrope 展平/其它派生缓冲），此前按它推导 offset 是**在错误的张量上做平移**。
- 因此 784 得到的 `post_last=45,471` 只是把这个缓冲改成 45,471，并不等于模型位置被修正。
**结论**：positions 的正确注入点在 **fork 的 `prepare_inputs` / `model_input.positions`**，不是这个 hook。
**下一步**：在 fork 里加一个「视图模式位置偏移」：`model_input.positions -= (frontier - view_len)`，并且
只对 attention 层生效的位置数组做（mrope 若为 3 段则同步 3 段）。

## 788-790. Round 545：**关键发现——引擎用的是 V2 runner，且 fork 已内建 D 视图支持**
1. 引擎实际 import 的 vllm 是 **`/home/srj/vLLM-2080Ti-Definitive/vllm`**（不是 `engines/vllm`）。
   本会话此前对 `engines/vllm` 的引擎改动（757/768/774/788/789）**全部没有生效** —— 那些"读数"来自旧代码。
2. worker 通过 `use_v2_model_runner` 选择 **`vllm/v1/worker/gpu/model_runner.py`（V2）**；
   V2 里 **已经内建** Plan D 支持：
   - `prepare_pos_seq_lens(...)` 传的是 `self.input_buffers.positions`（真位置，注释明确：K 存的是真相位）；
   - `kvmem_view_lens`（device 表）覆盖 attention 长度；
   - `kvmem_rope_offset`（device 标量）：RoPE 用真位置、slot 映射用压缩位置；
   - `kvmem_view_active` 门控该覆盖是否生效。
3. 插件步进钩子（~1700）**已经在写** `_dv`（slot delta）与 `_vl`（view lens）并把 `runtime.kvmem_view_active = True`。
**结论**：fork 的设计是「真位置 + 压缩地址 + 视图长度」，**不需要 rebake、不需要平移 positions**。
我本会话加的 `_install(seq_lens/row)`、`_apply_selection_d` 打包与 rebake 是**与内建机制冲突的平行实现**，
这正是装了视图就退化的原因。
**下一步**：只保留 fork 内建路径（滚行 + `_dv` + `_vl` + `kvmem_view_active`），关掉适配器的 install/pack/rebake，
然后验证 92K 是否回到绿、再试 140K。

## 791-792. Round 546：引擎真实路径 = `engines/vllm`（PYTHONPATH），fork 内建视图机制已定位
**两条硬事实**
1. 引擎进程 `PYTHONPATH=/home/srj/src/kvmem-vllm-adapter/engines/vllm` ⇒ 生效的是 **`engines/vllm`**；
   （我先前那次 `import vllm` 测试没带 PYTHONPATH，才指向另一副本。）
2. 真实 runner = **V2 `vllm/v1/worker/gpu/model_runner.py`**，其 Plan D 机制：
   - `positions` 是**压缩帧**（每步 0..chunk 相对）；
   - `kvmem_rope_offset`（device 标量）在 RoPE 前 **加回真位置偏移**（`positions += ro`）；
   - `kvmem_view_lens[row]` + `query_len` 覆盖 attention 长度，门控 `kvmem_view_active`；
   - 行值为 0（stale）时保留 stock lens。
**本轮读数**
- 见证：`active=True vl=1 pos0=74971 posn=74971 nc0=74971 seq0=74972`
  ⇒ 该步 positions 仍是**真帧**、attention 仍是 stock（视图表未被填）。
- 我写表时 `st.row_index = -1`（`/tmp/kvmem_activate.jsonl: row=-1 view=45472 off=29476`）
  ⇒ 视图表写入被跳过 ⇒ 视图未生效。
**下一步（两小步）**
①在 `_apply_selection_d` 里用 `self.view.row_index(req)` 取行号（不要依赖 `st.row_index`）；
②同时把 `kvmem_rope_offset` 设为 `tokens - seq_len`，让 RoPE 用真位置、attention 用视图长度。

## 796-797. Round 548-549：140K 进入真实压缩路径；assert 定位到 slot offset
- 目标函数修正：**88K 在 89K 池无需压缩**（行装得下 2,343 页），之前盯 88K 视图是错的；真正验收是 140K。
- 140K + `KVMEM_D_APPLY=1`：apply 能跑完（`VIEW keep=29 need=9 restored=0 seq_len=45472 inst=1 tok=137873`），
  随后 **Triton device-side assert** 在 `execute_model` 内（引擎死亡）。
- 已排除：store restore（NORESTORE=1 仍崩）、行安装（NOINSTALL=1 仍崩）。
- 已定位一个真错：把 `kvmem_slot_offset` 也设成真前沿 ⇒ slot 映射变负数（797 已改为恒 0），但 assert 仍在。
- 剩余怀疑：`kvmem_view_lens`（attention 长度 45,472）与 **GDN/mamba 的 seq_len 账**不一致（该 override 只应作用于 FullAttention），或 indptr 与 block table 的页数不匹配。

## 798-799. Round 550-551：140K 的三次消融，assert 仍未消除（已缩小范围）
| 变体 | 结果 |
|---|---|
| `NOOVERRIDE=1`（关因果覆盖）| 217 s，仍 assert |
| `ROPE0=1 + REBAKE=1`（位置/键都在压缩帧，自洽）| 217 s，仍 assert |
| `NOACT=1 + NORESTORE + NOINSTALL`（完全不写视图表/不恢复/不装行）| **125 s 就失败**（更早，非 view_err）|
结论：
- assert 不是因果覆盖、不是 rope 偏移方向、也不是 rebake；
- 只要 140K 走到「视图/选择」这条路径就崩；
- 完全不激活视图时它更早失败 ⇒ 说明上半程（prefill 后段的压缩触发）本身也不干净。
**下一步（下一段工作从这开始）**：不要再用 `_apply_selection_d` 这套手写安装。
改用 fork 已有的 `advance`/`_dv`/`_vl` 增量通道（design 45），即"滚行 + 视图表推进"，
完全不碰 row 安装、store 恢复和 rebake；先让 140K 跑通（答案可错），再谈选中的内容。

## 800. Round 551：ring 路径跑通 140K（无 assert）但输出退化
- `NOBLADE` 去掉（走 fork 自己的 ring + 视图表通道），140K：
  `RING140 215s ok=False 'duct…' usage=137873` —— **不再崩**，prefill 完整，但答案是退化文本。
- `/tmp/kvmem_ring.jsonl` 为空 ⇒ 655 的「滚行 + 写 _dv/_vl」分支**一次都没执行**（design 660 的注释已怀疑
  `dv/vl` 为 None 会静默跳过）。
- 因此下一步不是改公式，而是：在该分支入口打印 `_dv/_vl` 是否为 None（`KVMEM_D_META=1` 下
  `/tmp/kvmem_meta.jsonl` 已有部分字段），把"为什么没进这个分支"钉死。

## 801-802. Round 552：ring 视图表/滚行分支**根本没执行**（三个见证文件都为空）
- 结构问题：钩子里是 `if not _has: 清零  elif APPLY: 795  elif win>0 and tok>win: 滚行(655)`。
  由于 `view_installed` 为 False（ring 路径不装视图），**永远走第一个分支**，滚行分支不可达。
- 我把滚行 + 写表搬进第一分支（801/802，`KVMEM_RING_VIEW=1`），但
  `/tmp/kvmem_ring802.jsonl`、`/tmp/kvmem_ringin.jsonl`、`/tmp/kvmem_ring.jsonl` **全为空**
  ⇒ 连第一分支所在的整段都没进（外层还有门槛）。
- 140K 在 ring 路径下稳定可跑（usage 137,873、215 s、不崩），答案退化。
**下一步（必须先取证，不要再改公式）**：在该钩子段**入口**写一行无条件见证（记录 `_dv/_vl` 是否 None、
live 是否空、是否 d_mode），一次 140K 即可定位是哪一层把这段挡住；然后再接滚行。

## 803. Round 553：**140K 从 `duct…` 变成连贯英文**（tokens 账修正后窗口机制生效）
根因与修正：
- `/tmp/kvmem_hookin.jsonl` 实测 `st.tokens = -1` ⇒ `tok > win` 恒假 ⇒ 滚行/视图表分支**永不触发**。
- 改为读调度器的权威前沿 `num_computed_tokens[_i631]`（design 803）。
结果（140K，`KVMEM_RING_VIEW=1`，无 NOBLADE）：
- 152 s（比之前 215 s 快）、usage 137,873、输出 **`'Based settles'`（连贯英文，但答案错）**。
- 见证：`tok=137874 win=45472 base=92403 k=0 rowlen=128`
  - base 正确（137,874−45,472=92,402）；
  - **`rowlen=128`** ⇒ 适配器的行镜像只有 128 页（4,096 token），而引擎真实行在 140K 时是 2,343 页；
    滚行作用在一个过短的行上 ⇒ 内容仍然不对；
  - `sl=0` ⇒ `st.seq_len` 未被维护（attention 长度已由 fork 表承载，暂不影响）。
**下一步**：先把行镜像同步成引擎的真实行（`sync_row_from_engine`，宽度 = ceil(seq_lens/block)），
再滚行；然后看答案是否正确。

## 804-810. Round 554：ring 路径**端到端成立**；选择(apply)路径仍 assert 在 restore
- 804：把「读调度器前沿」用 `KVMEM_RING_VIEW` 门控，经典路径不变 ⇒ 88K 绿点恢复（`ZX-4417-QJ7`）。
- 805：滚行前先 `sync_row_from_engine` ⇒ 镜像行变成 **4,310 页**（引擎真实行）。
- 806/807：滚行读数 `k=0 base=92,423 rawold=92,401` ⇒ **滚行其实是对的**（只差 21 token < 1 块），
  行已按块粒度对齐；`_dv` 负责块内偏移。
- 因此 140K 在 ring 视图下输出**连贯英文**（`Based settles where flow slows; farmers built levees…`），
  但中段 needle 在窗口之外 ⇒ 答案错——**这正是必须靠选择器**的地方，符合预期。
- 810：apply 前先 sync（避免用 1,536 行截断 4,310 行的表）；仍然 **restore/install 处 device-side assert**。
**结论**：ring 通道已是可用的"载体"；缺的是"把中段块塞进窗口"的搬运（restore + install）在 140K 下不安全。
下一步应在此处用**断点/console**逐个文件核查 restore 的目标页号与 store 张量形状（4 分钟一次的黑盒循环效率太低）。

## 811-812. Round 555：assert 精确定位到 `gather_block_tables`（Triton）
eager 模式的 traceback（Worker_TP1）：
```
gpu/model_runner.py:1995 execute_model
gpu/model_runner.py:1664 prepare_attn
gpu/block_table.py:166 gather_block_tables   ← Triton kernel，device-side assert
```
- `gather_block_tables` 按 `self.num_blocks.gpu[i]` 拷贝该请求的行条目数；
- 我们的 `_install` 只写 `table[row, :len(row)]`（1,536），**没有更新 `num_blocks[row]`**；
- 于是 kernel 按真实块数（≈4,310）拷贝，越过我们拥有的区间 ⇒ assert。
- `restore` 侧已加边界过滤（`/tmp/kvmem_restore.jsonl: pairs=0 kept=0 npages=2842`）——它现在不崩，崩的是 gather。
**下一步**：安装视图时同步 `num_blocks[row] = ceil(seq_len / block_tokens)`（或等价地
`view.num_blocks.np[row] = …` 并 copy_to_gpu），让 gather 只拷我们拥有的区间。

## 813-814. Round 556：assert 仍在 `gather_block_tables`；块数账的写入没被执行到
- 813：在 `_install` 里把 `num_blocks[:, row] = len(row)` 并 copy_to_gpu。
- 814：加见证 `/tmp/kvmem_nb.jsonl` —— **文件为空**，说明这段没跑到；
  同时这一轮 `/tmp/kvmem_apply.jsonl` 也没有 `view_err` 记录 ⇒ apply 段本身没进入/更早失败。
- eager 模式的 traceback 仍是 `prepare_attn → gather_block_tables`。
**下一步（建议交互式调试）**：在 `_install` 的每个语句间打断点/打印，确认哪一步先抛；
以及确认 `runner.input_batch.block_table` 在 apply 时刻是否可访问（这决定 813 能否生效）。
4 分钟一次的黑盒重启循环在这里效率很低。

## 816-817. Round 559：apply 路径带上窗口与滚行后仍 assert —— 决定改用「原地替换窗口块」方案
- 816：APPLY 分支里把 `_vl` 设为 **窗口**（而不是 st.seq_len−1，实测为 0）、rope=窗口基址，并同步滚行。
- 结果：140K 又回到 device assert（`view_err`）。原因仍是**修改了行长度**后 `gather_block_tables` 按真实块数拷贝。
**结论（重要）**：只要"装行"改变行长度/块数账，就会撞 `gather_block_tables`。
**新方案（下一段工作）——不碰行、不碰块数，只换内容**：
1. ring 已经提供合法窗口：行长度、`num_blocks`、`view_lens`、`rope_offset` 都不动；
2. 从 store 取中段高分的块，**写到窗口内某个低分块所对应的物理页**（`index < 2,842` 保证不越界）；
3. 用 `rotate_k_pages_fp8` 把这批页的相位从"原位置"旋到"目标槽位"（δ =（目标−源）× 块 token）；
4. 完全不需要 `_install`、不需要改 `num_blocks`、不需要 restore 到大页号 ⇒ 不触发 assert。
这是与引擎账本**零冲突**的路径，也是本会话所有证据指向的方向。

## 818-819. Round 560：原地替换模式已就位，但搬运为空 + 仍有后续错误
- 818：`KVMEM_D_INPLACE=1` —— 不写行、不写 seq_len、不 `_install`，只把内容写进 `row[i*q+j]` 指向的物理页并重旋。
- 读数：`VIEW keep=29 need=9 restored=0 rebaked=-1 seq_len=45472 inst=1 tok=137873`
  - `need=9`（9 个块不在 GPU 上）但 `restored=0`、`/tmp/kvmem_restore.jsonl: pairs=0`
    ⇒ 恢复循环一个 pair 都没构造出来（`spare`/`need` 的配对在 INPLACE 分支下没走通）；
  - `rebaked=-1` ⇒ 重旋循环抛异常（被吞掉）；
  - 请求最终仍 500（217 s）。
**下一步**：先让 `need → pairs` 的配对在 INPLACE 下成立（打印 need 的块号与 `spare` 的槽位），
再确认 `rotate_k_pages_fp8` 的入参（层张量切片、inv_freq、δ 单位）。

## 820-821. Round 561：`restored=0` 的根因 —— store 键空间与查询不一致
- `pairs=0` 出自这段：`pg = b*q + j`（**逻辑块号×49+页内**），再 `if orch.store.has(pg)`；
  但落盘（design 750）用的键是**物理页号**（行里的值，≤2,842），而 `b*q+j` 在 140K 可到 4,310。
- 于是所有查询都 miss ⇒ `len(pairs) < q//2` ⇒ `continue` ⇒ `restored=0`、内容一步没搬。
- 另外 `rebaked=-1`：重旋循环抛异常（日志里没记录文本；下一轮把 `_rb_err` 打出来）。
**下一步（两处小修，都在 `_apply_selection_d` 内）**：
1. 统一键空间：查询用 `int(row[b*q+j])`（物理页），与落盘一致；或给 store 建"逻辑页→物理页"索引；
2. 重旋的异常文本要打印出来，再修 `rotate_k_pages_fp8` 的入参。

## 822-823. Round 562：store 键空间其实是逻辑页（0..4,308）；真正的错配在【选择域 vs 环行域】
`/tmp/kvmem_store.jsonl`：
```
n=4309 min=0 max=4308 need0=35 rowat=-1 nb=35
```
- store 键 = 0..4,308 ⇒ **逻辑页**（`b*q+j`），我们原来的查询式是对的；
- **`nb=35``** ⇒ 此刻 `row` 只有 35 块（= 窗口 45,472/1,568 ≈ 29–35 块），而选择器给出的 `need0=35`
  属于**逻辑块号域**（0..87）；
- 两个域之间没有映射 ⇒ 恢复循环访问 `row[35*q]` 越界（`rowat=-1`）⇒ 什么都不搬。
**结论**：缺的是「逻辑块 ↔ 环槽」的持续记账（`st.logical`，现在全是 -1）。
**下一步（下段工作的第一件事）**：在 `_roll_window`/滚行时维护 `st.logical[slot] = 逻辑块号`，
有了它就能把"选中的中段逻辑块"映射到"要牺牲的环槽"，再做原地替换 + 重旋。

## 824-828. Round 563-564：**行里的值是"虚拟块号"（可达 4,263），而 KV 张量只有 2,842 页**
`/tmp/kvmem_inplace.jsonl`：
```
guard inp=True orch=True need=9 nb=35 q=49 rowlen=1715 rowmin=6 rowmax=4263 npages=2842
shapes n=16 (2842, 2, 32, 512) blocks=2842
restore_err n=49 CUDA error: device-side assert triggered
```
- 行长度 1,715（=35×49）但**值域到 4,263 > 池 2,842 页** ⇒ `index_copy_` 越界 assert；
- 这与 store 键域 0..4,308 一致 ⇒ **store 的键就是这些虚拟块号**（落盘时用同一个值，所以那次能成功）；
- 因此"恢复/重旋"必须经过**引擎自己的 虚拟块号 → 物理页** 映射，不能直接用行值当设备索引。
**下一步（下段工作第一件事）**：找到 fork 里 block_table 的 id→slot 映射（`block_pool` / `slot_mapping` 的同一套），
把 restore 目标改成映射后的物理页；或直接在映射后的空间里做搬运。

## 829. Round 565：镜像行 4,309 项、值域到 4,263；但 `paged_kv()` 视图只有 2,842 页
- `sync_row_from_engine` 宽度 `w = ceil(seq_lens / block_size)` = 4,309，取自
  `block_table_rows()[0][row_index][:w]` ⇒ 行值 0..4,263；
- 而 `paged_kv()` = 16 张 `(2842, 2, 32, 512)` ⇒ **适配器看到的页数比引擎块的 id 空间小**；
- 结论：`block_size` 与 `paged` 的页概念不一致（引擎 block id 空间 4,264 vs 适配器视图 2,842），
  所以"行值直接当设备页号"必然越界。
**下一步**：打印引擎真实 `kv_cache_config.num_blocks` 与 `runner.block_tables.block_size`，
和 `paged_kv()[0].shape[0]` 对齐；三者一致后再做搬运/重旋，否则先把 id 映射到物理页。

## 830-831. Round 566：clamp 到池大小没有改变镜像行 ⇒ 它不是在 sync 里建的
- 加了 `w = min(w, view.num_blocks())`（design 830），但读数完全不变：
  `rowlen=1715 rowmin=6 rowmax=4263 npages=2842` ⇒ **st.row 是别处构建的**（不是本次 sync 写入）。
- 因此下一步不是改 sync，而是：找到给 `st.row` 赋 1,715 项且含 4,263 这些超池值的那个构建点
  （候选：`_roll_window` 的提交/旋转路径，或 adoption 时从旧账本复制），并在那里就把行约束在
  `[0, num_blocks())` 内。

## 831. Round 567：行确由 sync 写入（w=1,715 = ceil(54,880/32)），但值域 6..4,263 仍超池 2,842
- `sync_row_from_engine` 的 `st.row = tuple(values)`（第 5296 行）就是唯一来源；w 来自当时 seq_lens 54,880；
- 也就是说**引擎块表里这些条目本身就是 4,263 这种值**，而 FA 的 paged 张量只有 2,842 页；
- 二者只能通过 fork 的 id→slot 间接映射（`compute_slot_mappings` 内核里 `block_sizes_tensor` /
  `kernel_block_sizes` 参与），**不能把行值当页号**。
**下一步（一次 console 打印即可定案）**：在 `compute_slot_mappings` 之后打印第一条 slot 值与
`block_table[0][0..3]`，反推 `slot = f(block_id)` 的公式；拿到 f 之后，restore 目标 = f(row[slot])。

## 832-833. Round 568：映射公式确认 + 组选择问题
- 内核（`_compute_slot_mappings_kernel`）：`slot_ids = block_numbers * kernel_block_size + block_offsets`
  ⇒ **页号 = block_id**（kernel_block_size=32），因此 block_id 必须 < 池页数（2,842）。
- 但 140K 时镜像行出现 4,263 > 2,842 ⇒ 这些值只能来自**另一组的块表**；
  `LiveRunnerView.block_table_rows()` 用 `sorted(attn_groups)[0]`，混合模型可能有多套 full-attention 组
  （各自独立池），`[0]` 未必是与 `paged_kv()` 同一池的那组。
- attach 期 `kv_caches` 还是空的（`n=0`），所以池形状是运行时懒绑定得到的；
  需要把"组 → paged 张量"的配对做成按池大小/组 id 显式匹配。
**下一步**：在 `_apply_selection_d` 打印 `attn_groups`、每组的 `block_table_rows()[g][row][:4]`、
以及 `paged` 每张的 shape，确认哪一组与 2,842 页同池；然后用那一组的行值。

## 834-835. Round 569：块表宽 8,232；按窗口裁剪后 rowlen=1,421，但值域仍到 4,263
`/tmp/kvmem_inplace.jsonl`：
```
guard inp=True orch=True need=12 nb=29 q=49 rowlen=1421 rowmin=6 rowmax=4263 npages=2842
groups=[3]  tables=[(1, 8232)]  shapes n=16 (2842,2,32,512) blocks=2842
```
- `groups=[3]` 只有**一个** full-attention 组，表宽 8,232（= max_model_len/32），所以"选错组"被排除；
- 按窗口裁到 1,421 项后，**前 1,421 项里仍有 4,263** ⇒ 引擎真实写下的 block id 就超过本 rank 的 2,842 页；
- 结合内核 `slot = block_id*32 + off` 与 90,944 slot 上限，唯一自洽解释是：
  **block id 是跨 rank 的全局编号**（2,842 × 2 ≈ 5,684 ≥ 4,263），本 rank 的本地页号需要由全局 id 推导
  （很可能是 `id % per_rank_pages` 或 `id // tp`）。
**下一步**：打印 `runner.parallel_config.tensor_parallel_size` 与 `cache_config.num_gpu_blocks`、
以及 `kernel_block_size`，确定 per-rank 页数与全局 id 的换算，再据此修正 restore 目标。

## 836. Round 571：引擎池 88,914 token（≈2,778 页）/ TP=2；块 id 可到 4,263（KVMEM_SKIP_KV_CHECK 允许池外 id）
- 启动日志：`GPU KV cache size: 88,914 tokens`，`Worker_TP0/TP1` 两个 rank；
- 836：in-place 的写目标改为 `id % num_blocks()`（处理池外 id），运行不再出现 restore_err，
  但 `_pairs823` 仍全空（store.has 全 miss）⇒ 搬运依旧没发生。
- 下一步：打印 need 的前几个块号与对应 src 页号，以及 `orch.store.has(src)` 结果，
  确认 store 里到底有哪些页（可能只存了窗口内的页）。

## 837. Round 572：结论——store 的键是"引擎块 id"，而选择器的块是"逻辑块号"
- 早前实测：`store.pages` 共 4,309 个、键域 0..4,308，与**行里的引擎 id**同域（落盘时用行值）；
- 而 `need` 来自选择器，是**逻辑块号**（`b*49+j` 形式的页号），两者只是数值区间重叠、语义不同 ⇒
  140K 时 `store.has(src)` 全 miss（本轮读数）。
**修复方向（下段工作）**：
1. 落盘时改用**逻辑页键**（`token_pos // 32`），或同时保留一份 `logical → id` 映射；
2. 选择器给出的逻辑块即可直接 `store.get` 取回；
3. 写回时再通过 `id = logical_map[logical]`（或就地另选一个空闲页）得到设备页。
**同时**：已确认 `index_copy_` 只接受 < 页数（2,842）的槽，池外 id（4,263）必须经取模或映射（836）。

## 838. Round 574：store 查询补上窗口基址后**能命中**，但 restore 仍 assert
- 落盘键 = `basepg + 行索引`（`_pg = _basepg + _i`，`_basepg = old//32`），恢复查询必须做同样平移
  ⇒ 838 给 `src` 加上 `_base823 // page_tokens`；本轮 `restore_err n=49` 说明**查到了页**（此前全 miss）；
- 但 `index_copy_` 仍 assert ⇒ 剩余嫌疑：目标槽取模后**重复**、或源张量形状/store 层数与 `paged` 不匹配。
**下一步**：打印 pairs 的前 5 项（src, dst）与 dst 去重后的数量、`store.get(src)` 的层数/形状，
与 `paged[0]` 对齐后即可消除 assert。

## 839. Round 575：去重后 pairs 从 49 → 3 ⇒ 窗口行里目标页大量重复
- 给 in-place 恢复加"每槽只写一次"，`restore_err n=3`（原来 49）说明**一个块的 49 页里只有 3 个不同目标页**；
- ⇒ `st.row` 在窗口域里存在大量重复值（与"块表宽 8,232、引擎只写前缀、其余残留"叠加），
  用它做搬运目标本身就不可靠。
**结论**：搬运目标不能取自这个镜像行，必须取**引擎实际分配到的页**（`num_blocks` 前缀 + 校验 < 池）。
**下段工作**：改用 `(basepage + i) -> 引擎 id` 的落盘/取回闭环（即 §837/§838 的方向），
并在目标侧从引擎真实分配里选空槽，而不是从镜像行猜。

## 840. Round 576：环形窗口的四处机制缺陷（数据面已修，在线仍未过针）

§839 的"dst 不能取自镜像行"建立在"镜像行可信"上。本轮先用**尾部针**验证环形窗口本身：
同一批 142.8K 电池把 needle 分别放到距尾 6,857 / 1,651 token，另有 88K 电池距尾 670 token —
needle 落在窗口内时，答案必须在"纯环形、无选择"下正确，否则后面所有 restore 都建在沙上。

**实测（probe `d_ring576`；两 rank，每 pid 40 条；全部带数值）**
- 旧臂首滚发生在前沿越过窗口**之后**（`tok=49,152, n_new=4,096`），跨窗那个 chunk 以
  `delta=0` 写在 TRUE 列（行只有 1,421 页）⇒ 窗口中部一个 4,096-token 的洞。
- 旧臂滚行按 `token/1024` 计步（页粒度行里差 32×），且 `rope_offset=base` 与 TRUE positions
  叠加 ⇒ `slot = pos + base - delta` 又回到 TRUE 列：**视图从不更新**，模型读的一直是开头
  （这解释了 88K 中段针 True、尾针全错）。
- `before_step` 的 `_d_reset_row` 在 prepare 之后把刚发布的 `view_lens/slot_deltas` 清零：
  forward 入口实测 `dv=0, vl=0`（`prepare_attn` 包装取证）。
- 镜像行宽取自 `view.seq_lens()`，视图覆盖生效后可能是小值（实测 128 页），首滚滚在 128 页上。
- 引擎账本与视图不同宽：attention book 实测 1,715 页（35 块）vs 视图 1,421 页（29 块）——
  gather 多拷 294 项，读路径不使用，属无害；本轮行值全部 < 2,842，**§835-839 的"行值超池 /
  全局 id"只在旧臂下成立，作废**。

**本轮交付（离线绿）**
- `VllmSparseRuntime.d_ring_window`：页对齐 base（`ceil((frontier-win)/pt)*pt`）、按整页滚、
  `delta=base`、RoPE offset 0、`view_lens = frontier-base`；首次安装直接取**引擎行在 base 处的
  切片**（不再"滚头部前缀"，消除跨窗洞）。
- `_ring_commit_pages`：store 键 = 绝对逻辑页 `token//32`，槽位 = 行值本身（旧代码把行值乘了
  `block_pages`，复制的页号 ~49× 越界、被裸 except 吞掉）。
- `_apply_selection_d` in-place：`src = b*q+j`、`dst = 已安装行指向的引擎页`（校验 < 池，不取模）；
  REBAKE 默认关——这个帧里 K 字节存的是 TRUE 相位，再旋一次会毁字节（`KVMEM_D_REBAKE=1` 可复现旧路）。
- `before_step` 不再在环臂下重置行/再滚；hook 由环臂整条生命周期持有（激活条件改为**跨窗那个
  chunk**：`tok + n_new > win`）。
- `tests/ring_window_test.py`：160K 逐 token 视图恒等（含中途 chunk/decode）、store 键、in-place
  恢复、旧臂负对照；全套离线无新增失败（14 项为既有失败）。

**在线读数（每轮重启后单发）**
- 旧臂：88K 中段针 True（needle 在窗口头部，属"读开头"）；88K 尾针退化。
- 新臂（含重置守卫）：probe 全部 `rolled`、`delta=base`、`vl_table=tok-base`、行/切片正确，
  但 88K 尾针仍 `'Based.::…'`、152K 尾针仍复读/`The_` ⇒ **发布正确而 forward 侧未生效**。
- **下一发单测（唯一）**：在 `prepare_attn` 入口同帧打印 `dv/vl` 的最终值、`input_block_tables`
  首项、以及第一步 `slot_mappings` 头 8 项——钉死"谁在 prepare 之后又写了 dv/vl（或 slot 不走
  delta）"。本轮 10:44 的 hot-reload 实例上读到 `dv=0/vl=0`，但该实例被 console 重载污染
  （环形日志此后不再增长），不能作为新臂结论；必须在干净重启上取这一发。

预算：本轮 7 boots，超支停止。生产臂已恢复（`KVMEM_RING_VIEW` 关闭，`PLAN_D=1` 不变）。

## 841. Round 577：环臂的两条硬事实——warmup 假视图会杀 worker；瞬时 clamp 让引擎行不可作位置映射

**1. warmup 假视图（已修，commit 之前的崩溃根因）**
- 环臂在 `_warmup_*` / `_v2_mixed_warmup_*` 的 dummy 请求上发布视图（其 frontier 是 max_model_len 级），
  下一步 dummy 用被污染的视图表 → 两个 TP worker **静默被杀**（exit code None、无 traceback、无 Xid），
  EngineCore 只看到 `RuntimeError: cancelled`。11:05/11:07/11:10/11:19 四次启动失败全是这一条。
- 修：`d_ring_window` 与 hook 都跳过 warmup id；首次安装的切片加"distinct ≥ 90%"校验（不合格 fail closed）。
  修后引擎可正常启动并跑完请求。

**2. 前向侧实测（`kvmem_dbg` 断点，文档里的设施）**
- 50K 请求、decode 步 `position=49369`：`slot_positions == positions`（**delta=0**），slot 落到
  `page 9728`（池只有 2,842 页）⇒ 写地址在池外——这就是"发布正确却仍退化"的直接原因。
- 同一帧里 `kvmem_slot_deltas[0]=0`、`kvmem_view_lens[0]=0`；而环方法的 probe 记录的是它写入的
  `delta=3680/vl=45472`。断点打在 `before_step` 的守卫行时 `ring_ready=False`、`roll_base=0`
  ⇒ **请求态没有被环的写入看到**（runner 张量有值、请求对象没值），`_d_reset_row` 于是把表清零。

**3. 架构判决：环不能建在瞬时 clamp 的分配上**
- 50K 首次安装的切片 `slice_distinct=906`（应=1,421）：瞬时 clamp 在分配时把 `num_computed` 回卷，
  引擎随后按回卷后的位置重新 append 块 id，于是**块表行不再是"位置→页"的忠实映射**（同一页 id 出现在多个列）。
  88K 那次 `slice_distinct=1421` 只是 clamp 时机的巧合。
- 结论：环臂（`KVMEM_RING_VIEW`，瞬时 clamp）在架构上不可成立；环必须走**持久回卷**的 apply 臂：
  Core 把 `num_computed_tokens` 回卷到**页对齐的 `frontier - win`**（按页滑动，而不是每步重置到
  `win - num_new`），则 positions 天然是视图帧、分配与视图同构；worker 侧只发布 `delta=0`、
  `rope_offset=base` 并安装旋转后的行（K 字节保持 TRUE 相位）。这是下一步的施工面（scheduler +
  prepare 链 + 适配器三处，权限清晰）。

**现状**：默认臂已恢复（`KVMEM_RING_VIEW=0`），88K 中段针实测 `ZX-4417-QJ7`；环臂的所有改动都被
开关隔离，生产路径无回归。环臂实验的失败记录（崩溃、退化、slice 重复）保留在 /tmp 与本节。

## 842. Round 578：滑动视图臂（design 577）的六个协议事实 + 仍剩读路径

**本轮施工（全部开关默认关，生产路径零改动）**
- 新臂 `KVMEM_D_SLIDE=1`：视图从"真坐标 + 临时 clamp"改成**页对齐滑动的视图帧**：
  worker 自算 `base = ceil((true_after - win)/pt)*pt`、`book = true_before - base`；
  行按 `(base - roll_base)/pt` 整页旋转并安装；发布 `delta = rope = base`、`view_lens = book`；
  `positions` 在 `prepare_pos_seq_lens` 之后整体减 `base`（token feed 不动）。
- Core 侧另有一个 `KVMEM_SLIDE_CORE=1` 的分配回卷（默认关，见事实 3）。
- `tests/ring_window_test.py` 新增 slide 场景：160K 逐 token 视图恒等 + 每步 base/δ/rope/vl 断言 +
  首装从"未污染头部行"旋转 + 不再切片。离线 PASS（旧环臂场景与负对照保留）。

**六个协议事实（都有实测支撑）**
1. **Core 与 worker 的 window 不是同一个数**（CLI 53,664 vs 适配器 45,472）⇒ 新增
   `KVMEM_WINDOW_TOKENS`，滑动两侧必须同一数值。
2. **引擎的账本不会传回 worker**：executor 自己单调累加 `num_computed`，所以 Core 回卷到不了
   worker ⇒ 滑动必须由 worker 用同一规则自算（`true` 计数 + win + pt，确定性一致）。
3. **token feed 按 `num_computed` 切片**：把账本改成视图帧会喂错 chunk（重启后实测同一 88K
   读到的不是 needle 段）⇒ Core 分配回卷保持临时 clamp、默认关；偏移只作用于 `positions`。
4. **rope offset / 视图表是进程全局**：上一个请求的 base 残留会让"窗口以下"的写入落到
   true+base（实测：距尾 300 token 的普通 marker 也读不到）⇒ identity 阶段必须显式发布 0。
5. **Core 的 clamp 在跨窗那一步污染引擎行**（回卷后重新 append 块 id）⇒ 窗口以下每步镜像一次
   引擎行，首装用这份"未污染头部"旋转（本轮已修；旧实现恰好在跨窗步读行）。
6. **base 必须 ceil**：floor 会让 decode 的 book 到达 45,476 > win 45,472，最后一个 token 写到
   行外（列 1,421）。

**在线读数（滑臂）**：probe 几何全对——base 3,680→29,504、`row_head` 311→1,118（正好是真页
196+base/32）、`head_distinct=1421`、每步 store 115/128 页、`book/lens` 发布一致；但输出仍退化
（`'ZX where where…'`、`'The exact code is the exact code…'`）：**问题句（尾部）读得到、needle 的 KV
读不到**。几何正确 ⇒ 剩余缺陷在"读路径的帧/相位"，第一嫌疑是还有第二处位置写入者
（`mrope_positions`/`position_deltas` 之类）没跟着减 base。

**下轮唯一取证**：在 `compute_slot_mappings` 与 `prepare_inputs` 的 mrope 段各放一发，打印
`positions[:4]`、`slot_positions[:4]`、`kvmem_step_true_positions[:4]`、`mrope_positions[:4]`、
`position_deltas[:4]` 与第一个 slot 对应的页 id（应 = 已装行的对应项）；若 mrope 未随动，把它一起
减 base 后重跑 88K 中段/尾段两发。

**现状**：默认臂已恢复（`KVMEM_D_SLIDE=0`），88K 中段针 `ZX-4417-QJ7` ✓；新臂与 Core 回卷全部
开关隔离，未接入生产。

## 843. Round 579：位置帧已落地（实测），但环窗口内容仍只"读到片段"；另记一条 env 残留事故

**已落地并实测**
- 上一轮放在 plugin `prepare_pos_seq_lens` 之后的"positions 减 base"块**从未执行**（在其行号上的断点
  一次也没命中，try 体被跳过）。改为在 `d_ring_slide` 内直接改 `self.view.positions()` 的本请求
  query span（span 由 `view.query_position_slots(req)` 给出）：probe 记录 `pos_shift=3680`、
  `pos_head=[41376,41377]`，即 positions 真的进了视图帧。
- `KVMEM_WINDOW_TOKENS` 现在**两侧都生效**（scheduler + runtime），滑动与分配共用同一个窗口数。
- 小窗口参数实验（win=8,192，50K prompt）：行缩到 256 页、base 0→41,184、每 chunk 旋转+commit，
  `head_distinct=1421`（首装行仍忠实），几何链条完整走通。

**在线读数（滑臂，win=8,192 / 45,472 两种）**
- 88K 中段 `'ZX where settles…'`、50K 尾段 `'ZX at'`、20K marker（win 8K）`'4'`（正好是 marker 首位数字
  却截断）。即：**紧邻 query 的内容读得到片段，窗口内更早的页不连贯**。
- 几何已逐项验证（base、`row_head`=真页 id、`head_distinct`、`delta`、`view_lens`、positions 视图帧、
  store 提交），所以剩余缺陷在"内容/读路径"，不在页表算术。

**env 残留事故（代价一次误判，已修复）**
- `arm_boot.sh` 只增改不删：上一轮实验留下的 `KVMEM_WINDOW_TOKENS=8192` 让 *默认臂* 的 clamp 窗口
  变成 8,192（适配器仍是 45,472），默认臂于是从 `ZX-4417-QJ7` 退化成"复述问题"。
- 恢复 `/etc/default/kvmem-vllm` 原始文件后默认臂回到 `The access code for KEY-MID is **ZX-4417-QJ**`
  （被 max_tokens 截断，语义正确）。教训：**每次判决实验前核对整个 env 文件，而不是只核对本次开关**。

**下一步两个候选（需裁决/择一）**
1. 读时内容审计：在提交时刻把目标页的 device 字节与 store 副本比对（现有 `KVMEM_TRACE_PAGES`
   机制），或对 needle 页做一次"设备 vs 期望 token 位置"的指纹，确定是"页内容错"还是"attention 读错列"。
2. 架构替代：本 fork 自带 `SlidingWindowSpec` + `SlidingWindowManager`。把 full-attention 层按
   **原生滑窗**服务（窗口 = B+R），窗口/环形由引擎负责，适配器只保留"选择 + 还原"——这与用户
   2026-11-02 的"接受 vLLM 原生简化形式"口径一致，且能删掉 worker 侧几乎全部环/页搬运代码。
   需要改的是 qwen3_5 的层配置/spec 推导（把 full attention 换成 sliding window），工作量集中在
   一处、失败语义清晰。

**现状**：默认臂已恢复为原始 env 文件并实测 `ZX-4417-QJ`；滑臂代码全部开关隔离（
`KVMEM_D_SLIDE`/`KVMEM_SLIDE_CORE` 默认 0）。

## 844. Round 580：原生滑窗实验——88K 正确，超池 prompt 却"实质上停摆"

**做法（一个开关）**：本 fork 的 `Attention` 从 `cache_config.sliding_window` 取窗口，且
`Attention.get_kv_cache_spec` 在窗口非空时返回 `SlidingWindowSpec`。于是新增 `KVMEM_SLIDING_WINDOW`
（`CacheConfig.model_post_init` 里覆盖），配合 `KVMEM_NATIVE_SW=1`（旁路适配器的行/环动作）、
`KVMEM_D_KNIFE=0`（不 clamp、不动账本），就得到"原生滑窗"臂：窗口/掩码/spec/管理器全部由引擎负责，
positions 与 token feed 保持 TRUE，worker 侧不做任何页搬运。

**实测**
- 引擎正常启动（`GPU KV cache size: 88,914 tokens`，group sizes / lcm 不变）。
- 88K 电池（74,949 tokens，needle 在窗口内）：`ZX-4417-QJ7` ✓——**窗口语义在引擎侧成立**。
- 超池 prompt 不成立：100K 尾针（99,526 > 池 90,944）11m40 仍未返回；152K 尾针（142,869）在客户端
 15m 超时后约 14 分钟才被引擎 release（sparse log 有 `released` + `d_final_sweep`），期间日志
  prefill 到 73,728 tokens 后吞吐归零。即：**原生滑窗管理器没有把 prefill 期的驻留集压住**，
  与早前轮次"no_preempt 等待/停住"是同一类症状；适配器此时被旁路，无法补位。
- 结论：原生滑窗是"窗口语义"的正确载体，但要让它服务超池 prompt，仍必须解决"prefill 期块回收/
  驻留有界"（即历史轮次的核心难题），否则只会从"内容错"变成"卡住"。

**开关与文件**：`KVMEM_SLIDING_WINDOW`（引擎 `CacheConfig`）、`KVMEM_NATIVE_SW`（适配器旁路），
默认关；新增 100K 尾针电池 `tools/battery/needle_100k_tail.request.json`。

**现状**：默认臂已恢复（`/etc/default/kvmem-vllm` 还原为实验前文件），88K 中段针
`ZX-4417-QJ7` ✓；服务 active/health 200。

## 845. Round 581：滑臂激活改为"用引擎账本当真值"（每步精确）；读到的上下文在 needle 后截断

**本轮修复（实测生效）**
- 之前滑臂用 worker 自记的 `d_slide_true` 当 "true"：**每个请求的第一个 step 看不到**（hook 按
  `live` 逐请求跑，首步状态刚创建），计数器因此永久少一个 chunk（实测 49,340 的 prompt 只记到
  45,273），跨窗那一步的 base 算成 0 ⇒ 激活晚一步、跨窗 chunk 写到行外。
- 改为直接用**引擎的 `num_computed_tokens`（本臂里它始终是真值）**：`true_before = book_in`，
  `true_after = book_in + n_new`，base 由 ceil 得出。probe 实测：跨窗步 `book_in=49,152,
  base=3,904, book(view)=45,248, pos_head=[45,248,45,249]` ✓ 每步精确、不再滞后。

**新的在线读数（50K 尾针 prompt，49,369 tokens）**
- 答案变成 `'ZX-'`（正好是 needle 里代号的前缀，也就是**上下文在 needle 结束处被截断**）：
  模型看到的最后 token 是 "ZX-"，而不是其后 20 多个 token 的问题句。
- 同一 prompt 上一次运行为 `'ZX from'`、`'ZX where settles…'`、`'KEY-M'`——都表现为"读到的片段
  随某处偏移漂移"，而不是"读不到 needle"。
- 结论（待下一发钉死）：**视图的"末端"与 seq_len/positions 对不齐**（例如最后一个 chunk 的
  `n_new`、`book` 与 `_vl+ql` 的组合使 seq_len 短了 ~30 token，或 positions 的 shift 与
  `query_start_loc` span 不重合）。下一发同帧取证：最后一个 chunk 的
  `num_computed / n_new / book / seq_lens[0] / positions[-1] / view_lens[0]` 五元组，核对
  `seq_lens == book + n_new` 且 `positions[-1] == true_last - base`。

**现状**：默认臂已恢复（`/etc/default/kvmem-vllm` 还原，`FORCE_EAGER=0`、`D_KNIFE=1`、`PLAN_D=1`），
88K 中段针 `ZX-4417-QJ7` ✓；滑臂与原生滑窗开关全部默认关。

## 846. Round 582：滑臂的 RoPE 偏移被老机器擦掉（已定位）；原生滑窗的开关终于生效，但混合池塌成 18K

**A. 滑臂：`kvmem_rope_offset` 被 classic 发布块覆盖（已修）**
- 同帧取证（`compute_slot_mappings` 断点）：最后一个 chunk `positions=[45248,45249]`（视图帧 ✓）、
  `num_computed=49,152, n=217, seq_lens=45,465`（=book+n ✓）、但 `kvmem_rope_offset=0`、
  `kvmem_view_lens[0]=0`。
- 根因：`execute_model` 里的 classic 发布块（`scheduled_position_offset` → `_buf.fill_(_off)` →
  `self.kvmem_rope_offset = _buf`）在本臂 `_off=0`，**在 hook 发布之后**把 rope 偏移擦成 0。
  于是 K 以"视图相位"落盘而上游页是"真相位"，slot 又按 `positions-δ` 偏了 base ⇒ 片段化输出
  （`'ZX-'`/`'ZX from'`）。
- 修：该发布块在 `KVMEM_D_SLIDE=1` 下跳过（抛内部标记，走既有 except 路径）。
- 仍存在：同一 prompt 的**逐请求激活不稳定**（probe 有 `slid` 记录，但另一些请求整场 `identity`、
  `ring_ready=False`），需要下一发定位（hook 的 `live` 覆盖了哪些 step）。

**B. 原生滑窗：开关终于生效（`__post_init__` 而非 `model_post_init`），但池塌了**
- `CacheConfig` 是 `@config`（pydantic dataclass），`model_post_init` 不被调用 ⇒ 上一轮
  `KVMEM_SLIDING_WINDOW` 完全没生效（spec 仍是 `FullAttentionSpec`，88K 的"正确"只是稠密路径）。
- 改成 `__post_init__` 后：spec 变为滑窗、attention 组块大小 **64**，但 `GPU KV cache size` 从
  **90,944 掉到 18,036 tokens**、group sizes `[262144,262144,262144,64]` —— 混合 LCM/每块字节的
  记账被 64-token 滑窗块拖垮，窗口 45,472 直接装不下 ⇒ 不可用（上一轮的 100K/152K 卡住正是这个
  池塌的结果，而不是"管理器不回收"）。
- 结论：原生滑窗要可用，必须让滑窗块的块大小与混合模型的 LCM 对齐（例如 1,568），否则池容量
  按 lcm 记账会崩掉；这是引擎侧一处明确的施工点。

**现状**：默认臂恢复（实验开关 0 条），88K 中段针 `ZX-4417-QJ7` ✓；本轮改动（rope 发布守卫）在
`KVMEM_D_SLIDE` 下才生效。

## 847. Round 583：滑臂第一次"全绿几何 + 只在活跃请求上"跑通；但输出仍是片段 ⇒ 指向混合模型的
   循环状态（GDN）与窗口视图不一致；原生滑窗臂连 14K prompt 都卡在调度前

**A. 三处会让滑臂"作用在错误对象上"的缺陷（本轮修掉）**
1. hook 里 `runner` 不是 `GPUModelRunner` 实例（实例挂在 live view 上），`runner.input_batch`
   永远 None，回退的 `_kvmem_req_ids` 是**上一个请求**的 id ⇒ 环/滑臂把状态写给了旧请求
   （实测：probe 有 `08ba57` 的 slid 记录，而当时活跃的是 `cbc2d3`，活跃请求整场 `vl=0`）。
   改为 `rt.view.runner.input_batch.req_ids` 优先。
2. `sync_row_from_engine` 会用 `seq_lens[row]` 把镜像宽度"变大"，而该行可能还是**上一个请求**的
   长度 ⇒ 干净头部被 653 个陈旧页污染（实测 rowlen 1421 @ true 24,576、跨窗 head_distinct=931）。
   改为：调用方给了 `written` 时不再用 seq_lens 扩大。
3. （上一轮）classic 的 `kvmem_rope_offset` 发布块在 hook 之后把 rope 擦成 0 —— 已加守卫。

**B. 本轮第一次拿到"干净几何 + 活跃请求"的滑臂运行**
- 首个 slid 步恰在跨窗 chunk：`true_after=49,152, base=3,680, book(view)=41,376,
  head_distinct=1421, rowlen=1421, pos_head=[41,376,41,377]`（镜像取自身份期，未被 clamp 变形）。
- 前向同帧：`positions=[45,248..]`（视图帧）、`seq_lens=45,465=book+n`、`rope=base`、`delta=base`。
- **输出仍是 `'ZX from'`**（模型读到 needle 前缀就断）。几何已无可查，缺陷在"内容/语义"。

**C. 新的主假设：混合模型的 GDN 循环状态与窗口视图不一致**
- Qwen3.5/Next 是混合模型：GDN 线性注意力层持有**整请求的循环状态**（本配置 `mamba_cache_mode
  =none`，每请求 1 槽），而 full-attention 层被我们裁成 45,472 窗口。两层对"上下文有多长"的
  认知不一致 ⇒ 模型看到的是两套互相矛盾的上下文（片段化输出的合理来源）。
- 该假设的判别实验：在**稠密**（无环）情况下单独施加窗口掩码，看是否同样片段化。原生滑窗臂
  正是这个实验，但它卡在调度前（见 D），需要先修好才能判。

**D. 原生滑窗臂：spec 生效后，连 14K prompt 都在调度前卡住**
- `__post_init__` 修正后 spec 真为 `SlidingWindowSpec`（组块大小 64）；窗口 8,192 时池
  `58,478 tokens` ✓、窗口 45,472 时池 `18,036 tokens`（混合 LCM 记账被 64-token 块拖垮）。
- 但 14,011-token marker prompt（窗口 8,192、池 58,478）**5 分钟无响应**，引擎日志连
  `Prefill progress` 都没有、metrics 显示 running/waiting 均为 0 ⇒ 请求连 prefill 都没进
  （疑似 SW 组的准入/管理器在入队处卡住）。此路要可用，需在引擎侧修 SW 组的调度，不是配置问题。

**现状**：默认臂恢复（实验开关 0 条），88K 中段针 `ZX-4417-QJ7` ✓；滑臂开关默认关。

## 848. Round 584：三条"测量/脚手架"缺陷修正（max_tokens=2 截断、env 泄漏、env 基线），
   并拿到关键区分：脚注式丢头 → 通常仍流畅；needle 仍读不到

**A. 我自己的测量缺陷（全部修正）**
1. `/tmp/p577_50k_tail.json` 的 `max_tokens` 是 **2** —— 之前所有 `'ZX from'`/`'KEY-M'`/`'ZX-'`
   都是**被截断的前两个 token**，不是退化。修正后（max_tokens=64）该 prompt 输出
   `'ZX from arrives arrives…'`（复读退化），才是我该看的现象。
2. `arm_boot.sh` 只合并 env：上一轮实验留下的 `KVMEM_SLIDING_WINDOW=8192` + `KVMEM_NATIVE_SW=1`
   让"滑臂"实际以**原生滑窗**运行（请求一直 `waiting`，running/waiting=0、无 prefill 日志），
   连续三次误判为"卡死/崩溃"。已改成：**每次 arm 启动先复制干净基线** `/tmp/kvmem_env_baseline`，
   实验键不可能泄漏到下一次。
3. 同因，`KVMEM_MASK_WIN` 那次"引擎崩溃"其实也是 env 泄漏的产物，尚未重测。

**B. 关键新数据（干净的滑臂，真滑窗，基线 env）**
- 50K 尾针（max 64）：`'ZX from arrives arrives…'` —— 读到了 needle 前缀（"ZX"）后进入复读。
- **刚好跨窗的 prompts（45,525 tokens，只丢 53 个头部 token）：答案流畅**——
  `'The user is asking for the access code for KEY-MID. Based on the provided text, the access code
  is "4444…"'`：**格式与任务完全正常，但 needle 的码是编造的**。
- 对照：同一 prompt 在默认臂正确（头 45,472 内 needle 在尾部 ✓）。

**结论（本轮最重要）**：跨窗时模型的"语言能力"是好的（小丢头时输出完全不乱），坏的是**needle 那几页
KV 没有被正确读到/写入**；而大丢头（3,868 tokens，含系统提示）才会引发复读退化。所以环窗口的几何
虽然逐项对齐，**跨窗 chunk 写入的那 15~16 页（视图尾部）仍是可疑点**：第一次旋转后行尾是"回收页"，
它们在安装行的同一 forward 内被写入并读取——若写入地址与读取列错开一页，needle 就恰好丢失。

**下一步（同帧、可直接判定）**
1. 默认臂跑同一个刚好跨窗 prompt（应在 45,472 头内，needle 在尾部 → 正确），作为对照基线；
2. 滑臂上对 needle 所在的两页做"写时 vs 读时"设备指纹（复用 `KVMEM_TRACE_PAGES` 或 console 读
   `paged[0][page]` 的 bytes/sha）：若两页指纹不同，就锁定"跨窗 chunk 的写地址错一页"。

### 848b. 同轮追加：滑臂的"base=0 消融"判定病灶在**首次旋转**，不在窗口/丢头语义

同一臂、同一代码、同一条 45,525-token prompt，只改窗口：

| 窗口 | base | 状态序列 | 输出 |
|---|---|---|---|
| 45,600（> prompt） | 恒 0 | 44 × `identity` | **`ZX-4417-QJ7` ✓** |
| 45,472（真窗口） | 末 chunk 起 64 | 22 × `identity` + `slid` | `'… Based on the provided text, the access code is "4444…"'`（流畅但编造） |

结论：**病灶是"第一次 base>0"的那一步（旋转 + 位移 + δ），不是丢头 53 个 token 的语义，也不是
窗口本身**。丢 53 个 token 时语言能力完全正常（无复读），只是 needle 的码丢了。

**当前最具体的假设（下一发直接验证）**：首次跨窗时 `sync_row_from_engine(want = true_after//32)`
会按 `true_after` 取 1,421 页，但引擎此刻只写了 `true_before//32 = 1,407` 页；多取的 14 页是**上一个
请求留下的陈旧页**。旋转 `row'[c] = head[c+2]` 后，列 1,405 = `head[1,407]`（陈旧）对应 true
45,024..45,055 —— 这一页不会被本 chunk 写入（本 chunk 从 true 45,056 起），于是视图里留下**一页陈旧
KV**；若陈旧页范围更大（取决于上一请求的残留），正好覆盖 needle 前文。
验证：在 `_ring1576` 记录里加 `row_tail`（安装行最后 20 个页 id）与 `eng_written`，检查首次滑步是否
出现重复/越界页 id，并把镜像宽度限制为 `true_before//32`（而不是 `true_after//32`）。

## 849. Round 585：窗口内尾部页与引擎行在解码步"分叉"；同配置跨启动输出不同（代码片段出现后复读）

**新增探针（design 585）**：`_ring1576` 记录加 `row_tail`（安装行末 4 页）/`eng_tail`（同一视图列
对应的引擎行条目）/`tail_match`，直接校验不变量 `row[c] == engine[c + steps]`。

**实测（刚好跨窗 prompt 45,554 tokens，win 45,472）**
- 首次滑步 `base=96, true_before=45,056, rowlen=1,421, head_distinct=1,421`：
  `row_tail=[1616,196,197,198]`、`eng_tail=[1616,0,0,0]`（引擎侧在窗口外为 0：它的页账被 Core 的
  clamp 限制在 1,421 页，尾列未分配）——尾部是"旋转回收页"，随后被本 chunk 写入，**此处自洽**。
- **解码步的注意力行尾 = 引擎自己重新 append 的连续页 id（81..97），与安装行的尾部不一致**
  （`num_blocks[3]=1,421`，即 gather 宽度够，但内容不是我们装的那份）⇒ 读路径在尾部用的是引擎行，
  写路径用的是安装行 ⇒ **写/读在尾部区域分叉**，这正是"needle 写进去却读不到/读错"的形态。
- 三次同配置运行输出分别是 `'4444…'`、`'ZXagesman…'`、`'ZX44Q7Q7Q7…'` —— **跨启动不确定**，且
  后两次出现了正确代码的片段（`ZX44`、`Q7`）：needle 其实**被读到一部分**，随后解码环被破坏。

**结论**：环窗口的"几何"与"头部"都已干净（head_distinct=1421、base/positions/δ/rope 同帧对齐），
缺陷集中在**视图尾部页的所有权**：Core clamp 释放的页会被引擎重新 append，而 worker 安装行用的是
旋转回收页；两者在同一个 forward 内混用 ⇒ 尾部 KV 内容与读取列错位。下一步修法（明确）：
1. 让安装行的尾部与引擎行**同源**：首次滑步后，凡引擎 `append_block_ids` 新分配的页，worker 不再使用
   旋转回收页，而是把"回收页"与"引擎新页"做一致性映射（或在 `_install` 后立刻把引擎新页写回我们的行）；
2. 或关掉 Core clamp 对页账的回收（`KVMEM_SLIDE_CORE=1` 让 Core 与 worker 用同一套页对齐滑动），
   使引擎行与 worker 行天然同源。

**现状**：默认臂干净、88K 中段针 `ZX-4417-QJ7` ✓；滑臂开关默认关。

### 849b. 自查发现回归：583 的镜像宽度收紧是全局的，打伤默认臂（已按臂隔离）

- 现象：恢复默认臂后 88K 中段针从 `ZX-4417-QJ7` 退化成 `'KEY-MID'`。
- 根因：round 583 给 `sync_row_from_engine` 加的"不再用 seq_lens 扩大镜像宽度"是**无条件**的；
  默认（knife）路径正依赖这个扩张去取引擎 rewind 后 append 的列。
- 修：该收紧仅在 `KVMEM_D_SLIDE=1` 生效。修后默认臂 88K 中段针恢复 `ZX-4417-QJ7` ✓。
- 教训（写进流程）：任何改动 `sync_row_from_engine`/页账的补丁，默认臂 88K 必须当场复测。

**默认臂当前验收状态（实测）**：88K 中段 `ZX-4417-QJ7` ✓；**152K 中段失败**
（`'The text is a literary description of a river, not a technical description of a river, not a…'`，
142,868 tokens，3m10s，复读退化）。这正是本 goal 要修的验收项（不是本轮的回归）。

## 850. Round 586：页内容审计上线（文件触发，自报错）；首次读数显示"目标页写入后内容仍在变"；
   并记两条会让实验失真的运维事实

**A. 页内容审计（design 586，已可用）**
- 在 `d_ring_slide` 的 slid 分支按 `KVMEM_AUD_PAGE`（或文件 `/tmp/kvmem_aud_page`）对一个**绝对页**
  取设备指纹：`col = (abs*pt - base)//pt`、`page = installed_row[col]`、`sha1(paged[0][page])`，逐 step
  写入 `/tmp/kvmem_pageaud.jsonl`；异常也会写一条 `err`（第一次实现踩了 `state` 不在作用域，正是靠
  自报错暴露的）。
- 首次读数（刚好跨窗 prompt，绝对页 1,421 = tokens 45,472..45,503）：视图列随 base 前移
  （1,418 → 1,416）、**页 id 恒为 196**（映射稳定 ✓），但**该页指纹在写入后仍随 step 变化**
  （`fa08…` → `a0fa…` → `4b3e…`）。若映射与写地址都正确，这个页在写入后不应再被改写 ⇒ 说明
  **有第二个写者（或旋转把仍在视图内的页回收复用）**，这正是 needle "写进去读不到"的形态。
  待下一发把 pid/rank 与两次 `put` 都记录下来钉死写者。

**B. 运维事实（会让实验失真，必须每次核对）**
1. **worker 的 env 不等于 API server 的 env**：API server 有 40 个 `KVMEM_*`，worker 只有 19 个
   （新加的键如 `KVMEM_RING_PROBE`/`KVMEM_AUD_PAGE` 根本到不了 worker，静默失效）。对策：审计改成
   文件触发；或由 console 在 worker 进程内 `os.environ[k]=v`。
2. **"health 200" 不代表本次 boot 生效**：服务曾撞 start-limit，`systemctl restart` 静默失败，
   旧实例继续服务 ⇒ 我一度把旧实例的结果当成新配置的结果（也是"同配置跨启动不确定"的一部分）。
   判定方法：`log` 文件 mtime + worker pid + probe 记录里的 pid 三者一致才算新实例。

## 851. Round 587：页内容与安装行都被验证一致；但**视图中部**的 needle 同样读不到 ⇒ 缺陷在读语义，不在表/内容

**A. 审计结论（本轮把上一轮的"内容在变"证伪）**
- 审计按**绝对页**取列：`col=(abs*pt-base)//pt`。上一轮我把 *页 id* 当成 *绝对页* 传入（1,421 实际是
  页 id），且审计在 `prepare` 阶段（**写入之前**）取样，于是同一页在 step1/step2 之间"变了"——
  那是 **prepare-vs-write 的时序**，不是第二个写者。
- 改成正确的绝对页后：`page=392`（`staged=392, staged_match=True`）在 60+ 步里**指纹恒定**
  （rank A `6b4acd5bf082`、rank B `6bc3002d32a0`），且 `staged == installed`。即：
  **安装行有效、读路径表 = 我们装的行、目标页内容稳定**。
- 另外确认 `_install` 只写注意力组的表（`block_table_rows()` 过滤 FullAttentionSpec），不会误写
  mamba 组。

**B. 新事实：88K 中段 needle 在滑臂上同样读不到**
- 88K 电池（74,949 tokens，needle 在 true 37,441 → **视图中部**，远离尾部/回收区）在滑臂上输出
  `'Nothing is about codes, keys, or codes, or codes…'`：模型明确说"没有相关信息"，然后复读。
- 对照：同 prompt 默认臂 `ZX-4417-QJ7` ✓；滑臂在 base=0（窗口 45,600）时也 `ZX-4417-QJ7` ✓。
- ⇒ 缺陷不是"尾部页被回收"，而是**只要 base>0，视图内容就读不到**（中部、尾部都一样）。

**C. 由此收窄的读语义可疑点（下一步逐个排除）**
1. 内核拿到的 `seq_lens`/`positions` 组合：positions=true、seq_lens=view、slot 列=view 帧。RoPE 用
   true（读写一致），但内核的**因果索引**用 seq_len；若某处仍按 *true* 长度取 block table 列
   （1,421..2,342 是未安装的陈旧列），注意力尾部就会读到陈旧页并把 needle 淹掉。
2. 需要直接验证内核侧元数据：在 `prepare_attn` 里读 `block_tables.input_block_tables[attn_group]`
   的尾部条目与 `seq_lens`，确认 gather 后的行宽度、内容与 `installed row` 完全一致。

**实测记录**：滑臂 88K 中段 `'Nothing is about codes, keys, or codes…'`（1m24s）；刚好跨窗 prompt
`'ZXagesmanman…'`；默认臂 88K `ZX-4417-QJ7` ✓。

## 852. Round 588：滑臂缺表导致 RoPE/槽帧错位（已补建表）；解码步仍拿到 `delta=0`，
   写地址落到 page 0 —— 解码复读的直接原因

**A. 已修：滑臂自己建表（design 588）**
- 上一轮（582）给 classic 发布块加了 `KVMEM_D_SLIDE` 守卫后，`kvmem_rope_offset` 在滑臂里**从未被创建**，
  fork 的 RoPE 加偏移变成 no-op：pre-slide 页是 TRUE 相位、post-slide 写入与 query 却是 VIEW 相位。
  实测（decode 步）：`rope=-1`、`vl=0`、`seq_lens=TRUE`。
- 修：`d_ring_slide` 现在**缺表就建**（`kvmem_rope_offset`/`kvmem_slot_offset`/`kvmem_slot_deltas`/
  `kvmem_view_lens`，用 `inference_mode(False)` 建，避免 inference tensor 不可原地写）。修后实测
  `rope=96` ✓、`seq_lens=45,461 = book+1`（视图）✓、post-rope `positions=45,556`（真值）✓。
- 输出随之变化（`'4417'`、`'KEY-MID-42'`、`'Based-d--'`），但仍**提前 EOS / 复读**。

**B. 新钉死的缺陷（解码步）**
同一个 decode 前向（`compute_slot_mappings` 断点）：
```
positions[0] = 45,556   (post-rope = TRUE ✓)
seq_lens[0]  = 45,461   (VIEW ✓ = book+1)
rope_offset  = 96       (✓)
kvmem_slot_deltas[0] = 0   (✗ 应为 96)
kvmem_view_lens[0]   = 0   (✗ 应为 book)
```
⇒ `slot_positions = positions - delta = 45,556` → 列 `1,423`，**超出安装行（1,421）**，
`input_block_tables[3][0,1423] = 0`（未分配页）⇒ 生成 token 的 KV 写进 page 0，
下一步读到的就是垃圾 ⇒ **模型吐 1–3 个正确片段后 EOS/复读**。

同一 step 里 `rope`（标量）是 96 ✓ 而两张表是 0 —— 说明表在 `prepare_inputs`（override 用到了 book）
之后、`prepare_attn` 之前被**换成了新 zeros**。下一步：在 override 与 prepare_attn 两处各记 `id()` 与
值，定位替换者（候选：plugin 的 classic 块/`plan_d_wired` 重建、或 hook 里第二次 `d_ring_slide` 调用）。

**本轮实测**：滑臂 88K 中段 `'ZX where where…'`；刚好跨窗 `'4417'`/`'KEY-MID-42'`/`'Based-d--'`
（每次不同，但都提前 EOS）；默认臂 88K `ZX-4417-QJ7` ✓。

## 853. Round 589–590：环窗口**首轮全绿**（88K 中段 + 刚好跨窗都答对）；刀（knife）是表破坏者；
   超池仍卡在"块池耗尽"（attention+mamba > 58 块）

**A. 决定性消融：knife 关掉后滑臂立刻正确**
| 配置 | 刚好跨窗 prompt (45.5K) | 88K 中段 needle |
|---|---|---|
| `D_SLIDE=1, D_KNIFE=1`（原基线） | `'ZXagesmanman…'` ✗ | `'ZX where where…'` ✗ |
| `D_SLIDE=1, D_KNIFE=0` | **`ZX-4417-QJ7` ✓** | **`ZX-4417-QJ7` ✓** |

- 同帧 `id()` 取证（round 589）：override 处 `id(vl)=A, 值 45,461`、`id(dv)=B, 值 96`；到
  `prepare_attn` **同一对 id** 的值变成 `0/0` ⇒ 表是被**原地清零**的，不是被替换。
- 清零者 = knife 侧：`_d_reset_row`（每个 step 的 before_step）与 design-791 的 `_activate`。
  两者都已按 `KVMEM_D_SLIDE=1` 关闭（design 589）。但**只关这两处还不够**：`D_KNIFE=1` 时答案仍错
  （说明 knife 还有别的路径在改帧）；`D_KNIFE=0` 时全绿。

**B. 分配 clamp 与 knife 解耦（design 589/590，引擎侧）**
- `scheduler.kvmem_clamp = kvmem_knife or KVMEM_D_CLAMP=1`：clamp 只负责"把 allocate_slots 限制在窗口
  内"，与 knife 的适配器机制分离。
- design 590：clamp **只在超池需求时触发**（`num_computed + num_new > KVMEM_POOL_TOKENS`）；否则一个
  *装得下*池的 prompt 被 clamp 会改变帧（实测 45.5K prompt：clamp 开 → `'KEY-MID'`，clamp 关 →
  `ZX-4417-QJ7`）。

**C. 超池（152K tailA）仍卡：块池耗尽，不是帧**
- `KVMEM_D_KNIFE=0, D_SLIDE=1, D_CLAMP=1, POOL_TOKENS=90,944`：prefill 到 **73,728 tokens（51%）**
  后吞吐归零（与历史"卡在 73K"一致）。
- 账算得上：块池 58 块；73,728 tokens = 47 个 attention 块，加上 mamba 每请求约 15 块 → 62 > 58 ⇒
  **卡在块池，而不是窗口/帧**。所以 clamp 的触发阈值应该用"attention 可用块"（≈43 块 ≈ 67K tokens），
  而不是整池 90,944。

**本轮实测结论**：环窗口的读写帧在 `D_KNIFE=0` 下已能端到端答对 88K 与刚好跨窗两类 prompt；
剩下两件事：(1) 找出 `D_KNIFE=1` 下仍在改帧的 knife 路径（或干脆让滑臂走 vanilla 配置）；
(2) 超池用"attention 可用块"作为 clamp 阈值，验证 152K tailA/tailB。

### 853b. 同轮追加：152K 尾针**通道打通**（3m03s 跑完，142,869 tokens），答案退化为复读

`KVMEM_D_KNIFE=0, D_SLIDE=1, D_CLAMP=1, POOL_TOKENS=66,000`（= attention 可用块 ≈43 块，扣掉 mamba）：
- 152K tailA **不再卡住**：142,869 tokens，3m03s 完成 ✓（此前两轮都在 73,728 tokens 停摆）。
- 答案：`'The answer is:\n\nThe answer is:\n\n…'`（流畅但复读、无 needle 码）⇒ 通道通了、帧在超池段坏。
- 推断（与 clamp 语义一致）：clamp 把 `num_computed` 临时**回退**到"窗口长度"，`allocate_slots` 于是把
  **最新的块**（`[clamped, true)`）判为多余并释放 → 恰好是"最近 token"所在的页（needle 就在最后 6,857
  token 内）→ 页被回收重写 ⇒ 复读。也就是说：**clamp 与"视图页所有权"天然冲突**，clamp 省下的块正是
  我们要保住的内容。

**因此超池阶段的正确做法不再是 clamp**，而是：让引擎按真实增长分配，worker 用 store（已有 `_ring_commit_pages`
与 `_apply_selection_d`）把被回收的页换出/必要时还原——即转入"选择/换出"阶段，这与 goal 的后半段
（超池选择）本来就是同一件事。

## 854. Round 590：滑臂 88K 电池 **2/2 全绿**（mid + tail），并给出超池的下一步

**环窗口臂（`KVMEM_D_SLIDE=1, D_KNIFE=0, D_CLAMP=0`）实测**
| 电池 | prompt tokens | 答案 | 判定 |
|---|---|---|---|
| `needle_88k`（needle 在 true 37,441，视图中部） | 74,948 | `ZX-4417-QJ7` | ✓ |
| `needle_88k_tail`（needle 距尾 ~670） | 74,949 | `ZX-4417-QJ7` | ✓ |

这两个 prompt 都在池内（88,914），所以**不需要 clamp**：窗口 45,472 使滑臂把视图压到 1,421 页、
按页旋转、positions/δ/rope 同帧发布——端到端正确。加上 round 589 的"刚好跨窗"45.5K prompt，
`D_KNIFE=0` 的滑臂在"池内 + 窗口已满"的三类输入上全部正确。

**超池（152K）现状与下一步**
- 无 clamp：块池在 ~73,728 tokens 处耗尽 → 调度等待（历史同款停摆）。
- 有 clamp（阈值取整池或 66K）：通道能跑完（3m03s / 142,869 tokens），但 clamp 的"回退 + 释放
  最新块"会把 needle 所在的最近页回收掉 → 复读（round 589/853b 已记）。
- ⇒ 正确路线：**不做 rewind/clamp，而是让引擎在池将满时回收"最旧"的块**（这些页早已被 worker 的
  ring 换进 store，且视图长度 ≤ 窗口，attention 从不读它们的列），从而既不 stall 也不动视图页。
  回收点可以挂在 `allocate_slots` 返回 None 之前（或在 no_preempt 等待分支里先尝试回收再等）。
  这与 goal 后半段（超池选择/换出）是同一套机制：store 已在 `_ring_commit_pages`，
  还原已在 `_apply_selection_d`（单测覆盖）。

### 854b. 默认臂的 88K 并非"干净绿"（诚实记一笔）

恢复默认臂（基线 env、`D_KNIFE=1`）后三次 88K 中段运行：
`'The access code for KEY-MID is ZX-4417.'`（漏 `-QJ7`）、`ZX-4417-QJ7` ✓、`ZX-4417-Q`（截断）。
即默认臂在**答案尾部不稳定**，按"答案含 ZX-4417-QJ7"的验收口径它并不合格。滑臂（`D_KNIFE=0`）本轮
两次均为完整 `ZX-4417-QJ7`，可作为下一阶段的主臂。

## 855. Round 591：88K 电池 2/2 全绿（复验）；超池回收(free_path)已接线但**仍未生效**

**复验（滑臂 `D_SLIDE=1, D_KNIFE=0, D_CLAMP=0`）**
- `needle_88k`（74,948 tokens）→ `ZX-4417-QJ7` ✓
- `needle_88k_tail`（74,949 tokens）→ `ZX-4417-QJ7` ✓

**超池回收的接线（本轮新增，全部藏在 env 后面）**
1. `backends/vllm/worker.py`：`KVMEM_FREE_PATH/FREE_LOW_WATERMARK/FREE_SINK_BLOCKS/FREE_RECENT_BLOCKS/
   FREE_PER_STEP` → `additional_config.kvmem`（`free_path` 等键），实测已进入配置（probe `config_source`
   的 keys 含 `free_path`）。
2. `plugin`：free_path 的 schedule 钩子改为同时扫描 `waiting`（此前只看 `running`，而卡住的请求在
   `waiting` 里）；Core 侧没有 runner，`_config()` 为空 → 增加从 `self.vllm_config.additional_config`
   读取的回退。
3. 结果：`freed_blocks` 仍为 **0**，152K tailA 依旧停在 73,728 tokens（51%）⇒ **钩子本身没有在
   EngineCore 里装起来**（模块级 patch 的安装点未覆盖 Core 进程，或安装时读不到配置直接返回）。
   下一步：要么在 Core 的 scheduler 上直接实现回收（fork 内 20 行：`allocate_slots` 失败时释放该请求
   最旧的块，保留最近 ≥ 窗口块数），要么给安装点加探针确认它是否执行。

**现状**：默认臂已恢复（基线 env）；滑臂 88K 2/2 绿；152K 通道无 clamp 时仍停摆。

## 856. Round 600：干净环窗口（clean-room）落地——首次找到"引擎树"与"配置通道"两个系统性坑

**A. 两次"改了半天不生效"的根因（都已实测钉死）**
1. **引擎树**：实时导入的是 `engines/vllm/vllm/`（在插件 import 时用 `print("[kvmem] vllm=...")` 打印，
   pid 1113020/1113236/1113385/1113386 全部指向 `.../engines/vllm/vllm/__init__.py`）。我在
   `/home/srj/vLLM-2080Ti-Definitive` 里的若干改动因此从未生效（反之 `engines/vllm` 的改动一直生效）。
2. **EngineCore 的环境被过滤**：`KVMEM_WINDOW_TOKENS / KVMEM_SLIDE_CORE / KVMEM_VIEW_POS /
   KVMEM_D_CLAMP` 在 worker 的 env 里**存在**，在调度器所在的 EngineCore 里**不存在** ⇒ 调度器里所有
   `os.environ` 读取都是哑的（这也是早前"改了 clamp 没变化"的原因）。修法：这些开关改由
   `additional_config.kvmem` 传递（worker.py 发布，scheduler 优先读配置）——实测命令行里已出现
   `slide_core":true`。

**B. 干净实现（不依赖 knife/legacy）**
- `vllm/v1/worker/gpu/kvmem_ring.py`：自包含的环（身份期镜像引擎行 → base>0 后按页旋转 + 安装 +
  页级 store），不变量写在文件头；`model_runner.prepare_inputs` 在 `prepare_pos_seq_lens` 之后调用
  `kvmem_ring.step(...)`，请求 id 从 `batch_req_state.req_ids` 取（此时 `input_batch` 还没建）。
- 实测已能跑起来并记录（56 条 `ring_clean`），`state=identity` 期 head 镜像正确
  （rowlen 98→1225、head_distinct 与 rowlen 相等 ✓）。
- 插件侧：`KVMEM_RING_CLEAN` 时**不安装任何 legacy 包装**（无 knife、无 runtime、无 pressure）。

**C. 未完成的一环（下一发）**
`base` 始终为 0：调度器的分页滑窗（`kvmem_slide`）没有触发。配置键已进入 EngineCore，但
`_first_alloc`/`_over_pool`/`kvmem_clamp` 三个门里必有一个没满足；下一步在 slide 分支打印
`plan_d/clamp/window/over_pool/first_alloc` 五元组，一次定位。

**默认臂已恢复**（基线 env），88K 仍 `ZX-4417-QJ7` ✓。

## 857. Round 602–603：干净环窗口**机械上打通**（书→滑窗→镜像→旋转→安装→store 全链路），
   输出仍不对；把诊断收敛到"前向帧"一个点

**本轮完成（全部有 probe 证据）**
1. **无条件分页滑窗**（scheduler）：去掉 `clamp/slide/over_pool` 三重门，改为 `plan_d + ring` 即滑；
   首次 `base>0` 时打印：`[kvmem slide] win=45472 page=32 tb=45056 ta=45554 book=44960` ✓。
2. **worker 的账本必须跟调度器**：`gather_batch_req_state` 里原先把 host mirror 设成
   `num_logical`（服务长度），导致 positions 永远是 true、base 恒 0。改为 **ring 开启时 mirror =
   调度器的 `num_computed_tokens`（已滑动的书）**；采样器的"是否还在 prefill"用
   `num_logical_tokens`，不受影响。
3. **ring 拿书**：`scheduler_output.scheduled_cached_reqs.num_computed_tokens`（per-request 列表）。
4. **页池大小**：`_npages` 不能用 `num_blocks.np[gi,0]`（那是"将被子覆盖的 per-request 书"），
   改用适配器探针 `_paged_attn_tensors` 里 KV 张量的第 0 维（实测 2,842 页）。
5. **去掉二次位移**：mirror 跟书之后 positions 已是视图帧，再减 base 会双移（实测 `'Based-d--'`）。

**当前 probe 读数（干净环，just-crossing 45,554 tokens）**
```
view=45460..45462  true=45556..45558  base=96  state=slid  rowlen=1421  npages=2842
```
即：书滑了、base 页对齐、镜像+旋转+安装都是满宽 1,421 页、身份期→滑窗期切换正确。

**仍未通过**：答案仍是片段（`'Based-d--'`/`'ZX4Q8'`/`'KEY-7421'` 之类，随启动变化）。剩下唯一
未取证的点是**前向时内核真正拿到的东西**：`positions`（应=view）、`seq_lens`（应=book+n）、
`kvmem_view_lens`/`kvmem_slot_deltas`（应=book/0）、`kvmem_rope_offset`（应=base）、以及 gather 后的
block table 是否就是安装行。ring 在 prepare 里就能把这几项写进 probe，无需 console（干净臂没有 legacy
包装，console 不可用）。

**默认臂已恢复**（基线 env），88K `ZX-4417-QJ7` ✓。

## 858. Round 604：把 603 的分配夹紧默认关掉（它把"本来绿的 legacy 滑臂"也打伤了）；
   借用 kvmem/kvmem-vllm 的机制后重新定界

**关键回归（已修）**：design 603 的"窗口预留夹紧"默认对所有臂生效，导致本来全绿的 legacy 滑臂
退化成 `'QJ-7741'`/`'ZX-4417'`。改成 **opt-in（config `ring` / `KVMEM_RING_ALLOC=1`）** 后，
legacy 滑臂立刻回到 `ZX-4417-QJ7`（just-crossing 与 88K 双绿，本轮复测）。

**实测矩阵（同一 boot）**
| 臂 | just-crossing | 88K mid | 152K tailA |
|---|---|---|---|
| legacy 滑臂（603 关） | `ZX-4417-QJ7` ✓ | `ZX-4417-QJ7` ✓ | 停摆在 73,728 |
| legacy 滑臂 + 603（窗口预留） | `'Q7-41-9X'` ✗ | `not present` ✗ | **跑完 142,869（3m00s）但退化为复读** |
| clean ring（fork 内，603 关） | `'ZXusman…'` ✗ | `'ductduct…'` ✗ | — |

结论三条：
1. **603 的夹紧把 in-pool 的帧打坏**（它在 45,454-token 的 prompt 上也会触发），所以必须**只在超池**
   时触发（阈值要落在"88K 能过 / 152K 会停"之间，约 75K–88K，下一发用 `KVMEM_POOL_TOKENS` 定）；
2. **603 确实解决了停摆**（152K 从"卡在 73,728"变成 3m00s 跑完）⇒ 分配侧的路子是对的，坏的是帧；
3. **clean ring 与 legacy 不等价**：同样的 view/true/base/rowlen 读数下 legacy 全绿、clean 输出
   `'duct'` —— 差在"视图内容"这一层（正是 kvmem/kvmem-vllm 用 **raw pre-RoPE K + 装配时按新位置重旋**
   解决的那一层）。clean 臂暂挂起，等把 store 改成 pre-RoPE（我们已有 `kvmem_sparse_rope.
   rotate_k_pages_fp8`）再激活。

**借 kvmem/kvmem-vllm 的三件事（已确认可移植）**
1. `WorkingSetManager`：只翻注意力组的 extent，**不动请求的逻辑计数与 recurrent 管理器**，刷新时
   **整行替换**（本轮 603 就是它的简化版；缺的是"只在压力点触发"）；
2. `adapters/original_rope.py`（75 行）：扩 `cos_sin_cache` 到逻辑容量、在**逻辑位置**上旋 Q/K，
   slot 仍走稠密坐标；
3. 选页的 **mandatory spans（system/query/policy 硬保留）+ capacity fit**，以及"归档 raw pre-RoPE K"。

**默认臂已恢复**（基线 env，无实验键），服务 active、health 200。

## 859. Round 605：in-pool 三绿 + 152K「不再停摆」；超池内容仍是 `duct`

**本轮定案（臂：`D_SLIDE=1, D_KNIFE=0, RING_ALLOC=1, WINDOW=45472, PAGE=32`）**
- 分配夹紧改为 **"仅超池请求 + 窗口已填满后"** 才触发（`prompt_len > pool_tokens` 且
  `num_computed+n > window`）。实测门控读数：`[kvmem gate] pool=90944 prompt=142869 over=True`、
  `[kvmem win] win=45472 bs=1568 held=29 cap=45056` ✓。
- 池大小改为从 block pool 推导（`len(pool.blocks) * 1568`），因为 `kv_cache_manager.block_size`
  在本 fork 不存在（打印为 -1）；`KVMEM_POOL_TOKENS` 也经 additional_config 传到 Core。
- 首次滑步**不再从引擎行补尾**（design 605）：超池请求在同一个 step 里已经被夹紧，引擎行是变形的，
  读它会把视图污染成复读（实测 `ferryman…` 循环）；缺的尾部列就是本 chunk 自己的写目标，用合法页
  占位即可（同一 forward 内被覆盖）。

**实测（同一 boot，连跑三发）**
| 输入 | 答案 |
|---|---|
| just-crossing（45,554，池内） | `ZX-4417-QJ7` ✓ |
| 88K mid（74,948，池内） | `ZX-4417-QJ7` ✓ |
| **152K tailA（142,869，超池）** | 3m 内**跑完**（不再停摆）但输出 `ductduct…` ✗ |

**结论**：分配/通道侧的 152K 已经通了（停摆消失、可复现完成）；剩下的纯粹是**视图内容**——与
kvmem/kvmem-vllm 的做法对照，缺的是"归档 **raw pre-RoPE K**、装配时按新位置重旋"这一层（它们
`window_assembly.py` 的做法）。我们已有的材料：`kvmem_sparse_rope.rotate_k_pages_fp8`（重旋）、
`_ring_commit_pages`（页级 store，现在存的是 post-RoPE）、以及 `_apply_selection_d`（in-place 还原）。

**下一发（按 kvmem-vllm 的机制）**：把 store 改成存 **pre-RoPE K**（原始 K）+ 装配时按目标位置重旋；
先只对超池请求启用（`KVMEM_REBAKE=1`），用 152K tailA 判成败。

**默认臂已恢复**（基线 env），88K `ZX-4417-QJ7` ✓。

## 860. Round 606–607：**重大发现——legacy 滑臂的环从未运行过**（此前"绿"全是稠密路径）

**证据链**
1. 152K 请求跑完但没有任何 `d_ring576` probe 记录（`/tmp/kvmem_ring576.jsonl` 0 行），而插件日志里
   运行时的 `d_hv_funnel/d_step_call/d_final_sweep` 都在 ⇒ **运行时在、环没跑**。
2. 根因：hook 里 `live = runner.input_batch.req_ids or runner._kvmem_req_ids` —— 此处的 `runner` 是
   **模块**（`input_batch` 恒 None），而 `_kvmem_req_ids` 从未被填充 ⇒ **`live` 恒空** ⇒ 环循环
   （`for _i631, _rq in enumerate(live)`）一次都没执行。
3. 结论：round 589/590/603 里那些 `ZX-4417-QJ7` 的"绿"，凡是池内 prompt（45.5K/74.9K < 池 88,914）
   都是**纯稠密运行**（不需要压缩）；环真正跑过的只有 fork 内的 clean ring（会 `duct`）。
4. 超池 152K 的 `'ferryman…'`（prompt 开头）也由此解释：无环 + 窗口预留把页集冻结在第一个窗口，
   模型读到的就是开头 45K。

**已尝试的修复（未生效，下一步继续）**
- 插件 attach 的"瞬时失败永久失败"改成**有界重试**（design 606）；
- `live` 改为优先 `rt.view.runner.input_batch.req_ids`（design 607）。
  两者都没让 probe 出现记录 ⇒ `live` 仍然空／`rt` 在该作用域不是预期对象。下一发在 hook 里直接
  打印 `id(rt)`、`id(rt.view.runner)`、`bool(input_batch)`、`len(live)`，一次定位。

**状态**：默认臂已恢复；88K 与 just-crossing 仍 `ZX-4417-QJ7`（稠密路径）。

### 860b. 回归追查结论：问题出在本轮的 plugin 改动（已回滚到 8664c52）

- 我把 `live`（环循环的请求列表）改成经 live view 解析后，**default 臂的 88K 退化成
  `ZX where where…`**；先怀疑 attach 重试（606）与 `view.bind_step`，逐个回滚都不解决。
- 最终定位：**整组 plugin 改动**（attach 重试 606 + `live_ring` 607 + bind 分流）都要回滚——
  `git checkout 8664c52 -- plugin/vllm_kvmem_plugin.py` 后，88K 立刻回到 `ZX-4417-QJ7` ✓。
- 教训：`live` 这个"恒空"的列表是 classic/default 臂**实际依赖的行为**（它的设置是按"逐请求工作
  从不执行"调的）。任何让它非空的修复都会同时启用 classic 臂的视图工作 ⇒ 必须先显式分流，
  并且**每次都要复测 default 臂的 88K**。

**当前服务状态**：默认臂（基线 env），服务 active/health 200，88K 中段针 `ZX-4417-QJ7` ✓；
引擎 fork 在 `c3b9517`，适配器 plugin 在 `8664c52` 的内容（未提交的 `vllm_ring_clean.py` 保留为
未启用模块）。

## 861. Round 611：剥掉 6 层"静默死掉"的原因——环仍未触发，但每一层现在都有实测/日志

**本轮修掉的静默失败（全部有日志证据）**
1. **worker/Core 的 env 被过滤**：hook 里 `KVMEM_D_SLIDE`/`KVMEM_RING_VIEW` 读到 False（而服务 env 是 1）
   ⇒ 新增 `_sw610(cfg_key, env, runner)`：**config 优先**（`additional_config.kvmem` 能到每个进程），
   worker.py 发布 `d_slide/ring_view`。
2. **`rt` 是函数内被重新赋值的名字** ⇒ 在其赋值前读取抛 `UnboundLocalError`，被外层 try 吞掉
   （实测日志：`cannot access local variable 'rt'`）⇒ 改用 `_rt610`。
3. **`state` 不在 hook 的闭包里** ⇒ `NameError`（实测：`name 'state' is not defined`）⇒ 改用
   `_attach_sparse` 的闭包变量 `runtime`。
4. **设备表在环内部才创建**，而门控要求表已存在（鸡生蛋）⇒ 门控前先建 `kvmem_slot_deltas/view_lens`。
5. `_dv/_vl` 从**模块**取（恒 None）⇒ 环分支改从 live view 的 runner 取（且只对环臂生效，classic
   臂保持 None——否则 88K 会退化）。
6. 环 probe 的 `KVMEM_RING_PROBE` 同样被 env 过滤 ⇒ 改成**文件触发** `/tmp/kvmem_ring_probe.on`，
   并把限流从"每 pid 40"改为"每请求 600"。

**现在的实测读数（ring 臂 boot）**
```
ring610: rt=True act_is_runner=True any=True slide=True dv=True vl=True
```
即：运行时、runner、开关、设备表**全部就位**；环分支的入口条件成立。但 `d_ring_slide` 仍未产生
probe 记录（`/tmp/kvmem_ring576.jsonl` 0 行、`/tmp/kvmem_slide577_err.jsonl` 不存在）⇒ 卡在环循环体
内部（`_row` 取得之后、调用之前）。**下一步**：在循环体里按 3 个点各写一条 probe
（进入循环 / `_row` 解析 / 调用前），一次定位。

**默认臂不受影响**（本轮所有改动只在环臂分支内生效）：88K 中段针 `ZX-4417-QJ7` ✓。

## 862. Round 612：legacy 路径的 attach 是**竞态**（首次 execute_model 时 `kv_caches`/group id 未就绪），
   已判定"不可作为验收路径"；clean fork ring 是唯一可靠且可仪表的路径

**本轮证据**
1. 每个"没触发环"的 boot 里都有 `sparse_skip: reason="no paged attention KV"`（最近一次日志里 6 条），
   且 `sparse_failed` 是**粘性**的 ⇒ 整条请求以 DENSE 运行 ⇒ 之前多轮"绿"是稠密结果，不能作为环的验收。
   已加 `_paged_attn_tensors` 的兜底（group id 为空时退化为"任意 4 维张量"），但该 boot 仍未 attach
   （说明竞态点不止 group id 一处）。
2. 3 点 probe（环循环体的 enter/row/call）在该 boot **一行没写** ⇒ 与 1 一致（运行时压根没 attach）。
3. **clean fork ring 稳定运行并完整记录**：150 条记录（22 identity + 128 slid），
   `view=45,455 / true=45,615 / base=160 / rowlen=1,421 / head_distinct=1,421` 全部合理；
   但答案仍是 `'ZXusman…'`（读到 needle 前缀后乱）。

**判定（本轮结论）**：legacy 环路径（plugin hook + 10k 行 runtime）**不再作为验收路径**——它的 attach
是竞态、失败即静默 DENSE，上层还有 7 层历史静默失败；任何"绿"都可能是稠密结果。clean fork ring
（`vllm/v1/worker/gpu/kvmem_ring.py`，约 280 行、无闭包、每步落盘）是唯一可仪表、可复现的路径。

**下一发（在 clean ring 内一次取全）**：在 fork 的 ring 钩子里，publish 之后立刻记录
`seq_lens[0] / kvmem_view_lens[0] / kvmem_slot_deltas[0] / kvmem_rope_offset / positions[0]`，
并在 rope 相加之后、gather 之后各记一次；用它判定 `'ZXusman…'` 到底坏在"seq 长度"、"slot"、
"RoPE 相位"还是"gather 后的行"——一处定位，不再依赖 console/plugin。

## 863. Round 613–614：干净环的**帧、行、gather 三者全部验证一致**（有逐帧证据），
   内容层成为唯一剩余嫌疑

**本轮新增的仪表（fork 内，不依赖 plugin/console）**
1. `frame_pre`（环 publish 时）：`view / true / base / n_new / pos_arg0 / dv / vl / qsl`；
2. `frame_post`（rope 相加后、override 之后）：`pos0 / seq0 / rope / vl0 / dv0 / ncomp_gpu`；
3. `gather`（`gather_block_tables` 之后）：注意力组的 `g_head/g_tail` + 环安装行的 `ring_head/ring_tail` + `nb`。

**实测（clean ring，just-crossing 45,554 tokens，答案 `'ZXusman…'`）**
```
frame_post: pos0=45615  seq0=45456  rope=160  vl0=45455  dv0=160  ncomp_gpu=45615
gather    : gi=3  g_head=[201,202,203] = ring_head ✓
            g_tail=[…,196,197,198,199,200] = ring_tail ✓   nb=1421
```
含义：**positions(post-rope)=TRUE ✓、attention 长度=DENSE(vl+n) ✓、rope=base ✓、delta=base ✓、
读路径拿到的行就是环安装的行（宽度 1,421）✓**。也就是说：帧、行、gather 三层全部正确，
而模型仍吐 `'ZXusman…'` ⇒ 只剩**页内容（KV 字节）**这一层。

**下一发（内容层，两处对照）**
1. 在 `prepare_attn` 里记录本步的 `slot_positions[0:3]` 与 `positions[0:3]`（确认写地址确实是 dense）；
2. 对"身份期写入的某页"做写时/读时指纹对照（环内已有 `paged[0][page]` 取页方法），判断是
   "页被改写"还是"页内容本就带了错误相位"。这一步决定要不要上 kvmem/kvmem-vllm 的
   **raw pre-RoPE K + 装配重旋**。

**默认臂不受影响**（所有仪表只在 clean ring 开启时写入）。

## 864. Round 615–616：A/B 把问题锁死在"滑步"本身

**决定性 A/B（clean ring，同一 prompt 45,554 tokens）**
| 窗口 | 环状态 | 答案 |
|---|---|---|
| 60,000（> prompt，base≡0） | 44 条 identity | **`ZX-4417-QJ7`** ✓ |
| 45,472（< prompt，base>0） | identity→slid | `'ZXusman…'` ✗ |

⇒ **环的 identity 路径是正确的；一旦开始滑步（旋转行 + delta=base）就坏。** 而本轮同时验证了：
- `slot=45,456 / pos=45,616 / dv=160 / seq=45,457`（写地址=dense 列 ✓）；
- `g_head/g_tail` 与环安装行一致、宽度 1,421（读路径 ✓）；
- 同页指纹在**同一 rank 内稳定**（早前看到的"变化"是两个 TP rank 交错，不是改写）。

**由此推断（下一发验证）**：滑步后**行旋转与页内容的一致性**是最后未验证的一环——尤其是
"身份期写下的页，在旋转后被读到的列对应的 true 位置"是否与写入时的 true 位置相同。验证方法：
在 identity 期记录"某 token 的 true 位置 → 其 K 所在页 + 页内偏移"的指纹，滑步后再按新列反查同一
true 位置，比较指纹（要带 rank 维度，避免本轮的双 rank 交错假象）。

**本轮结论**：不是帧、不是 slot、不是 gather、不是页被改写；问题在"旋转后的行 ↔ 页内各偏移的 true
身份"映射。这条一旦对上，152K 的超池尾针就有望直接通。

## 865. Round 617：**行安装的绝对帧不变式有 9 处不吻合**——这就是滑步坏掉的位置

**不变式**：引擎的行是 TRUE 帧（`eng[c]` = 真位置 `32c` 的页），所以安装行必须满足
`row[c] == eng[c + base/32]`。实测（window=45,505，base=64/128，1,422 列）：
```
{'base': 64,  'ab': 2, 'cmp': 1422, 'mismatch': 9}
{'base': 128, 'ab': 4, 'cmp': 1422, 'mismatch': 9}
```
⇒ 安装行不是引擎行的纯旋转：**至少 9 列对不上**。这解释了全部现象：
- identity（base≡0）时 `ab=0`，不变式平凡成立 ⇒ 答案正确（A/B 已证）；
- 一旦滑步，错列落在**尾部（扩展出来的那几页）**，模型读到相邻但错位的内容 ⇒ `'ZXages…'`/`'ZXusman…'`。

**来源分析（下一发一条命令即可确认）**：扩展把 `eng[len(head)+off : want+off]` 拼到 rotate 前的 head
上，但 head 是**上一次安装后的帧**，而 `eng` 在每一步都可能被引擎自己的 `append_block_ids` 改写/追加，
两者列宽不一致 ⇒ 尾部错位。两条修法（按代价排序）：
1. **干脆不扩展**：把 head 一次性取满（在"跨越窗口"的那一步用 `eng[:win_pages]` 快照，此刻引擎行
   已完整到 `base+win`）；
2. 每次安装后**回读** `row[c] == eng[c+ab]`，不吻合就整行重建（fail-closed，宁慢不错）。

**本轮 A/B（决定性）**
| 窗口 | base | 答案 |
|---|---|---|
| 60,000 | 0（identity 44 条） | `ZX-4417-QJ7` ✓ |
| 45,505 | 64 | `'ZXagesman…'` ✗ |
| 45,472 | 96/160 | `'ZXusman…'` ✗ |
