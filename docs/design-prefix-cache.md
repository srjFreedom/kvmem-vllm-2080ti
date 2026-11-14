# 显式前缀缓存（prefix cache）重写设计

目标：同一 prompt 的第二次请求（**不带任何 session 字段**）与冷跑输出一致，并且更快。
规格来源：上游 `kvmem-vllm` 的 `docs/prompt-sessions.md`（身份=内容摘要、发布=封存前缀、恢复含
GDN 状态、恢复前零化 native 页并在 attention metadata 之前完成）。

## 0. 为什么重写而不是继续修

已实测（见 `docs/design-plan-d.md` §961–§968）：
- legacy 复用树有 **≥5 条 adoption 路径** 与 **≥3 处恢复调用**，逐条加闸门后仍有一条未被拦住；
- 正面事实：**跳过 prefill 但不恢复页 = 输出正确**；指纹证明存量页/目标页/字节都对 ⇒
  坏的是"某处恢复把前缀字节写到正在使用的页上"；
- 因此：**恢复入口必须是唯一的**，页归属必须显式，发布即合格（含状态）。

## 1. 模块

`src/kvmem_prefix_cache.py`（已实现，CPU 契约测试 `tests/prefix_cache_test.py` 全绿）

- `block_hash_chain(ids, block_size)`：块哈希链，身份唯一来源；
- `PrefixCache.publish(...)`：**只**接受 `compacted_in_prefill=False`、`state is not None`、
  逻辑页恰好 `0..covered/pt-1` 的记录，否则抛 `PrefixCacheError`（拒绝即语义）；
- `PrefixCache.lookup(ids, block_size)`：链前缀匹配 + 前缀 digest 复算 + 状态在 + 页完整，
  四条同时满足才返回记录；
- 记录内的 `pages` 是**逻辑页**（第 i 页 == token `[i*pt,(i+1)*pt)`），`state` 是 GDN 快照。

## 2. 接入点（只有两个）

1. **发布**：请求正常结束（`finish_reason` 为正常终止）且该请求 **prefill 未被压缩** 时，
   在运行时里把 `(ids, block_size, page_tokens, pages, state)` 交给 `PrefixCache.publish`；
   压缩过的请求走 `compacted_in_prefill=True` 分支 ⇒ 必然被拒 ⇒ 不发布。
2. **接受**：请求进入调度前，`lookup(ids, block_size)`：
   - 命中 ⇒ ①按序把逻辑页 i 恢复到本请求第 i 页；②恢复 GDN 状态；③把引擎的
     `num_computed_tokens` 设为命中长度；④日志 `prefix_hit {covered, pages}`；
   - 未命中 ⇒ **完全不触碰**引擎账本与页（冷跑）。

关键约束（本轮教训）：**②③必须在①成功之后**；任一步失败 ⇒ 整条请求冷跑，
不允许"跳过了但没恢复"。

## 3. 要删除/停用的旧路径

全部由 `KVMEM_PREFIX_REUSE` 之外的开关关闭，重写验收通过后整段删除：

| 位置 | 说明 |
|---|---|
| `plugin/vllm_kvmem_plugin.py: _maybe_resume` | 调度器侧跳过（改为向 `PrefixCache` 查询） |
| `src/kvmem_sparse_runtime.py: _adopt_session` | runtime adoption（含 self-derive、伪造 entry） |
| `_ensure` 的 adopted 分支 | 借用他人 orchestrator |
| `d_keepalive_graft` 单幸存者桥接 | 猜测续接 |
| `_complete_resume` | 旧的恢复入口（由新入口替代） |
| `_spill_stores` / `_session_hashes` 认领 | 旧的跨请求 store 借用 |
| `KVMEM_SESSION_REUSE` / `KVMEM_SESSION_PARK` | 语义合并进 `KVMEM_PREFIX_CACHE` |

## 4. 验收

1. `tests/prefix_cache_test.py` 全绿（已达成）；
2. 30K prompt（≤B）：cold 与 warm 的 greedy 输出**逐 token 相同**，warm 明显更快；
3. 47K/88K prompt（>B，压缩）：warm **不允许发布**记录 ⇒ 与冷跑一致（不退化）；
4. 变更前缀（末尾改一个 token）⇒ 未命中、冷跑、输出与各自冷跑一致；
5. 每次命中/拒绝都有日志：`prefix_hit` / `prefix_miss {reason}`。
