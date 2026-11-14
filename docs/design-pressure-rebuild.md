# 压力压缩 + 重建对齐协议（设计草案）

状态：**未实现**。本文件定义"物理池 < prompt 长度"时必须存在的那套协议，
供后续独立工作项施工。触发原因见 `docs/design-plan-d.md` §938（代码自述与实测）。

## 1. 问题陈述

- 池内（pool ≥ prompt）：`prefill_pressure` 与 `compact_rebuild` 各自都能工作，
  实测 88K/152K/250K 全 `ZX-4417-QJ7`。
- 超池（pool < prompt）：必须压缩；但本版本对"压力压缩后的重建"是 **fail-closed**：
  `src/kvmem_sparse_runtime.py:441-448` 明确写"Aligning rebuild with a pressure-compacted
  request is a separate, not-yet-designed protocol"。
- 实测表现：读到 prompt 开头文本，或请求挂起（`_rebuild_select` 从未出现，只有
  `pressure_rebuild_queued`）。

## 2. 需要定义的三件事

### 2.1 坐标系所有权
- **原始坐标系**（original frame）：prompt 的真实 token 序号 `0..P-1`；核心打分器、needle 定位、
  `_rebuild_select` 的产物都应在这个坐标系里表达。
- **视图坐标系**（view frame）：压缩后设备页/列的稠密序号 `0..W-1`；`block_table` 行、
  slot 映射、`seq_lens` 属于这个坐标系。
- 规则：
  1. 任何"选页结果"从视图坐标系进入原始坐标系前，必须**带上它所属的 base**（`base_view`，
     即压缩时被丢弃的前缀长度），否则一律拒绝（这就是 v12 engine death 的来源）。
  2. 反向（原始 → 视图）只允许在 `base` 单调不减的前提下进行；回退（rewind）不允许
     （与"无 rewind/无 child"约束一致）。

### 2.2 阶段与失败语义
| 阶段 | 谁拥有 | 不变量 | 失败时 |
|---|---|---|---|
| prefill 分块写 KV | 引擎 | 每块写地址 = 原始位置（未压缩时） | 拒绝压缩，请求排队 |
| 水位触发压缩 | runtime | 工作集 ≤ budget+gen_reserve；被换出页进 store 且可读 | 不压缩，等待解码结束 |
| prompt 结束重建 | runtime | 只重建一次；重建期间该请求禁止投机；重建后的 KV 覆盖完整工作集 | 回落到"读开头"并**显式记录失败**（不得静默答错） |
| decode | 引擎 | `seq_lens` = 视图长度；RoPE 相位 = 原始位置 | 中止请求并记录 |

### 2.3 单趟保证
超池请求**必须**在同一请求 id 内完成"分块 prefill → 压缩 → 重建 → decode"，不得依赖
抢占/重算（抢占会使窗口页被后续趟覆盖，本轮已多次观察到该现象）。

## 3. 施工顺序（每步都要能独立验证）

1. **坐标系断言先行**：在 `_rebuild_select` 的输入输出上写断言（选页必须落在
   `[base_view, base_view + W)` 映射回原始坐标后的范围内），用池内用例（0.972）先跑通，
   确保断言不影响已通过的长上下文电池。
2. **超池最小余量**：`GPU_UTIL` 调到池 = prompt + 1 块（约 1,568 token），验证
   "压缩 1 次 + 重建 1 次"能答对；观察 `pressure_rebuild_queued` 之后必须出现
   `rebuild_select`（当前从未出现）。
3. **余量扩大**：池 = prompt/2，验证工作集恒定（resident 页数不随 prompt 增长）。
4. **回归**：池内 88K/152K/250K 电池、A/B 锚、离线 8/8。

## 4. 验收（与目标一致）

- 池 < prompt 时：88K、152K（mid/tailA/tailB）全部答出 `ZX-4417-QJ7`；
- 同一 boot 连续多请求（≥3）保持正确（本轮已证明跨请求污染问题存在过）；
- HTTP 200 不算通过；必须检查答案文本。
