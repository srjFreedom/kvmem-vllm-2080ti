# vLLM-2080Ti fork 与 KVMem 的设计与实现（复盘笔记）

> 记录日期 2026-11-10。目的：把两条上游/核心项目的**真实实现事实**固定下来，供后续所有优化（超池、resume、显存分层、选择器）引用。
> 所有结论都标注了出处文件；未验证的部分明确写“未验证”。

## 0. 三个代码位置

| 用途 | 路径 | 说明 |
|---|---|---|
| 干净参照 fork | `/home/srj/src/vllm-2080ti-ref` | SM75 专用 fork 的参考副本（上游 `b23433088b` / v0.29.1rc0） |
| 生产 fork | `/home/srj/vLLM-2080Ti-Definitive`（与 `VLLM-...` 大小写两份并存） | 实际被 systemd 启动的那一份 |
| KVMem 核心 | `/home/srj/src/kvmem-v018rc1`（0.18，llama.cpp/NInfer 后端）；适配器内 `build/kvmem` 为当前使用的版本 | 设计文档在 `docs/architecture.md`、`docs/active-kv-tiering.md` |

---

## 1. vLLM-2080Ti fork：原版怎么设计

### 1.1 定位
- SM75（Turing：2080Ti/T10/T4/TITAN RTX…）专用路线；双 2080Ti 22G + NVLink = 44GB，136 SM。
- 卖点技术栈：Marlin、FlashInfer/FlashQLA、TurboQuant/INT8 KV、MTP/DFlash2、CUDA Graph（见 `README.md`）。
- 目标不是“功能最多”，而是**在该硬件上把吞吐/兼容性做到最好**。

### 1.2 执行模型里最关键的一条事实（决定一切）

写入地址由 `positions` 直接算出：

```
BlockTables.compute_slot_mappings(idx_mapping, query_start_loc, positions, ...)
  -> Triton _compute_slot_mappings_kernel(..., pos, block_table_ptrs, block_sizes, ...)
```
（`vllm/v1/worker/gpu/block_table.py:191`、`:277`）

即：**RoPE 的“位置”与写缓存的“地址”共用同一个张量 `positions`**。
- 因此“保持逻辑位置、只压缩写地址”在**不改 kernel** 的前提下必须把这两个用途拆开；
- 我们过去所有“改行 / 改 seq_len / 改 override”的尝试，本质都在绕这条约束。

### 1.3 注意力/缓存的可扩展点

- KV cache 规格是插件式的：`vllm/v1/kv_cache_interface.py` 内有 `HiSparseHotSpec`、`HiSparseResidentSpec`、`MLAAttentionSpec`、`UniformTypeKVCacheSpecs` 等。
- `vllm/v1/hisparse/`：**resident(GPU) + host(pinned mmap) + hot suffix** 的数据面。
  - `runtime.py`：`HiSparseCacheHandle`、`HiSparseHostPool`、`initialize_hisparse_runtime_buffers`、`pin_tensor`、`SharedOffloadRegion`；
  - `binding.py`：把 HiSparse cache 绑到 attention layer（`resolve_hisparse_block_size` 等）；
  - `block_pool.py`、`coordinator.py`、`layout.py`（`HISPARSE_HOT_SUFFIX` / `HISPARSE_RESIDENT_SUFFIX`）。
  - **限制**：稀疏注意力后端目前只存在于 MLA 路径（`backends/mla/flashinfer_mla_sparse.py`、`flashinfer_mla_sparse_sm90.py`），GQA/FlashInfer/Triton 没有 per-slot 稀疏。Qwen3Next(GQA+GDN) 不能直接套。
- `vllm/v1/kv_offload/`：完整的**块级 offload 框架**
  - `OffloadKey` / `Medium` / `Locality` / `TierMatcher` / `OffloadPolicy` / `LoadStoreSpec` / `PrepareStoreOutput` / `OffloadingManager`；
  - 子目录 `cpu/`（`shared_offload_region`）、`tiering/`；`simple_kv_offload/` 提供 `copy_backend.py`、`cuda_mem_ops.py`、`disk_backend.py`、`manager.py`。
  - 定位：块哈希为键的 host/disk 分层 + 按需 load/store。
- 混合模型：`attention/backends/gdn_attn.py`、`mamba*_attn.py`、`linear_attn.py` 均为一等后端 → GDN 循环态在本 fork 有原生支持。

### 1.4 由此得到的判断
fork 已经具备：块池、host 分层、块拷贝原语、插件式 KV 规格、会话标识（`session_id` 流入 `Request.session_id`）。
**唯一缺的结构性能力**：把“注意力位置(真)”与“写地址(压缩)”拆成两个输入。

---

## 2. KVMem：它是什么、为什么存在

出自 `kvmem/docs/architecture.md` 与 `docs/active-kv-tiering.md`：

1. **定位**：block-sparse KV **工作集管理器**（不碰推理、不改 Flash Attention）。
2. **GPU 注意力缓存 = 有界块槽池**，大小 `budget + gen_reserve`；每个逻辑块占 `block_tokens` 个 cell。
3. **cell 的 `pos` 是原始单调 token 位置；slot 下标不是 RoPE 坐标。**
4. **恢复 = 打包 GPU 格式的 memcpy 到该 orig pos**（不做 raw-K 反旋转、产品路径不 re-RoPE）。
5. **检索分 = mean-K（F32，pre-RoPE，首次写入时捕获）**；块写满时异步把打包 K/V 拷到 host；淘汰时若已有副本则跳过拷贝。
6. **reselection 是 `KvMemPlan` 的 diff**：仍被选中的块留在原槽，只把要淘汰的 cell `seq_rm` 掉。
7. **两层 host**：H（RAM，限制本 worker 活/非活历史的原生 KV 字节 + 预留）、D（磁盘，限制被持有的文件区间）；同一份原生 KV 只有一套所有权；写入先取配额、完成后才发布新副本、释放旧副本。
8. **会话复用键 = token LCP**；检索 query 取最后一段 `role=user`。
9. **循环状态（GDN）= 整请求字节快照**，按请求恢复；不按块搬。
10. 支持多会话（`--kvmem-conversations N`）：切换时整工作集 drain 到 host、再重灌另一会话；请求仍串行，`n_seq_max` 保持 1。

**好处**：显存只需容纳“这一轮真正要看的块”，其余按真位置打包躺在 host/disk，用 mean-K 找回真正相关的块 → 长对话/超池上下文可行，且恢复是纯拷贝、可预测、可并行。

---

## 3. 两者拼接的方案（本轮分析结论，作为后续实现的基线）

### 3.1 三层分工

| 层 | 归属 | 职责 |
|---|---|---|
| 推理引擎 | fork | token 流、**真位置**、采样、CUDA Graph、kernel |
| 视图层 | fork 新增薄接口 | `block_table` = 逻辑块→池槽；**RoPE 用真位置，slot mapping 用压缩下标** |
| 管理层 | 移植 KVMem | mean-K 选择、host 存储、stage-in/out 计划、会话/状态快照 |

### 3.2 唯一必须动的引擎改动（小且可回归）
- 现状：一个 `positions` 同时喂 RoPE 与 slot_mapping；
- 目标：`positions_rope`（真位置）与 `positions_slot`（压缩下标）分开；
- 做法：worker 里多传一个张量，`compute_slot_mappings` 改用 `positions_slot`，模型仍拿 `positions_rope`；**kernel 不动**。

### 3.3 为什么“压缩地址 + 真相位”是对的
- K 写入时已带真位置 RoPE 相位；
- attention 只按行内顺序做因果掩码与长度计算，**不用行下标作为相位**；
- 所以“丢块 = 从行里移除 + 下标压缩”即可，无需 re-RoPE、无需重算前缀。

### 3.4 host 存储复用 fork 的 `kv_offload`
- 淘汰：池块 → host（键用“逻辑块号”，稳定身份，不用哈希链）；
- 恢复：host → 池槽纯拷贝；
- 分层优先复用 `kv_offload/cpu` 与 `tiering`，**不再自造 store/账本**（过去那套正是错配来源）。

### 3.5 会话与循环态
- 会话身份 = vLLM 原生 `session_id` + token LCP；
- GDN 状态 = 整请求一份快照，恢复到同一槽；**resume 的 skip 深度必须与快照 cursor 同源**（否则会重演 141120 vs 72128 的错位）。

### 3.6 落地顺序（每步独立验收）
1. **positions 拆分**（不开任何 KVMem 行为）→ plain 88K/152K/250K 结果必须不变；
2. **有界池 + 满行**（未选块用零块占位，不丢块）→ 先验超池 250K 能答对；
3. **丢块 + 压缩下标**（sink + recent + mean-K top-k）→ 验 250K needle + 显存下降；
4. **host 分层**（接 `kv_offload`）→ 验冷/热两轮 + resume 5–10s；
5. **会话/状态快照** + 离线全套 + A/B 锚。

### 3.7 与旧做法的对照（为什么旧路必败，勿重蹈）
| 旧做法 | 失败原因 |
|---|---|
| 真位置 + 全长度行，只改行宽 / 发 `seq_override` | 行只覆盖视图宽 → 注意力越界读旧块（140K 红） |
| 压缩 `seq_len` 但不拆 positions | RoPE 与写地址仍绑定，相位错 |
| 自维护 `st.row` / `st.logical` / `seq_override` 三本账 | 永远对不齐（`st.logical` 全 -1 即证据） |
| 伪造 adoption 账 / keepalive 桥 | 绕过协议，必然错配（曾全局 duct） |

---

## 4. 风险与未验证点
- `HiSparse` 为 MLA 专用，GQA/SM75 不能直接用（未验证有无 GQA 变体）；
- `kv_offload` 以块哈希为键，与“按真位置块号”需要适配层（未验证改造成本）；
- GDN 状态与 resume cursor 同源是硬要求，尚未实现；
- “positions 拆分后 plain 结果字节不变”尚未实测（下一步验证目标）。
