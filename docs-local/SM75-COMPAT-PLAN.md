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

## 3. 真正的硬约束：SM75 与上游 GPU 路径不兼容

- 上游 `native_attention_sources.json` 固定的是 **FlashInfer FA2 + XQA**（`mha_sm90.cu`、`mla_sm120.cu` 等），support matrix 明确测试机型是 **Blackwell（SM120）与 A800（SM80）**，并要求 **FP8 E4M3 KV 原生布局**。
- 2080Ti = **SM75**：没有 FP8 张量核心，也没有 FA2/XQA 内核。
⇒ 上游的 “attention + KV 存储” 两层必须换成 SM75 可用实现；**宿主核心（选择/归档/规划器）可原样使用**。

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

**P3 attention 路径（SM75）**（3–5 轮，最重）
- 禁用上游 FA2/XQA JIT 内核（`native_attention_sources.json` 那条链），改为：
  - 用 fork 自带的 SM75 可用 attention（FlashAttention-2 的 SM75 分支 / Triton 后端）；
  - KV 页的“选中视图”安装 + RoPE 相位处理沿用旧适配器验证过的做法（页按 rank 旋转、稠密索引）。
- 关键契约（已从上游 `core/memory.py` 抄到）：帧必须是**有序稠密**、`history_end == checkpoint.position`、**K 和 V 都要重载**、GDN/draft 状态必须一起恢复。

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
