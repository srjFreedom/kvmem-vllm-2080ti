# SM75（双 2080Ti）补丁集

## 引擎侧：不需要额外补丁

目标运行时是 `/home/srj/vLLM-2080Ti-Definitive` 的分支 **`kvmem-0.2.2-clean` @ `3b871c663f2b276070dc642b59451d0b6238096e`**。
该分支自带旧项目留下的 kvmem 集成提交（scheduler 双账本、`num_logical_tokens` 等），**直接 checkout 即可**。

2026-11-16 曾经试过一个实验提交 `175950def3`（页环 `kvmem_ring.py` + 按页滑动双账本），
实测**不需要**：默认 `KVMEM_RING_CLEAN` 未开，钩子只是打印；删除后服务更快
（同一 4,888-token needle 请求 11.1s → **9.5s**）。已用 `3b871c663f` 回滚，
历史与内容仍在（也备份在 `/home/srj/vllm-fork-2080ti-20261116.bundle`）。

## kvmem 侧：一个补丁

```
0002-kvmem-sm75-compat.patch      # 本仓库 git diff 8654700..HEAD（也在 HEAD 里）
```

覆盖 6 个文件：
- `src/vllm_kvmem/compat_sm75/__init__.py`（新增）：block table 形状兼容 + capture runner 晚绑定
- `src/vllm_kvmem/core/native_pages.py`：只有设备行的 block table 按恒等扩展
- `src/vllm_kvmem/serving/serve.py`（+ `DEFAULTS`）：`--dtype` 由 profile 决定（SM75 无 bf16）
- `src/vllm_kvmem/adapters/capture.py`：capture 缓冲 dtype 跟随模型
- `src/vllm_kvmem/serving/capture_runtime.py`：激活 dtype 接受 fp16
- `local/serve-2080ti-smoke.json`：本机启动 profile

## 复现

```bash
cd /home/srj/vLLM-2080Ti-Definitive && git checkout kvmem-0.2.2-clean   # 引擎侧
cd /home/srj/src/kvmem-vllm-2080ti
uv pip install --python /home/srj/vLLM-2080Ti-Definitive/.venv/bin/python \
  --no-build-isolation --no-deps -e .
/home/srj/vLLM-2080Ti-Definitive/.venv/bin/vllm-kvmem serve \
  --config local/serve-2080ti-smoke.json --run-dir /tmp/kvmem-run-smoke
```

## 已验证（2026-11-16，删掉页环之后）
| 请求 | prompt_tokens | 答案 | 耗时 |
|---|---|---|---|
| 短（OK-2080TI） | 63 | `OK-2080TI` ✅ | — |
| 长 needle（超 select_budget） | 4,888 | `ZX-4417-QJ7` ✅ | **9.5 s** |
