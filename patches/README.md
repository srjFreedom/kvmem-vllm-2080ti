# SM75（双 2080Ti）补丁集

## 引擎侧：**原版，无任何补丁**

运行时 = `/home/srj/vLLM-2080Ti-Definitive` 分支 **`sm75-post3-clean`** @ `93c9013e375b1ecbf2da88179d2a31175a0d5c99`
（= 上游 `v0.2.2-post3` 标签本体，**post3 之上 0 个提交**）。

2026-11-16 实测：把旧项目遗留的 20 个 kvmem 提交全部去掉后，
上游 kvmem 插件**照常启动并答对**（短请求 `OK-2080TI`；4,888-token needle `ZX-4417-QJ7`，9.5s）。
依据：新项目对所有 fork 专有字段（`num_logical_tokens` / `kvmem_view_lens` /
`kvmem_rope_offset` / `num_kvmem_offset` …）的引用数都是 **0**。

旧的 `kvmem-0.2.2-clean` 分支与 20 个提交仅作历史参考（备份
`/home/srj/vllm-fork-2080ti-20261116.bundle`），**运行时不再使用**。

## kvmem 侧：一个补丁

```
0002-kvmem-sm75-compat.patch      # 本仓库 git diff 8654700..HEAD
```

6 个文件：
- `compat_sm75/__init__.py`（新增）：block table 形状兼容 + capture runner 晚绑定
- `core/native_pages.py`：只有设备行的 block table 按恒等扩展
- `serving/serve.py`（+ `DEFAULTS`）：`--dtype` 由 profile 决定（SM75 无 bf16）
- `adapters/capture.py`：capture 缓冲 dtype 跟随模型
- `serving/capture_runtime.py`：激活 dtype 接受 fp16
- `local/serve-2080ti-smoke.json`：本机 profile

## 复现

```bash
cd /home/srj/vLLM-2080Ti-Definitive && git checkout sm75-post3-clean      # 原版 post3
cd /home/srj/src/kvmem-vllm-2080ti
uv pip install --python /home/srj/vLLM-2080Ti-Definitive/.venv/bin/python \
  --no-build-isolation --no-deps -e .
/home/srj/vLLM-2080Ti-Definitive/.venv/bin/vllm-kvmem serve \
  --config local/serve-2080ti-smoke.json --run-dir /tmp/kvmem-run-smoke
```

## 已验证（2026-11-16，原版引擎）
| 请求 | prompt_tokens | 答案 | 耗时 |
|---|---|---|---|
| 短（OK-2080TI） | 63 | `OK-2080TI` ✅ | — |
| 长 needle（1× select_budget 之上） | 4,888 | `ZX-4417-QJ7` ✅ | 9.5 s |
| **会话续写（同 session_id）** | 4,880 | `ZX-4417-QJ7` ✅ | **0.60 s**（turn1 7.40 s，**12.4×**） |

会话命中证据：`scheduler_resources.json → sessions.hits=2`、`archive.json → session_restored_tokens=4857`。
