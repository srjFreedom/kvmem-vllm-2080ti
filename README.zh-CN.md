# 在双 RTX 2080 Ti（SM75）上跑 KVMem

KVMem 的核心思路：**GPU 上只保留有界的 KV 窗口，全量历史放在主机内存**，每次请求按需把相关历史选回窗口。
这样，显存放不下的长上下文也能服务；重复请求或续写请求则几乎完全跳过 prefill。

本仓库是**在别人成果之上打的补丁**（见文末[致谢](#致谢)）：
机制来自 [`kvmem/kvmem-vllm`](https://github.com/kvmem/kvmem-vllm)，引擎是可跑 SM75 的 vLLM fork，
我们补的是兼容层、部署 profile，以及两张 22 GB 图灵卡上的实测配置。

*English (default): [README.md](README.md)*

---

## 运行结构与我们的改动

```
┌──────────────────────────────────────────────────────────────┐
│ 机制层：kvmem-vllm 插件（VLLM_PLUGINS=kvmem）                 │
│   调度器、KV 连接器、归档、会话、请求规划                      │
├──────────────────────────────────────────────────────────────┤
│ 引擎层：vLLM 0.2.2.post3 SM75 fork                            │
│   模型执行、FlashInfer attention、Marlin FP8、MTP、CUDA graph │
└──────────────────────────────────────────────────────────────┘
```

| 层 | 来源 | 我们的改动 |
|---|---|---|
| `src/vllm_kvmem/**` | 上游 `kvmem/kvmem-vllm` | SM75/float16 兼容、自动 session id、页池需求按 guided 块计费、miss 诊断 |
| 引擎 | `vLLM-2080Ti-Definitive`（`v0.2.2.post3` + SM75 工作）| **仅一个 15 行补丁**：KVMem 有界窗口时允许 `max_model_len` 超过 KV 池 |
| `local/`、`deploy/`、`scripts/` | — | 服务 profile、看门狗、启动器、systemd 单元 |
| `patches/` | — | 两个补丁原文 + 基线 commit |

## 真机验证结果

宿主：2×RTX 2080 Ti 22 GB（SM75）、62 GB 内存、27B FP8 混合模型（16 层 full attention + 48 层 GDN，
仅权重量化 FP8、`lm_head` 为 BF16），TP=2、`dtype=float16`、`kv-cache-dtype fp8_e4m3`。

| 能力 | 结果 |
|---|---|
| 代码类生成 decode | **80–87 tok/s**（同配置原版引擎 80.8–87.2）|
| prefill | 1,021–1,063 tok/s（原版 1,174）|
| 4.8K needle | 答对，冷跑约 7.9 s |
| **prompt 超过物理池** | 218,220 token 答对；重复请求 **1.36 s**（约 365×）|
| 会话续写（超窗口）| 7.5 s → **0.33 s** |
| 会话续写（窗口内）| 2.4 s → **0.25 s** |
| 超窗口长输出 | 9,000 token，100.5 tok/s，触发 2 次 bounded refresh |
| 多会话交替 | 各自命中（0.25–0.40 s）|
| 并发 | 能准入，但**被串行调度**（每步只跑一个窗口化请求）|

完整命令与证据：[docs/sm75/zh-CN/development-log.md](docs/sm75/zh-CN/development-log.md)

## 硬件要求

- 2×RTX 2080 Ti / Titan（SM75），每卡 22 GB。27B FP8 权重约 32 GB，必须 TP=2；单卡 22 GB 装不下。
- 主机内存：256K 逻辑上下文需要 13–16 GB（见下方容量表）。
- Linux、CUDA 13 驱动、Python 3.12 venv（复用引擎的 venv；本机没有 `pip`，用
  `uv pip install --python …`）。

## 快速开始

```bash
# 0. 环境变量（写进 shell 或 /etc/default/kvmem-vllm）
export KVMEM_ENGINE_DIR=/path/to/vllm-0.2.2.post3-sm75   # 引擎 checkout，含 .venv
export KVMEM_MODEL_DIR=/path/to/checkpoint               # 含 config.json 与权重分片

# 1. 检查环境、打引擎补丁、以 editable 方式安装本插件，并做一次冒烟
cd /path/to/kvmem-vllm-2080ti
./deploy/install.sh

# 2. 启动与验证（轮询 /v1/models、采样 nvidia-smi、检测卡死）
./deploy/run-kvmem-vllm.sh start
./scripts/serve_watch.py status
./scripts/serve_watch.py req --file request.json

# 可选：systemd
./deploy/install.sh --print-unit > /etc/systemd/system/kvmem-vllm.service
systemctl daemon-reload && systemctl enable --now kvmem-vllm
```

仓库里没有任何硬编码路径：profile 用 `${KVMEM_MODEL_DIR}`，启动器从 `KVMEM_ENGINE_DIR`
（或 `KVMEM_ENGINE_VENV`）解析引擎，看门狗最后会退回 `PATH` 上找 `vllm-kvmem`。
`KVMEM_BASE_URL` 可覆盖状态/请求端点。

### 本机配置（仓库保持默认）

本机差异不要写进仓库：在外面放一个小启动脚本，设置路径并用 `--set KEY=VALUE` 覆盖，磁盘上的
profile 永远不动。

```bash
# ~/run-kvmem-2080ti.sh（可复制 deploy/example-local-launcher.sh 修改）
export KVMEM_ENGINE_DIR=/path/to/engine
export KVMEM_MODEL_DIR=/path/to/checkpoint
OVERRIDES=(
  --set gpu_memory_utilization=0.92
  --set kvmem.select_budget=36864
  --set kvmem.gen_budget=4096
)
exec /path/to/kvmem-vllm-2080ti/deploy/run-kvmem-vllm.sh start "${OVERRIDES[@]}" "$@"
```

`--set` 支持顶层键与点号路径（`kvmem.select_budget`），值会按 JSON 解析；未知键直接报错，不会被忽略。

开机自启（systemd，`oneshot` + `RemainAfterExit`，`ExecStop` 负责停掉 GPU 进程）：

```
/etc/systemd/system/kvmem-vllm.service
[Service]
Type=oneshot
RemainAfterExit=yes
User=<你的用户名>
Environment=HOME=/home/<你的用户名>
ExecStart=/path/to/run-kvmem-2080ti.sh start
ExecStop=-/path/to/run-kvmem-2080ti.sh stop
[Install]
WantedBy=multi-user.target
```

`./deploy/install.sh --print-unit` 会按你的用户名和路径生成同样的 unit。
```bash
systemctl enable --now kvmem-vllm     # 立即启动并开机自启
systemctl is-active kvmem-vllm        # active
curl -s localhost:18200/v1/models | head -c 120
```

Profile：`local/serve-2080ti-default.json`（B 36K / R 4K、util 0.92、chunk 2,048）、
`local/serve-2080ti-lean.json`（窗口 49K、util 0.82）、`local/serve-2080ti-200k.json`（窗口 200K、util 0.96）。

## 一页配置口径

KVMem 有四个独立预算，混淆它们是"放不下"的最常见原因。

| 预算 | 参数 | 买到什么 |
|---|---|---|
| **GPU 窗口** | `select_budget`(B) + `gen_budget`(R) | 单请求常驻 KV：选中历史 + 生成预留 |
| **逻辑上下文** | `max_model_len`(CTX) | 单个会话最长能做多长；超出窗口的部分在 RAM |
| **主机历史** | `cpu_read_cache_bytes` + `repository_blocks` | 能留多少 token 历史，**所有会话共享** |
| **会话 checkpoint** | `session_capacity` × `session_checkpoint_bytes` | 能保活多少个会话 |

```
GPU KV 需求       ≈ 17,408 B/token × (B + R) + 0.39 GiB（3 个 GDN 组的固定状态；与上下文无关）
历史 token 总量   ≈ cpu_read_cache_bytes / 17,408     # 每 rank，被所有会话共享
对应主机内存      = 上面数值 × 2                      # TP=2 在同一台机器上有两份 archive
会话数 × CTX      ≤ 历史 token 总量                   # 否则新会话只能排队
```

默认（已实测）：B=36K、R=4K（窗口 40,960）、CTX=256K、`gpu_memory_utilization=0.92`、
`prefill_chunk_tokens=2,048`、`query_replay_budget=4,096`、`session_capacity=1`、
`cpu_read_cache_bytes=6 GiB`。

> `query_replay_budget` 默认等于 R，约 **180 KiB/token** 的显存；不显式设置它，
> 48K 窗口看起来就"永远放不下"。这是最容易踩的坑。

参数全表、与旧适配器（`B/R/CTX/HOST_MIB/SESSIONS`）的对应关系、容量算式：
[docs/sm75/zh-CN/parameter-map.md](docs/sm75/zh-CN/parameter-map.md) 与
[docs/sm75/zh-CN/baseline.md](docs/sm75/zh-CN/baseline.md)。

## 已知边界

1. **并发是串行的**：两个窗口化请求能准入，但每步只调度一个，第二个要等第一个跑完。
   真正的修复需要插件为每个请求维护独立的选片帧/查询捕获流。
2. **rolling 模式**（`rolling_active_window=true`）可用，但只接受 token-ID 形式的
   `/v1/completions`，不兼容 chat；实测吞吐与默认模式相同。
3. **prefix image**（多任务共享前缀）上游拒绝 TP>1，本机不可用。
4. **上下文并不被 262,144 卡死**：启动器对 MTP 允许 `max_model_len` 到 **10,000,000**
   （条件是 `request_planning` 打开且 `B+R ≤ 262,144`）。本机已经用 `max_model_len=524,288`
   成功服务了 **269,654 token** 的 prompt（答案正确）和 299,280 token 的请求。真正的约束是
   **主机内存**（归档）和 **prefill 瞬态显存**，不是上下文上限。模型自身声明 262,144 个位置且
   `rope_type=default`，超出后 RoPE 属外推：能跑，但质量未验证。
5. 会话超额默认**不淘汰**：拿不到归档额度的请求会**排队**。要么按会话数配足
   `cpu_read_cache_bytes`，要么开启 `KVMEM_SESSION_IDLE_LRU_EVICTION=1` 并把
   `session_capacity` 设小。

## 仓库结构

```
src/vllm_kvmem/          插件（机制）
local/                   服务 profile
deploy/                  启动器 + systemd 单元
scripts/serve_watch.py   启动/停止/状态/请求（带 GPU 采样与卡死检测）
patches/                 引擎补丁 + 插件补丁（含基线 commit）
docs/sm75/               英文文档：架构、配置、验证、致谢
docs/sm75/zh-CN/         中文文档：设计日志、基线、参数对照
docs/                    上游 kvmem-vllm 文档（未改动）
```

## 致谢

这个项目完全建立在别人的成果之上，我们是**在它们之上打的补丁**：

| 项目 | 角色 | 许可证 |
|---|---|---|
| [`kvmem/kvmem-vllm`](https://github.com/kvmem/kvmem-vllm) | KVMem 机制本体：调度器、连接器、归档、会话 —— 本仓库是它的衍生 | Apache-2.0 |
| [`vLLM`](https://github.com/vllm-project/vllm) | 模型执行、attention 后端、调度器、MTP | Apache-2.0 |
| `vLLM-2080Ti-Definitive`（私有 fork）| SM75/图灵适配：SM75 上的 FlashInfer、FP8 仅权重路径、全图 CUDA graph 配方、MTP 草稿契约 | Apache-2.0（vLLM 衍生）|
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | paged attention 内核 | Apache-2.0 |
| [Triton](https://github.com/triton-lang/triton) | GDN / M-RoPE / 装配内核 | MIT |
| [Qwen](https://github.com/QwenLM) 模型家族 | 被服务的 checkpoint | 依其模型许可 |

我们的贡献仅限于：`src/vllm_kvmem` 里的 SM75/float16 兼容层、自动 session id、页池需求修正、
诊断探针、一个小的引擎补丁、部署 profile 与验证数据。这里跑得好的部分都是上游作者的功劳，
跑不好的部分是我们的问题。

历史说明：本仓库此前是另一个适配器（同一 fork 上的 KVMEM 入口中间件），那部分工作保留在
`legacy-adapter-v0.5` 分支；`main` 是本文描述的这份移植。
