# KVMem on 2× RTX 2080 Ti (SM75)

A serving stack for 22 GB Turing cards: the **bounded KV window** lives on the GPU, the **full history
in host RAM**, and relevant history is selected back into the window per request.

- **Long context on small VRAM** — a 262K-token conversation runs on a 4 GiB KV pool; past the window,
  history is served from the host archive.
- **Sessions stay warm** — per-session checkpoints resume a continued conversation in ~0.3 s instead of
  re-prefilling it, and several sessions can be kept side by side.
- **Concurrency without more VRAM** — independent sessions decode in one scheduling step; refresh and
  archiving stay serialized, so the pool holds one window per step rather than one per request.
- **Two speculative routes on one engine** — MTP4 and DFlash2 k=7 (4-bit W4A16 draft, 1.28 GB), both on
  the compiled-graph path.
- **Repeats are almost free** — a prompt whose cold prefill took 196 s is re-served from the archive in
  1.36 s; a continuation above the window went from 7.5 s to 0.33 s.

The engine is the SM75-capable vLLM fork listed under [Credits](#credits); the KVMem mechanism comes
from [`kvmem/kvmem-vllm`](https://github.com/kvmem/kvmem-vllm). What lives here is the integration: the
SM75 window/session layer, the DFlash2 route and its engine patches, the profiles, and the measured
configuration for two 22 GB Turing cards.

*中文文档：[README.zh-CN.md](README.zh-CN.md)*

---

---

## What's new in 0.6.5

| change | why it matters |
|---|---|
| **MTP defaults to `compiled_graph`** | torch.compile + `FULL_AND_PIECEWISE` beats `full_graph` on the same profile: code **66.8 vs 59.4 tok/s (+12%)**, repeat **103.6 vs 95.2 tok/s (+9%)** |
| **Chat rides the rolling window** | `/v1/chat/completions` used to be rejected with *"Rolling active window requires a token-ID completion prompt"*. The middleware now renders the messages with the served tokenizer and plans the chat request through the same B/R window, so chat no longer bypasses the mechanism |
| **DFlash2 draft support** | DFlash2 k=7 with a **4-bit W4A16 draft** (`syvai/Qwen3.8-27B-DFlash2-W4A16`, 1.28 GB) loads and serves: code **106 tok/s**, repeat **197 tok/s** (KVMem lane), **151 / 254** on the pristine engine. Needs the engine-side DFlash2 patch stack (see below) |
| **`enable_prefix_caching` switch** | the launcher used to hardcode `--no-enable-prefix-caching`; the profile can now ask for APC. The bounded DFlash2 contract still requires it off |
| **Draft page geometry knob** | `VLLM_DFLASH_DRAFT_PAGE_ALIGN=1` makes an APC-free target give the draft attention-sized pages. Measured neutral on the bounded lane, 6 ms/step better on the engine alone |
| **Diagnostics passthrough** | `PER_REQUEST_SPEC_DECODE_METRICS`, `VLLM_LOG_STATS_INTERVAL`, `VLLM_TORCH_PROFILER_DIR` now reach the engine, and `extra_vllm_args` carries argv-level diagnostics such as `--profiler-config` |

Measurement notes for the DFlash2 lane (all on the same box, `B=32K`, `R=8K`, pool 3.5 GiB):

- The bounded window reserves, per admitted request, `3 × (1 + k) mamba state blocks + ceil((B+R)/page) attention
  pages + 5 draft pages + 1 null page`. At `k=7` that is **54 native pages ≈ 3.0 GiB**; **44% of it is the
  mamba state set**, which scales with the draft depth. Dropping to `k=3` releases 12 pages (≈0.85 GiB).
- Against the pristine engine on identical prompts the bounded lane costs **≈8 ms/step**; the scheduler
  accounts for 0.14 ms and the connector hooks for 0.11 ms, so the remainder is launch-side, not logic.
- Pool ceiling with the FP8 target (28.7 GB weights): **4.0 GiB works, 4.5 GiB OOMs at startup**. With the
  NVFP4 target (18.8 GB weights) 5 GiB is reachable.

Engine-side prerequisites for the DFlash2 route (they live in the engine fork, not here):

- `patches/0003-engine-dflash2-w4a16-draft.patch` — [`weicj/vLLM-2080Ti-Definitive#265`](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/265):
  a quantized draft exposes its K/V rows through the quantization kernel instead of a dense `.weight`,
- `patches/0004-engine-dflash2-no-apc-gates.patch` — `VLLM_DFLASH_NATIVE_GROUPS_WITHOUT_APC=1`,
  `VLLM_DFLASH_DRAFT_PAGE_ALIGN=1`, `VLLM_MAMBA_NONE_SMALL_BLOCK=1`,
- `VLLM_KV_CACHE_LAYOUT=BLHNC` — mixed target/draft page sizes (a runtime flag, not a patch).

## What runs, and what we changed

```
┌──────────────────────────────────────────────────────────────┐
│ mechanism: kvmem-vllm plugin  (VLLM_PLUGINS=kvmem)           │
│   scheduler class, KV connector, archive, sessions, planning │
├──────────────────────────────────────────────────────────────┤
│ engine: vLLM 0.2.2.post3 SM75 fork                           │
│   model execution, FlashInfer attention, Marlin FP8, MTP,    │
│   CUDA graphs                                                │
└──────────────────────────────────────────────────────────────┘
```

| layer | source | our change |
|---|---|---|
| `src/vllm_kvmem/**` | upstream `kvmem/kvmem-vllm` | SM75/float16 compatibility, auto session id, page-pool demand in guided-block units, miss diagnostics, **chat through the rolling window**, **`enable_prefix_caching` switch**, **diagnostics passthrough** |
| engine | `vLLM-2080Ti-Definitive` (`v0.2.2.post3` + SM75 work) | **one 15-line patch** (allow `max_model_len` above the KV pool when KVMem bounds the window) plus the DFlash2 route: quantized-draft weight access, native draft groups without APC, and draft page alignment |
| `local/`, `deploy/`, `scripts/` | — | serving profiles, watchdog, launcher, systemd unit |
| `patches/` | — | engine and plugin patches as plain files, with base commits (0003/0004 add the DFlash2 route) |

## Verified on real hardware

Host: 2 × RTX 2080 Ti 22 GB (SM75), 62 GB RAM, 27B FP8 hybrid model (16 full-attention + 48 GDN
layers, weight-only FP8, BF16 `lm_head`), TP = 2, `dtype=float16`, `kv-cache-dtype fp8_e4m3`.

| capability | result |
|---|---|
| code generation decode | **80–87 tok/s** (pristine engine on the same config: 80.8–87.2) |
| **4K cold prefill** | **3.0 s, 1,371 tok/s** (pure sliding, MTP4, FP8, FlashQLA GDN kernel) |
| **200K cold prefill** | **196 s, 1,020 tok/s** (same profile) |
| 4.8K needle request | correct, ~7.9 s cold |
| **prompt above the KV pool** | 218,220 tokens answered correctly; repeat **1.36 s** (≈365×) |
| session continuation, over window | 7.5 s → **0.33 s** |
| session continuation, under window | 2.4 s → **0.25 s** |
| long generation past the window | 9,000 tokens, 100.5 tok/s, 2 bounded refreshes |
| multi-session alternation | each session hits (0.25–0.40 s) |
| concurrency | independent sessions **decode in one step**; refresh and archive stay serial |
| **MTP `compiled_graph` vs `full_graph`** | code **66.8 vs 59.4 tok/s (+12%)**, repeat **103.6 vs 95.2 tok/s (+9%)** |
| **DFlash2 k=7, 4-bit W4A16 draft** | code **106 tok/s**, repeat **197 tok/s** (bounded lane); pristine engine on the same draft: **151 / 254** |
| **Chat over the rolling window** | `/v1/chat/completions` accepted under `rolling_active_window`, planned through the same B/R window |

Details and the exact commands: [docs/sm75/VERIFICATION.md](docs/sm75/VERIFICATION.md).

## Requirements

- 2 × RTX 2080 Ti / Titan Turing (SM75), 22 GB each. The 27B FP8 checkpoint needs TP = 2 (~32 GB of
  weights); a single 22 GB card cannot hold it.
- 16 GB+ host RAM for 256K logical context (see the capacity table below).
- Linux, a working CUDA 13.x driver, Python 3.12, and the engine's virtualenv build from its
  repository (see that repo's own README for its toolchain and build script). The plugin needs
  nothing but the engine venv; `install.sh` handles the patch and the editable install.

## Clean-machine install

One heavy step (building the engine), two leaf steps (the engine patch, the plugin into the
engine's venv):

```bash
git clone https://github.com/weicj/vLLM-2080Ti-Definitive.git    # engine; its README lists the toolchain
cd vLLM-2080Ti-Definitive
git checkout v0.2.2-post3 && ./build.sh                           # produces .venv (uv) + the SM75 kernels
cd ..

git clone https://github.com/srjFreedom/kvmem-vllm-2080ti.git
cd kvmem-vllm-2080ti

export KVMEM_ENGINE_DIR=$PWD/../vLLM-2080Ti-Definitive
export KVMEM_MODEL_DIR=/path/to/checkpoint
./deploy/install.sh              # patch the engine, install the plugin (editable), smoke-test
```

Serve and verify (polls `/v1/models`, samples `nvidia-smi`, detects stalls):

```bash
./deploy/run-kvmem-vllm.sh start
./scripts/serve_watch.py status
./scripts/serve_watch.py params                 # effective window / history / sessions / MTP depth
./scripts/serve_watch.py logs --lines 40        # engine log: $RUN_DIR/serve.log
./scripts/serve_watch.py req --file local/request-example.json

# optional: systemd
./deploy/install.sh --print-unit > /etc/systemd/system/kvmem-vllm.service
systemctl daemon-reload && systemctl enable --now kvmem-vllm
```

### Environment

| variable | meaning |
|---|---|
| `KVMEM_ENGINE_DIR` | engine checkout that contains `.venv` |
| `KVMEM_ENGINE_VENV` | engine virtualenv `bin/` directly (overrides `KVMEM_ENGINE_DIR`) |
| `KVMEM_MODEL_DIR` | checkpoint directory used by the profiles (`${KVMEM_MODEL_DIR}` expansion) |
| `CONFIG` | serving profile (default `local/serve-2080ti-default.json`) |
| `RUN_DIR` | run directory: `serve.log`, `launch.json`, `capture/` traces (default under `/var/tmp`) |
| `KVMEM_BASE_URL` | endpoint used by `serve_watch.py status/req` (default `http://127.0.0.1:18200`) |

### Machine-local configuration

Machine-specific values belong outside the checkout. A launcher script supplies the environment and
passes `--set KEY=VALUE` overrides at start-up, so the committed profile stays the reference default.

```bash
# ~/run-kvmem-2080ti.sh  (copy deploy/example-local-launcher.sh and edit)
export KVMEM_ENGINE_DIR=/path/to/engine
export KVMEM_MODEL_DIR=/path/to/checkpoint
OVERRIDES=(
  --set gpu_memory_utilization=0.92
  --set kvmem.select_budget=36864
  --set kvmem.gen_budget=8192
)
exec /path/to/kvmem-vllm-2080ti/deploy/run-kvmem-vllm.sh start "${OVERRIDES[@]}" "$@"
```

`--set` accepts top-level keys and dotted paths (`kvmem.select_budget`); values are parsed as JSON
when possible and unknown keys are rejected.

Boot the machine with it (systemd, `oneshot` + `RemainAfterExit`, `ExecStop` stops the GPU processes):

```
/etc/systemd/system/kvmem-vllm.service
[Service]
Type=oneshot
RemainAfterExit=yes
User=<your-user>
Environment=HOME=/home/<your-user>
ExecStart=/path/to/run-kvmem-2080ti.sh start
ExecStop=-/path/to/run-kvmem-2080ti.sh stop
[Install]
WantedBy=multi-user.target
```

`./deploy/install.sh --print-unit` renders the same unit with your user and paths filled in.

```bash
systemctl enable --now kvmem-vllm     # start now and on every boot
systemctl is-active kvmem-vllm        # active
curl -s localhost:18200/v1/models | head -c 120
```

Profiles: `local/serve-2080ti-default.json` (B 36K / R 8K, util 0.92, chunk 2,048, FlashQLA GDN kernel), `local/serve-2080ti-lean.json` (49K window, util 0.82), `local/serve-2080ti-200k.json` (200K window, util 0.96).

## Configuration in one page

KVMem has three independent budgets. Mixing them up is the usual source of "it does not fit".

| budget | knob | what it buys |
|---|---|---|
| **GPU window** | `select_budget` (B) + `gen_budget` (R) | KV resident for one request: selected history + generation headroom |
| **logical context** | `max_model_len` (CTX) | how long a single conversation may be; history beyond the window lives in RAM |
| **host history** | `cpu_read_cache_bytes` + `repository_blocks` | how many tokens of history can be kept, shared by every session |
| **session checkpoints** | `session_capacity` × `session_checkpoint_bytes` | how many conversations stay warm |

```
GPU KV needed          ≈ 17,408 B/token × (B + R) + 0.39 GiB   # the 3 GDN groups hold fixed state
history tokens total   ≈ cpu_read_cache_bytes / 17,408      # per rank, shared by all sessions
host RAM for that      = 2 × the above               # TP=2 runs two archives on one host
sessions               × CTX   ≤ history tokens total        # otherwise the next session queues
```

Defaults (verified): B = 36K, R = 8K (window 45,056), CTX = 256K, `gpu_memory_utilization` = 0.92,
`prefill_chunk_tokens` = 2,048, `query_replay_budget` = 4,096, `session_capacity` = 1,
`cpu_read_cache_bytes` = 6 GiB, `mm_gdn_prefill_backend` = `flashqla_legacy`.

> `query_replay_budget` defaults to R and costs ~180 KiB of VRAM per token. Leaving it at 16K is what
> made a 48K window look impossible; set it explicitly.

`generation_config` selects the sampling defaults: `vllm` (repository default) keeps vLLM's neutral
defaults, while `auto` applies the checkpoint's `generation_config.json` to requests that omit
sampling parameters. A reasoning model left at the engine default (temperature 1.0) degrades into
rambling on long outputs and its draft acceptance collapses; the shipped launcher sets `auto` and the
verified checkpoint records temperature 0.3, top_p 0.95, top_k 20.

Full tables, the parameter mapping from the old adapter (`B/R/CTX/HOST_MIB/SESSIONS`) and capacity
arithmetic: [docs/sm75/CONFIGURATION.md](docs/sm75/CONFIGURATION.md).

## DFlash2 route

`draft_method=dflash`, `draft_tokens=7`, bounded window (`logical_kv_capacity=true`).

**Draft we run: [`syvai/Qwen3.8-27B-DFlash2-W4A16`](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16)**
— 4-bit pack-quantized (compressed-tensors, group 128), 1.28 GB. It needs the engine's W4A16
draft patch ([`weicj/vLLM-2080Ti-Definitive#265`](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/265)),
which reads the draft's K/V rows through the quantization kernel instead of a dense `.weight`.

The BF16 draft [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
(3.85 GB) also works and is the pairing the engine's own profiles use; it costs 1.3 GB per card
more, which comes straight out of the KV pool.

**Block-FP8 drafts do not work here.** Every `DFlash2-FP8` artifact we tried (the published one and
two we quantized ourselves, including a build with only `o_proj` quantized) drops the acceptance
length to 1.0 — no speculation at all. The weights are sound (dequantizing reproduces the BF16
tensor within 2%), so the failure is in the FP8 forward path for this five-layer draft, not in the
checkpoint. Treat FP8 drafts as unsupported on SM75.

Boundaries of the route:

| boundary | detail |
|---|---|
| concurrency | **1**. The bounded planner reserves one full window per admitted request; three windows would need ≈9 GiB |
| prefix caching | must stay **off** (`draft_contract.py`: bounded DFlash2 requires APC-off GDN state) |
| KV layout | `BLHNC` — the target and the draft have mixed page sizes |
| KV pool | FP8 target (28.7 GB weights): **4.0 GiB works, 4.5 GiB OOMs at startup**; NVFP4 target (18.8 GB): 5 GiB reachable |
| per-request reservation | `3 × (1 + k)` mamba state pages + `ceil((B+R)/page)` attention pages + 5 draft pages + 1 null page = **54 pages ≈ 3.0 GiB at k=7**. Dropping to k=3 halves the mamba term (24 → 12 pages, ≈0.85 GiB) |
| step cost | ≈8 ms/step above the pristine engine on identical prompts (scheduler 0.14 ms, connector hooks 0.11 ms — the rest is launch-side) |
| throughput | bounded lane: **106 tok/s** code, **197 tok/s** repeat; pristine engine on the same draft: **151 / 254** |
| chat | supported — a chat request is rendered to token ids and planned through the same rolling window |

## Known limits

1. **Concurrency is batched, not free.** Independent sessions decode in one step — the MTP lane runs
   `max_num_seqs=3` and three windowed sessions advance together — but window refresh and archiving
   stay serialized, so a session that needs a rebuild makes the others wait for that step. The bounded
   DFlash2 route is stricter: `max_num_seqs=1`, because every admitted request reserves a full window
   (see [DFlash2 route](#dflash2-route)).
2. **Rolling window mode** (`rolling_active_window=true`) plans from token ids. `/v1/completions`
   takes them directly; from 0.6.5 a `/v1/chat/completions` request is rendered with the served
   tokenizer and planned through the same window (before that it was rejected). What it still cannot
   do is accept a raw string prompt without tokenization on the server side.
3. **Prefix images** (multi-image prefix sharing) are refused for TP > 1 upstream.
4. **Context is not capped at 262,144.** The launcher accepts `max_model_len` up to 10,000,000 for
   MTP (with `request_planning` and `B+R ≤ 262,144`), and this host served a **269,654-token** prompt
   correctly (and a 299,280-token request) with `max_model_len=524,288`. What actually binds is host
   RAM for the archive plus VRAM headroom for the prefill transient — not a context limit. The
   checkpoint declares 262,144 positions with `rope_type=default`, so beyond that RoPE is
   extrapolated: it works, but treat quality as unverified.
5. Session overflow is not evicted by default: a request that cannot get archive credit **queues**.
   Size `cpu_read_cache_bytes` for the sessions you keep, or enable
   `KVMEM_SESSION_IDLE_LRU_EVICTION=1` and a small `session_capacity`.
6. **The API has no authentication.** The default profile binds `127.0.0.1`; a profile that sets
   `host=0.0.0.0` serves the whole network, so put it behind a firewall or a reverse proxy that
   adds auth.
7. **Prefill above the window rebuilds once per generation budget.** The scheduler triggers a
   window rebuild when the next step would not fit in what is left of the window
   (`QueryPolicy.refresh` with the step-limit lookahead), so a long cold prompt rebuilds about
   every `R` tokens instead of every aligned block: 80K prompt 184 s -> 121 s, 200K 591 s -> 306 s
   -> **196 s** on this host, with the same answers.
8. **Per-card VRAM is dominated by weights.** The 27B FP8 checkpoint plus its BF16 `lm_head` and
   MTP head load as 16.1 GiB per rank (the NVFP4 build is 9.4 GiB), a DFlash draft adds 0.64 GiB
   (4-bit W4A16) or 1.93 GiB (BF16), and the CUDA context/NCCL/Triton add ~1.2 GiB — so ~17–19 GiB
   of the 21.1 GiB usable on a 22 GB card are gone before any KV. Measured pool ceilings: **4.0 GiB
   with the FP8 target, 5 GiB with NVFP4** (4.5 GiB OOMs on FP8 at startup). `gpu_memory_utilization` then sizes
   the pool from the remainder while the real long-prompt transient (logits buffer, query replay,
   assembly staging) is not part of its activation profile — pin the pool instead:
   `kv_cache_memory_bytes=4294967296` (4 GiB) with `prefill_chunk_tokens=1024` and
   `query_replay_budget=2048` boots with ~0.4 GiB free per card and serves a 200K cold prompt with
   zero allocator retries; ~1.5 GiB is the smallest pool that boots, and below ~0.8 GiB admission is
   refused. A larger pool only eats the transient — the biggest lever is quantizing the BF16
   `lm_head` (~2.5 GB per rank). `enable_cumem_allocator=true` with
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` removes the occasional retry lines (200K peak
   21,990 -> 19,680 MiB) but needs `disable_custom_all_reduce=true`: the VMM segments break the
   custom all-reduce setup, and NCCL costs about 5% of decode throughput.

## Repository layout

```
src/vllm_kvmem/          plugin (mechanism)
local/                   serving profiles
deploy/                  launcher + systemd unit
scripts/serve_watch.py   start/stop/status/req with GPU sampling
patches/                 engine + plugin patches with base commits
CHANGELOG.md             release notes
docs/sm75/               architecture, configuration, verification, credits (English)
docs/sm75/zh-CN/         Chinese design log, baseline, parameter map
docs/                    upstream kvmem-vllm documentation (unchanged)
```

## Credits

Thanks to the projects below — this stack is built on them:

| project | role | license |
|---|---|---|
| [`kvmem/kvmem-vllm`](https://github.com/kvmem/kvmem-vllm) | the KVMem mechanism, scheduler, connector, archive, sessions | Apache-2.0 |
| [`vLLM`](https://github.com/vllm-project/vllm) | model execution, attention backends, scheduler, MTP | Apache-2.0 |
| [`vLLM-2080Ti-Definitive`](https://github.com/weicj/vLLM-2080Ti-Definitive) (SM75 vLLM fork) | SM75/Turing enablement: FlashInfer on SM75, FP8 weight-only path, full CUDA graph recipe, MTP draft contract | Apache-2.0 (vLLM derivative) |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | paged attention kernels | Apache-2.0 |
| [Triton](https://github.com/triton-lang/triton) | GDN / M-RoPE / assembly kernels | MIT |
| [Qwen](https://github.com/QwenLM) model family | the served checkpoint | its model license |

Scope of this repository: the KVMem SM75 layer in `src/vllm_kvmem` (float16 compatibility, auto-session
id, page-pool demand, chat on the rolling window, `enable_prefix_caching` switch, diagnostics), the
serving profiles and launchers, four engine patches under `patches/` (bounded capacity, W4A16 draft
support, draft page geometry, DFlash2 pools under `KVMEM_SKIP_KV_CHECK`), and the verification data.

This repository previously hosted a different adapter (KVMEM entry middleware for the same fork). That
work is preserved on branch `legacy-adapter-v0.5`; `main` is the port described here.
