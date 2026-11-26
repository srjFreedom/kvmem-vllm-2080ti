# KVMem on 2× RTX 2080 Ti (SM75)

**200K-token prefill at 1,000+ tok/s on 22 GB Turing cards, with concurrent sessions.**

FP8 weight-only quantization is what makes this possible: the 27B model's weights drop from 54 GB to
28.7 GB, leaving enough VRAM for a multi-GiB KV pool *and* a speculative draft on two 22 GB cards.
KVMem then bounds the resident window (B+R tokens on GPU, full history in host RAM) so the pool
serves **concurrent requests** instead of one giant context. The result:

- **200K cold prefill: 1,020 tok/s** (196 s) on a 4 GiB KV pool — the pool holds a 53K-token window,
  not the full 200K.
- **Concurrent decode without more VRAM** — independent sessions decode in one scheduling step; the
  pool holds one window per step, not one per request.
- **Sessions stay warm** — per-session checkpoints resume a continued conversation in ~0.3 s instead
  of re-prefilling it.
- **Two speculative routes on one engine** — MTP4 and DFlash2 k=7 (4-bit W4A16 draft, 1.19 GB), both
  on the compiled-graph path. DFlash2 lane: 100K prefill at **1,162 tok/s**, 2 concurrent.
- **Repeats are almost free** — a prompt whose cold prefill took 196 s is re-served from the archive
  in 1.36 s (≈365×).

The engine is the SM75-capable vLLM fork listed under [Credits](#credits); the KVMem mechanism comes
from [`kvmem/kvmem-vllm`](https://github.com/kvmem/kvmem-vllm). What lives here is the integration: the
SM75 window/session layer, the DFlash2 route and its engine patches, the profiles, and the measured
configuration for two 22 GB Turing cards.

*中文文档：[README.zh-CN.md](README.zh-CN.md)*

---

## What's new in 0.6.6

| change | why it matters |
|---|---|
| **DFlash2 lane is production-ready** | shipped profile B=36K / R=8K / 3.0 GiB pool / 2 concurrent; measured at B=46K, 100K cold prefill **1,162 tok/s**, two distinct 57K+ prompts decode together with 696 MiB headroom. The lane ships as `deploy/example-dflash2-launcher.sh` + `local/serve-2080ti-dflash2.json` |
| **Admission reserve** | Native speculation appends lookahead rows on top of the admitted extent; without the reserve the allocation overshoots the window and raises `RefreshRequired` on a step with no forward to drive the refresh — an EngineCore spin. `admission_reserve = max(lookahead, spec+1)` stops admission that many tokens short |
| **Native page charge** | Demand was computed in guided-block units (32 tokens) instead of native page units (1,648 tokens), inflating the pool demand ~50× and rejecting requests that physically fit. Now charged in `block_pool.block_size` |
| **Align-state env** | `mamba_cache_mode=align` now sets `KVMEM_DFLASH_ALIGN_STATE=1` so the capacity planner accepts the align geometry (small resident GDN state, draft merged into the target pool) |
| **`mamba_block_size` defaults to `max_model_len`** | The profile no longer pins 1,648. With the engine default each request holds 1 GDN state page + 7 speculative tail = 8 pages, the same geometry as the MTP lane. Pinning 1,648 made the state dense (42 pages/group at 57K ≈ 3.2 GiB) and OOMed the 3 GiB pool |

DFlash2 lane measurements (2 × RTX 2080 Ti, FP8 target, W4A16 draft, `B=47,104`, `R=8,192`, pool 3.0 GiB):

- **100K cold prefill: 1,162 tok/s** (86.1 s). R sensitivity: R=4K → 1,136 (−2.2%), R=2K → 1,060 (−8.3%).
  The window is `B+R`, so a smaller R means more slide refreshes on a long prompt.
- **2-concurrent cold** (two distinct 57K/62K prompts, archive cleared): both pass, peak headroom
  **696 MiB**. Sliding shared pages let the physical pool reuse across refreshes; `capacity=B+R` is a
  logical bound, not a per-request physical reservation.
- **Pool ceiling: 3.0 GiB.** 3.2 GiB OOMs at prefill completion (~192 MiB transient + ~700 MiB runtime
  residency growth). The DFlash2 draft (1.19 GiB/rank) costs ~0.8 GiB more than the MTP head (0.40 GiB),
  which is why the MTP lane reaches 4.0 GiB and DFlash2 stops at 3.0.
- **Per-step parity with the engine**: 45.5 ms/step under KVMem vs 45.7 ms on the pristine engine.
  Decode: 180 tok/s on repetitive text, 110+ tok/s on code generation.

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
| **DFlash2 k=7, 4-bit W4A16 draft** | decode **180 tok/s** repetitive, **110+ tok/s** code; per-step parity 45.5 vs 45.7 ms |
| **DFlash2 100K cold prefill** | **1,162 tok/s** (B=46K/R=8K, 3.0 GiB pool) |
| **DFlash2 2-concurrent cold** | two distinct 57K/62K prompts, peak headroom **696 MiB** |
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
GPU KV needed          ≈ 16,384 B/token × (B + R) + 0.39 GiB   # the 3 GDN groups hold fixed state
history tokens total   ≈ cpu_read_cache_bytes / 16,384      # per rank, shared by all sessions
host RAM for that      = 2 × the above               # TP=2 runs two archives on one host
sessions               × CTX   ≤ history tokens total        # otherwise the next session queues
```

`16,384 B/token` is exact for this checkpoint: 16 full-attention layers × (4 KV heads / TP2 = 2) ×
head_dim 256 × 2 (K+V) × 1 byte (fp8). Measured from the engine both ways — `archive.json` reports
524,288 bytes per 32-token record, and `cache-geometry.json` reports a 27,000,832-byte attention page
over 1,648 tokens. So **one 256K-token session costs 4.0 GiB of archive per rank, 8.0 GiB for TP=2**,
plus ~0.17 GiB of session checkpoint (window-sized, independent of history length).

Defaults (verified): B = 36K, R = 8K (window 45,056), CTX = 256K, `gpu_memory_utilization` = 0.92,
`prefill_chunk_tokens` = 2,048, `query_replay_budget` = 4,096, `session_capacity` = 1,
`cpu_read_cache_bytes` = 8 GiB, `mm_gdn_prefill_backend` = `flashqla_legacy`.

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

`draft_method=dflash`, `draft_tokens=7`, no bounded planner: the engine sizes the pool and KVMem
opens windows at runtime. The shipped profile runs B=36K, R=8K, 3.0 GiB pool, 2 concurrent
(window 45,056 = 28 attention pages, so two concurrent windows need 56 of the pool's 66 pages).
The numbers below were measured with `--set kvmem.select_budget=47104`, a B=46K override. B=36K is
strictly better on both counts: a smaller window means less attention work per token, so prefill
should get *faster*, not slower (attention is `196,608 × W` FLOP/token against a fixed 15.6 GFLOP of
linear work, so 55,296 → 45,056 tokens removes ~7.6% of the total), and it leaves 10 pool pages of
headroom instead of 0. Only `gen_budget` (R) trades throughput for anything, because R — not B —
sets the window slide distance.

**Draft: [`syvai/Qwen3.8-27B-DFlash2-W4A16`](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16)**,
1.19 GB, 4-bit pack-quantized (compressed-tensors, group 128). It stays essentially whole per rank
(TP2), which is why the DFlash2 pool ceiling is ~3.0 GiB versus 4.0 GiB on the MTP lane: the draft
costs 0.79 GiB/rank more than the MTP head. Needs the engine's W4A16 draft patch
([`weicj/vLLM-2080Ti-Definitive#265`](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/265)).

**Block-FP8 drafts do not work here.** Every `DFlash2-FP8` artifact we tried drops the acceptance
length to 1.0 — no speculation at all. The weights are sound (dequantizing reproduces the BF16
tensor within 2%), so the failure is in the FP8 forward path for this five-layer draft. Treat FP8
drafts as unsupported on SM75.

Boundaries of the route:

| boundary | detail |
|---|---|
| concurrency | **2** at the 3.0 GiB pool. Two distinct 57K+ prompts decode together with 696 MiB peak headroom; sliding shared pages reuse the physical pool across refreshes |
| prefix caching | must stay **off** (`draft_contract.py`: DFlash2 requires APC-off GDN state) |
| KV layout | `BLHNC` — the target and the draft have mixed page sizes |
| KV pool | **3.0 GiB** with the FP8 target. 3.2 GiB OOMs at prefill completion (~192 MiB transient + ~700 MiB runtime residency). The MTP lane reaches 4.0 GiB because its head is 0.40 GiB/rank vs the draft's 1.19 GiB |
| pool floors | from `patches/0005`: the recurrent pool is floored at `max(req+1, max_num_seqs*req + 1)` so the draft's GDN state and speculative tail always have a page |
| step cost | **the same as the engine**: 45.5 ms/step against 45.7 ms. Decode: 180 tok/s repetitive, 110+ tok/s code |
| 100K prefill | **1,162 tok/s** at B=46K/R=8K. Only R costs throughput: R=4K → 1,136 (−2.2%), R=2K → 1,060 (−8.3%), because R sets the window slide distance (`R − lookahead − block_tokens`) and a shorter slide means more refreshes. B is the opposite: a smaller B shrinks the attention term, so prefill improves |
| chat | native — with no rolling window there is no token-id planning to render for |

One lane is shipped for this route: `deploy/example-dflash2-launcher.sh` with
`deploy/kvmem-vllm-dflash2.service`, backed by the profile `local/serve-2080ti-dflash2.json`. It needs
`patches/0003`–`0005` in the engine and `VLLM_KV_CACHE_LAYOUT=BLHNC` at runtime.

## Known limits

1. **Concurrency is batched, not free.** Independent sessions decode in one step — the MTP lane runs
   `max_num_seqs=3` and three windowed sessions advance together — but window refresh and archiving
   stay serialized, so a session that needs a rebuild makes the others wait for that step. The DFlash2
   lane runs `max_num_seqs=2`, limited by what the 3.0 GiB pool holds alongside the draft
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
