# KVMem on 2× RTX 2080 Ti (SM75)

KVMem keeps a **bounded KV window** on the GPU and the **full history in host RAM**, selecting the
relevant history back per request. It makes long-context serving possible on hardware that cannot
hold the context, and it makes repeated or continued requests skip the prefill almost entirely.

This repository is a **patch set on top of other people's work** — see [Credits](#credits).
The mechanism comes from [`kvmem/kvmem-vllm`](https://github.com/kvmem/kvmem-vllm); the engine is a
stock SM75-capable vLLM fork; what we add is the compatibility layer, the deployment profile and the
measured configuration for two 22 GB Turing cards.

*中文文档：[README.zh-CN.md](README.zh-CN.md)*

---

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
| `src/vllm_kvmem/**` | upstream `kvmem/kvmem-vllm` | SM75/float16 compatibility, auto session id, page-pool demand in guided-block units, miss diagnostics |
| engine | `vLLM-2080Ti-Definitive` (`v0.2.2.post3` + SM75 work) | **one 15-line patch**: allow `max_model_len` above the KV pool when KVMem bounds the window |
| `local/`, `deploy/`, `scripts/` | — | serving profiles, watchdog, launcher, systemd unit |
| `patches/` | — | both patches as plain files, with base commits |

## Verified on real hardware

Host: 2 × RTX 2080 Ti 22 GB (SM75), 62 GB RAM, 27B FP8 hybrid model (16 full-attention + 48 GDN
layers, weight-only FP8, BF16 `lm_head`), TP = 2, `dtype=float16`, `kv-cache-dtype fp8_e4m3`.

| capability | result |
|---|---|
| code generation decode | **80–87 tok/s** (pristine engine on the same config: 80.8–87.2) |
| prefill | 1,021–1,063 tok/s (pristine 1,174) |
| 4.8K needle request | correct, ~7.9 s cold |
| **prompt above the KV pool** | 218,220 tokens answered correctly; repeat **1.36 s** (≈365×) |
| session continuation, over window | 7.5 s → **0.33 s** |
| session continuation, under window | 2.4 s → **0.25 s** |
| long generation past the window | 9,000 tokens, 100.5 tok/s, 2 bounded refreshes |
| multi-session alternation | each session hits (0.25–0.40 s) |
| concurrency | admitted, but **scheduled serially** (one windowed request per step) |

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

git clone git@github.com:srjFreedom/kvmem-vllm-adapter.git kvmem-vllm-2080ti   # this repo (private)
cd kvmem-vllm-2080ti

export KVMEM_ENGINE_DIR=$PWD/../vLLM-2080Ti-Definitive
export KVMEM_MODEL_DIR=/path/to/checkpoint
./deploy/install.sh                                               # patch, plugin, smoke
```

# 1. check the host, patch the engine, install this plugin (editable) and smoke-test
cd /path/to/kvmem-vllm-2080ti
./deploy/install.sh

# 2. serve and verify (polls /v1/models, samples nvidia-smi, detects stalls)
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
  --set kvmem.gen_budget=4096
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

Profiles: `local/serve-2080ti-default.json` (B 36K / R 4K, util 0.92, chunk 2,048), `local/serve-2080ti-lean.json` (49K window, util 0.82), `local/serve-2080ti-200k.json` (200K window, util 0.96).

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

Defaults (verified): B = 36K, R = 4K (window 40,960), CTX = 256K, `gpu_memory_utilization` = 0.92,
`prefill_chunk_tokens` = 2,048, `query_replay_budget` = 4,096, `session_capacity` = 1,
`cpu_read_cache_bytes` = 6 GiB.

> `query_replay_budget` defaults to R and costs ~180 KiB of VRAM per token. Leaving it at 16K is what
> made a 48K window look impossible; set it explicitly.

Full tables, the parameter mapping from the old adapter (`B/R/CTX/HOST_MIB/SESSIONS`) and capacity
arithmetic: [docs/sm75/CONFIGURATION.md](docs/sm75/CONFIGURATION.md).

## Known limits

1. **Concurrency is serialized.** Two windowed requests are admitted but scheduled one per step, so
   the second waits for the first. A real fix needs the plugin to keep one selection/query-capture
   frame per request.
2. **Rolling window mode** (`rolling_active_window=true`) works but only accepts token-ID
   `/v1/completions`; it is incompatible with chat and measured the same throughput.
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
7. **Long cold prompts need VRAM headroom.** A prompt far above the window is prefilled while the
   window is rebuilt every 256 tokens, and the assembly transient plus the logits buffer can exhaust
   the pool; `gpu_memory_utilization=0.88` with `prefill_chunk_tokens=1024` keeps an 80K-token cold
   prompt inside 20.5 GB per GPU (an 80K prompt at 0.92 / chunk 2048 failed to allocate).

## Repository layout

```
src/vllm_kvmem/          plugin (mechanism)
local/                   serving profiles
deploy/                  launcher + systemd unit
scripts/serve_watch.py   start/stop/status/req with GPU sampling
patches/                 engine + plugin patches with base commits
docs/sm75/               architecture, configuration, verification, credits (English)
docs/sm75/zh-CN/         Chinese design log, baseline, parameter map
docs/                    upstream kvmem-vllm documentation (unchanged)
```

## Credits

This work exists only because others built the foundation. We are a patch on top of them:

| project | role | license |
|---|---|---|
| [`kvmem/kvmem-vllm`](https://github.com/kvmem/kvmem-vllm) | the KVMem mechanism, scheduler, connector, archive, sessions — this repository is a derivative | Apache-2.0 |
| [`vLLM`](https://github.com/vllm-project/vllm) | model execution, attention backends, scheduler, MTP | Apache-2.0 |
| `vLLM-2080Ti-Definitive` (private fork) | SM75/Turing enablement: FlashInfer on SM75, FP8 weight-only path, full CUDA graph recipe, MTP draft contract | Apache-2.0 (vLLM derivative) |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | paged attention kernels | Apache-2.0 |
| [Triton](https://github.com/triton-lang/triton) | GDN / M-RoPE / assembly kernels | MIT |
| [Qwen](https://github.com/QwenLM) model family | the served checkpoint | its model license |

Scope of this repository: the SM75/float16 compatibility layer in `src/vllm_kvmem`, the auto-session
identifier, the page-pool demand fix, the diagnostics, one engine patch of 15 lines, the deployment
profiles and the verification data.

This repository previously hosted a different adapter (KVMEM entry middleware for the same fork). That
work is preserved on branch `legacy-adapter-v0.5`; `main` is the port described here.
