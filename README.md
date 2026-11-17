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
- Linux, CUDA 13 driver, Python 3.12 venv (the engine's venv is reused; no `pip` on this host,
  `uv pip install --python …` works).

## Quick start

```bash
# 1. engine (the model actually runs there); its venv also provides `vllm-kvmem`
cd /home/srj/vLLM-2080Ti-Definitive        # v0.2.2.post3 + SM75 fork
git apply /path/to/patches/0001-engine-skip-kv-check.patch

# 2. mechanism (this repository, editable into the engine venv)
cd /path/to/kvmem-vllm-2080ti
git apply patches/0002-kvmem-sm75-compat.patch   # only when starting from upstream
uv pip install --python /home/srj/vLLM-2080Ti-Definitive/.venv/bin/python -e .

# 3. serve + verify (polls /v1/models, samples nvidia-smi, detects stalls; never sleeps blindly)
./deploy/run-kvmem-vllm.sh start          # default profile: local/serve-2080ti-default.json
./scripts/serve_watch.py status
./scripts/serve_watch.py req --file request.json

# optional: systemd
sudo cp deploy/kvmem-vllm.service /etc/systemd/system/ && sudo systemctl daemon-reload
```

Profiles: `local/serve-2080ti-default.json` (32K window, 256K context, 1 retained session),
`local/serve-2080ti-128k.json`, `local/serve-2080ti-256k.json`.

## Configuration in one page

KVMem has three independent budgets. Mixing them up is the usual source of "it does not fit".

| budget | knob | what it buys |
|---|---|---|
| **GPU window** | `select_budget` (B) + `gen_budget` (R) | KV resident for one request: selected history + generation headroom |
| **logical context** | `max_model_len` (CTX) | how long a single conversation may be; history beyond the window lives in RAM |
| **host history** | `cpu_read_cache_bytes` + `repository_blocks` | how many tokens of history can be kept, shared by every session |
| **session checkpoints** | `session_capacity` × `session_checkpoint_bytes` | how many conversations stay warm |

```
GPU pool needed        ≈ 3 groups × (B + R)          # this model has 3 attention cache groups
history tokens total   ≈ cpu_read_cache_bytes / 17,408      # per rank, shared by all sessions
host RAM for that      = 2 × the above               # TP=2 runs two archives on one host
sessions               × CTX   ≤ history tokens total        # otherwise the next session queues
```

Defaults (verified): B = 32K, R = 16K, CTX = 256K, `gpu_memory_utilization` = 0.90,
`query_replay_budget` = 4,096, `session_capacity` = 1, `cpu_read_cache_bytes` = 6 GiB.

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

Our contribution is limited to: the SM75/float16 compatibility layer in `src/vllm_kvmem`, the
auto-session identifier, the page-pool demand fix, the diagnostics, one small engine patch, the
deployment profiles and the verification data. Anything that works well here is the upstream authors'
work; anything that does not is our fault.

History note: this repository previously hosted a different adapter (KVMEM entry middleware for the
same fork). That work is preserved on branch `legacy-adapter-v0.5`; `main` is the port described here.
