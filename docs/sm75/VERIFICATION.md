# Verification

Every number below was measured on this host, on the hardware and software listed, not simulated.

## Environment

| item | value |
|---|---|
| GPUs | 2 × RTX 2080 Ti 22 GB (SM75, CC 7.5) |
| host RAM | 62 GB |
| model | 27B FP8 hybrid: 16 full-attention + 48 GDN layers, weight-only FP8, BF16 `lm_head`, native MTP |
| TP / dtype | 2 / `float16` (SM75 has no bfloat16) |
| KV dtype | `fp8_e4m3` (KVMem's required spelling; `fp8` is the same alias upstream) |
| engine | `vLLM-2080Ti-Definitive` @ `01506a5471` = vLLM `0.2.2.post3` + one 15-line patch |
| plugin | this repository (`vllm_kvmem 0.1.0.dev1`) |

## Startup

| case | result |
|---|---|
| eager, window 6,144 | READY in 54–57 s |
| compiled graph, first run | READY in 290 s (compilation), 57 s when cached |
| `full_graph` + MTP4 | READY in 56–58 s |

## Correctness / prefix hits

| request | result |
|---|---|
| 4,888-token needle | `ZX-4417-QJ7` ✅ |
| 65,496 tokens | ✅ 2 m 06 s cold |
| 86,532 tokens | ✅ 2 m 49 s cold; repeat **0.85 s** (~200×) |
| **218,220 tokens (above the 205,752 pool)** | ✅ 8 m 18 s cold; repeat **1.36 s** (~365×) |
| session continuation, 7.2K, over window | 7.5 s → **0.33 s** (pm policy) |
| session continuation, 2.2K, under window | 2.4 s → **0.25 s** |
| A/B session alternation | each hits in 0.25–0.40 s |
| long generation, window 6,144, 9,000 tokens | 100.5 tok/s, 2 bounded refreshes, no stall |
| long generation, `max_tokens=12000` on the default profile | ~11K tokens at 84 tok/s |
| **258,743 tokens at the default profile** (window 49,152, `max_model_len=262144`, util 0.90) | ✅ `ZX-4417-QJ7`, 787 s, 329 tok/s prefill, no OOM |
| **269,654 tokens (beyond the model's 262,144 positions)**, `max_model_len=524288`, B=R=8,192, util 0.90 | ✅ `ZX-4417-QJ7`, 686 s, 393 tok/s prefill |
| 299,280 tokens, same profile | ✅ `FINISHED_STOPPED`, no OOM |

## Throughput

| workload | kvmem | pristine engine, same config |
|---|---|---|
| code generation (900-token answer), MTP4, full graph | **79.6 / 86.5 tok/s** | 80.8 / 87.2 tok/s |
| prose generation, MTP2 | 46.4 tok/s | 47.3 tok/s |
| prose generation, MTP4 | 44.6 tok/s | 45.1 tok/s |
| prefill, 4.1K cold | 1,021–1,063 tok/s | 1,174 tok/s |

Decode is bound by the Marlin FP8 weight GEMM (52 % of GPU time, ≈30 ms/step) plus the MTP draft
GEMV (18 %); fp8 KV costs nothing measurable (fp16 KV 47.8 vs fp8 48.4 tok/s for the pristine engine),
and TP all-reduce is 2 %. MTP acceptance decides the speed: ~4/5 on code, ~2/5 on prose.

## Behaviour checks

| check | observation |
|---|---|
| GPU page-pool demand | `3 groups × (B+R)`; `ValueError: GPU page pool cannot fit one worst-case KVMem request` when the pool is short |
| engine start beyond the pool | skipped: `KVMem: skipping the KV capacity check (max_model_len=262144 needs 4.66 GiB, pool has 2.95 GiB)` |
| session miss diagnosis | `_session_miss_probe` writes `/tmp/kvmem_session_miss.jsonl` with `miss01..miss10` codes |
| reason a >window continuation used to miss | `miss04`, `prompt_history=0` → `sealed` policy never retains a P snapshot (`retain_prompt = policy=='pm'`) |
| session overflow | deferred (queued), not evicted; `archive_capacity_deferrals` counts it |
| checkpoint overflow | graceful skip (`checkpoint_deferrals+1`), request still answers |
| 300K prompt with the 49,152 window at util 0.96/0.88 | CUDA OOM during prefill (128 MiB request) — VRAM, not a context cap |
| same window at util **0.90** | 258,743-token prefill passes, pool 167,889, base 19,760 MiB, peak ~21,290 MiB |
| same window at util **0.92** | 258,743-token prefill passes, pool **191,453**, base 20,188 MiB, **peak 21,822 of 22,528 MiB** |
| concurrency | 2 tiny prompts overlap (0.30 s total); 2 windowed prompts serialize (4.3 s for 2×2.2 s); `authority_limit=3`, `rejections=0` |
| rolling mode | booted; `/v1/chat/completions` → 400 (needs token-ID completions); 100.6 tok/s vs 100.5 default |

## Reproducing

```bash
./deploy/run-kvmem-vllm.sh start                       # default profile
./scripts/serve_watch.py status
./scripts/serve_watch.py req --file request.json       # prints wall time, tok/s, GPU busy ratio, stall flag
```

`serve_watch.py` polls `/v1/models`, samples `nvidia-smi` every 0.4 s and reports a `STALL` when the GPU
stays idle while a request is in flight, so a hang is visible instead of being waited out.
