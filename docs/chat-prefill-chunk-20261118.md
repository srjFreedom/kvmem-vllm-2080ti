# Native chat prefill chunk: 256 tokens is the optimum, 2048 kills the engine

> **Correction (2026-11-20):** the "2048 kills the engine" result was measured on the
> older 4 GiB pool. On the production 3.5 GiB pool (`kv_cache_memory_bytes =
> 3,758,096,384`) the 2,048-token chunk **no longer crashes at 200K** — it runs 200K
> prefill in 207 s (965 tok/s, no `RefreshRequired`) and is the **fastest** of
> 256/1024/2048 at both 8K and 200K. See the
> [2026-11-20 retest](#2026-11-20-retest-35-gib-pool) at the end.

Native chat requests carry no request plan, so `prefill_chunk_tokens` never sized their
prefill: `KVMemScheduler` capped every step at a hardcoded `8 * block_tokens` (256 tokens at
block 32). `chat_prefill_blocks` now exposes that cap; the default stays 8.

Measured on the 2x2080Ti SM75 profile with one 134,282-token chat prompt replayed with
`max_tokens=8` (the same payload for every arm), one restart per arm:

| chat_prefill_blocks | tokens/step | prefill steps | model time | maintenance | prefill wall | tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 256 | 525 | 179.5 s | 21.9 s (11%) | 201.4 s | 667 |
| 32 | 1024 | 131 | 149.1 s | 59.9 s (29%) | 209.1 s | 637 |
| 64 | 2048 | engine died during prefill (see below) | | | | |

Two facts follow. Bigger chunks do make the forwards cheaper (`model` time 179.5 s -> 149.1 s,
749 -> 894 tok/s of pure forward), but the per-refresh publication grows with the chunk: the
window rotates once every 2048 tokens, and at 1024 tokens per step each rotation publishes
about 2.8x more than at 256, so the saved forward time is handed back. Total prefill is 4%
slower at 32, and the logits transient doubles to ~0.5 GiB.

At 64 a chunk spans the whole 2048-token rotation quantum and `WindowManager._local()` raises
`RefreshRequired` ("Logical position predates committed working set" is not the message, but the
guard is the same: `value > self.capacity`). Nothing catches that exception, so it reaches
`EngineCore` and kills the engine:

```
File "serving/scheduler.py", line 1795, in schedule
File "vllm/v1/core/sched/scheduler.py", line 867, in schedule
File "serving/capacity.py", line 264, in allocate_slots
File "serving/window_manager.py", line 65, in _local
vllm_kvmem.serving.window_manager.RefreshRequired: chatcmpl-...
```

`grep -rn "RefreshRequired" some(adapter, fork)` finds only the definitions, so the guard is
defensive everywhere and a single unexpected chunk size is fatal rather than a clean step
retry. Raising the rotation quantum (fewer publications, then larger chunks) and catching the
guard at the native schedule boundary are the two follow-ups this measurement points at.

## Later measurement (2026-11-19)

The 134K prompt above is the regime where the refresh publication dominates. On
shorter cold prefill the step count wins: 8,192 tokens 956 → 1,108 tok/s and
32,768 tokens 869 → 1,020 tok/s going from 256 to 1,024 tokens per step, with
200K at 600 → 570 tok/s. The default is now 32 blocks; 64 blocks still dies with
`RefreshRequired` on a 200K prompt *(on the old 4 GiB pool; superseded by the
[2026-11-20 retest](#2026-11-20-retest-35-gib-pool) below, where 2048 runs 200K on the
3.5 GiB pool)*.

The split suggests a length-dependent cap: large steps while the prompt fits the
resident window (no refresh during prefill), the 256-token cap once it does not.

## 2026-11-20 retest (3.5 GiB pool)

The "2048 dies on 200K" result predates pinning the pool with `kv_cache_memory_bytes`
(3.5 GiB). The engine log now reports `Initial free memory 21.17 GiB, reserved 3.5 GiB
memory for KV Cache`; with that ~0.5 GiB of extra headroom the 2,048-token prefill
transient fits, and the crash does not reproduce.

Method: 2 × 2080 Ti SM75 profile, one warm engine per arm, synthetic prompt, a unique
random prefix on every request so vLLM automatic prefix caching never serves the
prefill from cache. Wall time is time-to-first-token with `max_tokens=1` (≈ pure
prefill), 3 runs, median.

| prompt | 256 tok/step | 1,024 tok/step | 2,048 tok/step |
| --- | --- | --- | --- |
| 8K | 1,177 tok/s | 1,356 tok/s | **1,406 tok/s** |
| 200K | not retested | 950 tok/s | **965 tok/s** |

On the 3.5 GiB pool, 2,048 is ~3.7% faster than 1,024 at 8K and ~1.6% faster at 200K,
and 256 is the slowest of the three at 8K. The 134K cold-chat result above (256 slightly
ahead of 1,024) is a different regime and is not contradicted by this table; what this
retest does refute is the blanket "2048 kills the engine at 200K" and "256 is the
optimum" conclusions. 256 was not retested at 200K.
