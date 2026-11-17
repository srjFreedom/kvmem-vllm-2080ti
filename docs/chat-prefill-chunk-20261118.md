# Native chat prefill chunk: 256 tokens is the optimum, 2048 kills the engine

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
