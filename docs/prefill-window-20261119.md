# Prefill is chunked prefill again, with one Query retrieval before decode

> **Current default (tag `0.6.4`):** a plan-less prefill **slides by recency**
> (`selection='pressure'`, sink band + newest fill) and runs the Query picker
> **once**, in the refresh inside the last batch. The per-slide Query picker is
> opt-in (`semantic_prefill=True`) and is not the shipped default. Shipped
> profile `local/serve-2080ti-default.json`: `B = 36,864`, `R = 8,192`,
> `sink = 8,192`, `recent = 2,048`, MTP4, FP8 weights, `max_num_seqs = 1`,
> `mm_gdn_prefill_backend = flashqla_legacy`; 200,052-token prefill = **196.2 s /
> 1,020 tok/s** on 2×2080Ti SM75 (38 `pressure` slides + 1 `chunk` refresh).
> The portable Triton GDN fallback measures 230.0 s / 870 tok/s.

Plan-less requests (every plain `/v1/completions` or chat call) used to run the
semantic refresh during prefill: a 204,800-token prompt refreshed **74 times**,
each refresh firing a private forward of the chunk to capture Q, then re-selecting
B with MeanK and re-assembling the window. 113 s of the 351 s prefill went there
and 47 GB moved host→device re-packing the same tokens.

Two things were conflated. `QueryPolicy.admit` caps a request at `B + R`, so a
prompt longer than the window must evict while it prefills; and the only eviction
path was the full semantic refresh. There is no decode query during prefill, so
the selection being run was the chunk's own Q — expensive and no more correct
than keeping the newest window in order.

`admission.refresh_selection` now splits the two:

| Situation | Picker | Probe forward |
| --- | --- | --- |
| mid-prefill (more than one batch of tokens left) | `pressure` (newest first) | none |
| refresh inside the last batch, and every decode refresh | `chunk` (Query) | yes |

The mid-prefill refresh still re-assembles the retained window: 36,851 tokens,
642 MB at 17,408 bytes per token, 0.13 s. That is 73 × 642 MB ≈ 47 GB of PCIe
traffic per 204,800-token prompt, unchanged by this work — the saving here is the
probe forwards, not the copies.

Measured on the 2×2080Ti SM75 profile, 204,800-token cold prompt, `R = 4096`
(historical; the shared profile now uses `R = 8,192`):

| | before | after |
| --- | ---: | ---: |
| prefill wall | 351 s | **259 s** |
| prefill tok/s | 584 | **790** |
| maintenance | 113 s | 13 s |
| probe forwards | 74 | 1 |
| host→device re-assembly | 47 GB | 47 GB (unchanged) |

Needle at 131,072 tokens: the final Query refresh is what recovers a needle that
the recency window no longer holds. With a needle at 65,536 the model answered
`ZX-7731-QK` only with that final retrieval; with pure recency it returned a word
from the filler. A needle at 122,000 is inside the window and answers either way.

Upstream `origin/main` refreshes even more eagerly here: its
`QueryPolicy.refresh` returns True for every aligned block while `history <
prompt_tokens`. The plan-based path (`plan.action`) already used `pressure`; only
requests without a request plan fell into the semantic branch.

What is left: the window is still rebuilt from the archive on every slide. A
ring/offset layout would make a slide a page-table update, and the per-chunk cap‑
ture cost (1.16 ms/token against the pristine engine's 0.68) is a separate item.

## Shipped profile: recency slide, one Query at decode entry

`refresh_selection(..., semantic_prefill=False)` makes every mid-prefill slide a
`pressure` (newest-first) pick; only the refresh inside the last batch runs the
Query picker. The retained window is the sink band plus the newest fill, so the
attended length is whatever the window holds:
`sink (8,192) + newest (B − 8,192) = 36,864`.

Measured 2026-11-21 on 2×2080Ti SM75 with the shipped profile
`local/serve-2080ti-default.json` (`B = 36,864`, `R = 8,192`, `sink = 8,192`,
`recent = 2,048`, MTP4, FP8 weights, `max_num_seqs = 1`,
`max_num_batched_tokens = 4101`, pool sized by `gpu_memory_utilization = 0.92`,
`mm_gdn_prefill_backend = flashqla_legacy`):

| Prompt | Prefill | Refreshes |
| ---: | ---: | ---: |
| 200,052 | **196.2 s / 1,020 tok/s** | 39 (38 `pressure` + 1 `chunk`) |

The GDN prefill kernel dominates this workload: the same profile with the
portable Triton fallback (`mm_gdn_prefill_backend = null`) measures 230.0 s /
870 tok/s, and the per-chunk forward drops from 1,134 ms to 947 ms with the SM75
FlashQLA build. `deploy/run-kvmem-vllm.sh` exports the FlashQLA
`PYTHONPATH`/`TORCH_EXTENSIONS_DIR` when the engine ships the extension and
falls back to `null` when it does not.

The `prefill_window_tokens` cap trades recall for speed; on the earlier local
override (`B = 49,152`, `B + R = 57,344`, 3.5 GiB pool) the same class of
204,800-token prefill measured 918 tok/s (223 s) with no cap, and the cap gave:

| prefill extent (local override `B = 49,152`) | 200K prefill |
| --- | ---: |
| 57,344 (`B + R`) | 918 tok/s (223 s) |
| 32,768 | 1,011 tok/s (202 s) |
| 24,576 | **1,082 tok/s (189 s)** |

## Opt-in: Query per slide (`semantic_prefill=True`)

Every refresh composes the same window: a fixed sink band, a fixed recent band,
and a scored middle, filled to `select_budget`. Sink and recent keep the chunks
after a slide connected to the text right before them; the middle carries
long-range evidence.

The queries for that scoring come from `_kvmem_graph_captures`, the buffers the
previous public forward already wrote. A slide therefore costs a slice and a
`clone` (~1–2 ms) instead of a private forward (~1.0–1.3 s). Requests that are
not alone in the batch still fall back to the probe.

`gen_budget` (R) is no longer just generation headroom: it sets how far the
window slides between refreshes, `R − lookahead − block_tokens`. With the local
override (`B = 49,152`, `R = 8,192`, `sink = 10,240`, `recent = 2,048`):

| Prompt | Chunks | Refreshes | Per-refresh growth | Probes | Prefill |
| --- | ---: | ---: | ---: | ---: | ---: |
| 87,035 | 85 | 6 | 6,144 (6 chunks) | 0 | 86.3 s (1,020 tok/s) |
| 204,951 | 201 | 26 | 6,144 | 0 | 220.8 s (928 tok/s) |

On the local override the two modes are within measurement noise at 200K
(220.8 s for Query-per-slide vs 223 s for recency). The shipped profile
(`B = 36,864`, `max_num_batched_tokens = 4101`) slides every ~4,096 tokens and
measures 230.0 s, because a larger `max_num_batched_tokens` leaves less room for
`R` between slides. The shipped default picks recency because it is simpler and
keeps the newest window contiguous; long-context repetition loops that appeared
with the per-slide Query picker have not reproduced under the recency slide.

For reference, the same 204,951-token prompt took 351 s when every refresh ran a
private probe forward, and 189 s with the fixed 16K prefill pool (which keeps
B-sized recall out of the window). Dropping R to 4,096 pushed the refresh count
to ~50 and cost 17.7 s, which is why the shared profile now defaults to 8,192.
