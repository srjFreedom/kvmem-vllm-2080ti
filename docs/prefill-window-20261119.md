# Prefill is chunked prefill again, with one Query retrieval before decode

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
| refresh inside the last chunk, and every decode refresh | `chunk` (Query) | yes |

The mid-prefill refresh still re-assembles the retained window: 36,851 tokens,
642 MB at 17,408 bytes per token, 0.13 s. That is 73 × 642 MB ≈ 47 GB of PCIe
traffic per 204,800-token prompt, unchanged by this work — the saving here is the
74 probe forwards, not the copies.

Measured on the 2×2080Ti SM75 profile, 204,800-token cold prompt, `R = 4096`:

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

## Fixed prefill extent

Mid-prefill the resident set is `sink + newest fill`, so the attended length is
whatever the window holds. Pinning it (`prefill_window_tokens`, block-aligned,
0 disables) makes a prefill chunk cost the same at any history length:

| prefill extent | 200K prefill | rebuilt per slide |
| --- | ---: | ---: |
| 57,344 (`B + R`, default) | 918 tok/s (223 s) | 36,851 tokens, 642 MB |
| 32,768 | 1,011 tok/s (202 s) | 16,384 tokens |
| 24,576 | **1,082 tok/s (189 s)** | 8,192 tokens, 140 MB |

Per-chunk model time is bounded instead of climbing to 1,243 ms and staying
there: 663 ms while the prompt still fits, ~850 ms once the window slides.
32K prefill is unaffected (it fits under every setting).

NIAH at 131,072 with the 24,576 extent: needle at 65,536 and at 122,000 answer
correctly; a needle at 30,000 is missed — with the default 57,344 extent it is
missed as well, so that case is the existing outside-the-window behaviour rather
than a regression from the smaller extent.

## Final shape: one retrieval per slide, no probe forward

Every refresh now composes the same window: a fixed sink band, a fixed recent
band, and a scored middle, filled to `select_budget`. Sink and recent are what
keep the chunks after a slide connected to the text right before them; the
middle is what carries long-range evidence.

The queries for that scoring come from `_kvmem_graph_captures`, the buffers the
previous public forward already wrote. A slide therefore costs a slice and a
`clone` (~1–2 ms) instead of a private forward (~1.0–1.3 s). Requests that are
not alone in the batch still fall back to the probe.

`gen_budget` (R) is no longer just generation headroom: it sets how far the
window slides between refreshes, `R − lookahead − block_tokens`. With
`B = 49,152`, `R = 8,192`, `sink = 10,240`, `recent = 2,048`:

| Prompt | Chunks | Refreshes | Per-refresh growth | Probes | Prefill |
| --- | ---: | ---: | ---: | ---: | ---: |
| 87,035 | 85 | 6 | 6,144 (6 chunks) | 0 | 86.3 s (1,020 tok/s) |
| 204,951 | 201 | 26 | 6,144 | 0 | 220.8 s (928 tok/s) |

For reference, the same 204,951-token prompt took 351 s when every refresh ran a
private probe forward, and 189 s with the fixed 16K prefill pool (which keeps
B-sized recall out of the window). Dropping R to 4,096 pushed the refresh count
to ~50 and cost 17.7 s, which is why the shared profile now defaults to 8,192.
