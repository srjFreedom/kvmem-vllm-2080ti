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
