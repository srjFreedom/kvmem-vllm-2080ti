# Changelog

## 0.6.11

DFlash2 runs two requests concurrently with per-request `B+R` isolation.

- **Draft shares the target's block geometry.** The draft kept a private
  64-token block ring, so a request needed a 4,224-token ring of its own (66 of
  67 pages); two requests needed 133 pages and the second was rejected on every
  decode step once both contexts approached the ring limit — the engine spun on
  empty steps (`running=2`, KV 100 %, GPU 0 %, 13,747 failed allocations). The
  draft now uses the target manager block (1,648), so one request costs 4 draft
  pages. The existing coarsening helper could never do this: it may only multiply
  the draft block by a factor dividing the target alignment, and 1,648 = 16×103
  has no multiple of 64.
- **Attention is floored at one window per concurrent request.** The planner
  trimmed the full-attention group below `max_num_seqs` windows; it now gets
  `max_num_seqs × ceil((B+R)/page) + 1`.
- **One spare recurrent page per request** so a full wave still has room for the
  next lookahead.
- Capacity now lands exactly on the 3.0 GiB budget: `[18, 18, 18, 59, 10]`.
- Verified: two concurrent 2.4k/3.0k/4.0k-token generations all complete; the
  MTP lane is unaffected and still reaches `running=2`.

## 0.6.10

DFlash2 ships serial (`max_num_seqs=1`); two concurrent requests are not feasible
at a 3.0 GiB pool.

- **Why.** The draft ring is sized from `attention_chunk_size + max_in_flight_tokens`
  (2,160 + 2,064 = 4,224 tokens = 66 blocks). Two requests must hold both draft
  windows plus the in-flight tokens: 2×2,160 + 2,064 = 6,384 tokens = 100 blocks =
  4.8 GiB, against the 67 blocks (4,288 tokens) the pool can give. Two requests do
  start together, then deadlock: once both draft contexts approach the ring limit
  the ring reports 0 free blocks, every decode allocation fails (13,747 in one
  run), and the engine spins on empty steps (`running=2`, KV 100 %, GPU 0 %).
- **Symptom.** Long generations hang permanently; short ones look fine, which is
  why a 400-token test passed and a 3,000-token one did not.
- Verified serial: 3,000-token generations complete in 39.5 s.
- The 0.6.8 side-pool credit fixes are still required; without them even one
  request per step was rejected.

## 0.6.8

DFlash2 runs two requests concurrently (it was serial).

- **Side-pool window source.** The per-request reservation read
  KVMEM_WINDOW_TOKENS from the environment, but vLLM spawns workers with a
  curated environment, so the value fell back to max_model_len: one request
  reserved a whole 262K context (66 pages) instead of the window (34). Now
  derived from the connector options every process carries.
- **Batch-wide rings.** The draft ring is sized for
  attention_chunk_size + max_in_flight_tokens across the whole step, so it is
  no longer charged per request (that demanded 133 pages of a 67-page pool).
- **Double-charged peers.** A pool reservation was scaled by admitted + new
  requests while admitted peers had already taken their pages physically; a
  17-page pool with 8 pages per request compared 8×2 against 8 free.
- Measured: running=2 at B=46K/R=8K/3.0 GiB, both requests ~35–37 s for 2000
  tokens where the serial path took ~62 s. MTP is unaffected (it has no side
  pools) and keeps its measured running=2.

## 0.6.6

DFlash2 lane is production-ready: B=46K / R=8K / 3.0 GiB pool / 2 concurrent.

- **Admission reserve.** Native speculation appends lookahead rows on top of the admitted extent;
  without the reserve the allocation overshoots the window and raises `RefreshRequired` on a step
  with no forward to drive the refresh. `admission_reserve = max(lookahead, spec+1)` stops admission
  that many tokens short of capacity.
- **Native page charge.** Demand was computed in guided-block units (32 tokens) instead of native
  page units, inflating demand ~50× and rejecting requests that physically fit. Now charges in
  `block_pool.block_size`.
- **Align-state env.** `mamba_cache_mode=align` sets `KVMEM_DFLASH_ALIGN_STATE=1` so the capacity
  planner accepts the align geometry.
- **`mamba_block_size` defaults to `max_model_len`.** The profile no longer pins 1,648. Each request
  holds 1 GDN state page + 7 speculative tail = 8 pages, the same geometry as the MTP lane.

Measured: 100K cold prefill **1,162 tok/s**; 2-concurrent cold 57K+62K both pass with 696 MiB peak
headroom; decode 180 tok/s repetitive, 110+ tok/s code.

## 0.6.5

MTP defaults to `compiled_graph`; chat rides the rolling window; DFlash2 draft support.

- **MTP `compiled_graph`.** torch.compile + `FULL_AND_PIECEWISE` beats `full_graph`: code 66.8 vs
  59.4 tok/s (+12%), repeat 103.6 vs 95.2 tok/s (+9%).
- **Chat through the rolling window.** `/v1/chat/completions` is rendered with the served tokenizer
  and planned through the same B/R window.
- **DFlash2 k=7, W4A16 draft.** 4-bit draft (1.19 GB) loads and serves at the engine's per-step cost:
  45.5 vs 45.7 ms. Needs `patches/0003`–`0005`.
- **`enable_prefix_caching` switch.** The profile can now ask for APC.
- **Draft page alignment.** `VLLM_MAMBA_NONE_SMALL_BLOCK=1` gives the draft attention-sized pages.

## 0.6.3

Prefill runs as chunked prefill with one windowed retrieval per slide.

- **One retrieval per slide.** Every prefill slide recomposes the working window
  as a fixed sink band, a fixed recent band and a scored middle filled to
  `select_budget`. The chunks after a slide keep the text right before them and
  the long-range evidence in the same window.
- **No probe forward.** The queries that score the middle come from the capture
  buffers the previous public forward already wrote, so a slide costs a slice and
  a clone instead of a private forward. A 204,951-token prefill drops from 351 s
  to 220.8 s while still selecting the full `B` on every slide.
- **`gen_budget` sets the slide distance.** The window slides by
  `R − lookahead − block_tokens`, so keep `R` at 8,192 or above; at 4,096 the
  window refilled every ~2K tokens and the refresh count roughly doubled.
- **One band pair.** `kvmem.sink_blocks` / `recent_blocks` size the head and
  tail kept on every retrieval, prefill and decode alike;
  `prefill_select_tokens` / `prefill_recent_tokens` bound the prefill retention
  when a smaller window is wanted.

## 0.6.2

Image requests through KVMem session reuse.

- **Multimodal requests.** Image inputs are prefilled by vLLM's own vision tower
  and the resulting KV enters the KVMem window, so image conversations reuse
  sessions like text ones. M-RoPE positions keep vLLM's native per-axis values
  and follow the window offset; text requests are unaffected.
- **Vision tower is a launch choice.** `mm_image_limit` (default 0) sets
  `--limit-mm-per-prompt` for images; text-only launches keep
  `--language-model-only`. `mm_gdn_prefill_backend` selects an alternate GDN
  prefill backend for hybrid checkpoints and defaults to vLLM's own choice.

## 0.6.1

Concurrent decode and multi-session reuse.

- **Concurrent decode.** Independent sessions are batched into one decode step
  instead of queueing behind each other, so two sessions stream tokens at the
  same time. Requests that share a session still serialize: the window is per
  session. Window refresh, compression and archive work stay serial too — only
  decode batches.
- **Multi-session reuse.** Each active session keeps its window in a host
  checkpoint, so alternating between conversations resumes without a cold
  prefill.
- **Capture sizes are derived, not configured.** `cudagraph_capture_sizes` now
  defaults to `(draft_tokens + 1) × 1..max_num_seqs`, the exact set of reachable
  decode batch sizes. Raising `max_num_seqs` no longer risks a stale hand-written
  set that silently drops decode back to eager. Startup also rejects a
  `max_num_batched_tokens` too small to admit one MTP step per sequence.
- **Concurrency costs KV pool.** Every running sequence holds `B + R` tokens, so
  window budget, `max_num_seqs` and the pool size have to be sized together.

## 0.6.0

Bounded, query-selected memory above the KV pool, on 2× RTX 2080 Ti (SM75).

- **Context above the KV pool.** KVMem keeps a bounded window resident and
  archives the rest in host RAM, so `max_model_len` can exceed the pool. A
  218,220-token prompt against a 205,752-token pool answered correctly; the
  repeat hit in 1.36 s.
- **Session checkpoints.** A session keeps its window on the host, so a
  continuation resumes without a cold prefill: 7.5 s → 0.33 s over the window,
  2.4 s → 0.25 s under it. Alternating sessions each hit in 0.25–0.40 s.
- **Automatic sessions.** The session id is derived from the conversation
  opening, so prefix reuse needs no frontend change.
- **SM75 full-graph MTP decode.** `execution_mode=full_graph` with MTP drafts:
  80–87 tok/s code decode, within noise of the pristine engine (80.8–87.2);
  prefill 1,021–1,063 tok/s (pristine 1,174).
- **Long generation past the window.** 9,000 tokens at 100.5 tok/s with two
  bounded refreshes, no stall.
- **Engine patch.** One 15-line change to the `vLLM-2080Ti-Definitive` fork lets
  `max_model_len` exceed the KV pool when KVMem bounds the window; both patches
  ship under `patches/`.

Concurrency was admitted but scheduled serially in this line; 0.6.1 brings real
decode batching across sessions.
