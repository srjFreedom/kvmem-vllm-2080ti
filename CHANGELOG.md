# Changelog

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
