# Credits and provenance

This repository is a **patch set on top of other people's work**. Everything that works well here is
theirs; the SM75 adaptation, the deployment profiles and the verification data are ours.

## Upstream projects we build on

| project | what we use from it | license | our change |
|---|---|---|---|
| [`kvmem/kvmem-vllm`](https://github.com/kvmem/kvmem-vllm) | the whole mechanism: `vllm_kvmem` package, KVMem scheduler, `KVMemConnector`, CPU archive, sessions, request planning, prefix images, docs | Apache-2.0 | `src/vllm_kvmem` compatibility edits (see below) |
| [`vLLM`](https://github.com/vllm-project/vllm) `v0.2.2.post3` | model execution, paged attention, scheduler, spec decode (MTP), CUDA graphs, API server | Apache-2.0 | one 15-line patch in `vllm/v1/core/kv_cache_utils.py` (`KVMEM_SKIP_KV_CHECK`) |
| `vLLM-2080Ti-Definitive` (private fork of vLLM, SM75 enablement) | FlashInfer on SM75, FP8 weight-only (Marlin) path, the SM75 full-CUDA-graph recipe for MTP, MTP draft contract, `VLLM_SM75_SPEC_SYNC_MODE` / `VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH` | Apache-2.0 | we use it as the engine; no changes beyond the patch above |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | paged prefill/decode kernels | Apache-2.0 | none |
| [Triton](https://github.com/triton-lang/triton) | GDN / M-RoPE / assembly kernels | MIT | none |
| [PyTorch](https://github.com/pytorch/pytorch) | tensors, CUDA streams | BSD-3-Clause | none |
| [Qwen](https://github.com/QwenLM) model family | the served checkpoint and its MTP head | model license | none |

`LICENSE` in this repository is the upstream `kvmem-vllm` Apache-2.0 license and applies to the whole
tree. The upstream README is preserved at [docs/sm75/upstream-README.md](upstream-README.md).

## What we actually changed

`src/vllm_kvmem` (mechanism, ~700 lines total across the patch):

| change | why |
|---|---|
| `compat_sm75/` + `core/native_pages.py` block-row handling | this fork reports block rows differently (`num_blocks` may be absent) |
| `dtype=float16` support (launcher, capture buffers, activation check) | SM75 has no bfloat16 |
| `auto_session_id` in `serving/http_sessions.py` | reuse without any frontend change: namespace = hash of the conversation opening |
| page-pool demand charged in guided-block units (`serving/capacity.py`) | the native page is 32,768 tokens; charging it made a window-sized pool impossible |
| `KVMEM_WINDOW_TOKENS` export | lets the engine size admission for B+R instead of CTX |
| session-miss probes | `KVMEM_SESSION_MISS_LOG`, ten named miss codes |
| `deploy/`, `local/`, `scripts/` | launcher, systemd unit, profiles, watchdog |

Engine patch (`patches/0001-engine-skip-kv-check.patch`): a `KVMEM_SKIP_KV_CHECK` guard around vLLM's
"max_model_len must fit the KV pool" startup rejection. Without it the engine refuses to start even
though KVMem only needs the window.

## Things we deliberately did not take from other branches

An older private fork of the engine carried ~20 commits of a previous KVMem integration
(+2,407/−147 lines). None of it is used here: this tree references zero fork-only fields and the
engine stays one patch away from upstream `v0.2.2.post3`. That keeps the port reviewable and lets us
track upstream instead of a divergent book.
