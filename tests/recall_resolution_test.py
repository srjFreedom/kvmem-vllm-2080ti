#!/usr/bin/env python3
"""Scorer resolution guard: a strongly aligned needle block must beat
orthogonal filler by a healthy margin, and must hold rank 0."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import vllm_kvmem_ext as ext
from kvmem_recall import KvmemRecall


class FakeStats:
    pass


def main():
    L, H, D, NB = 1, 2, 64, 100
    K = H * D
    st = ext.RawKvStore(L, K, 64)
    needle = np.zeros(K, np.float32); needle[0] = 5.0; needle[D] = 5.0
    orth = np.zeros(K, np.float32); orth[1] = 5.0; orth[D + 1] = 5.0
    st.write_layer_mean_sum(0, 64, 0, needle)
    for b in range(1, NB):
        st.write_layer_mean_sum(b * 64, 64, 0, orth)
    fs = FakeStats(); fs._store = st; fs.layers = L
    fs.frontier = NB * 64; fs.kv_heads = H; fs.head_dim = D
    fs.block_tokens = 64
    Q = np.zeros((1, H, D), np.float32); Q[0, :, 0] = 1.0
    rc = KvmemRecall(fs, block_tokens=64, sink_blocks=0, recent_blocks=0,
                     budget_blocks=NB, aggregation="mean", scorer="query")
    out = list(rc.scores({0: Q}, 1, H, H, D, NB))
    assert out, f"no scores: {rc.last_error}"
    rank = sum(1 for v in out if v > out[0])
    print(f"needle={out[0]:.6f} filler={out[1]:.6f} rank={rank} "
          f"margin={out[0]-max(out[1:]):.6f}")
    assert rank == 0, "aligned needle lost rank 0 - scorer directionality broken"
    # resolution floor (measured 2026-11-09: margin ~1e-4 for a 5-sigma
    # synthetic needle; below 1e-6 even a correct direction drowns in
    # the softmax share noise of a real 2000-block document)
    assert (out[0] - max(out[1:])) > 1e-6, "needle margin collapsed"
    print("recall resolution: ALL PASS")


if __name__ == "__main__":
    main()
