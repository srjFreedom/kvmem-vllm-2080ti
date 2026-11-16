"""Compatibility shims for the SM75 (2080 Ti) vLLM fork.

The target runtime is `vLLM-2080Ti-Definitive` (fork branch `kvmem-0.2.2-clean`).
Its vLLM interface differs from the upstream releases kvmem is written against in a
couple of object shapes. Every difference lives here so the upstream diff stays
in one greppable place; each helper documents the exact shape it bridges.

Measured drift (2026-11-16):
  * `MultiGroupBlockTable` in the fork exposes per-row counts as
    `block_tables[group].num_blocks_per_row` (numpy, one value per row) while
    upstream code reads `tables.num_blocks.np[group, index]` (2-D array).
  * `CaptureRuntime` in the fork can be driven without `runner` (mock/late-bind
    paths), while upstream dereferences `self.runner` unconditionally.
"""

from __future__ import annotations


def block_row_count(tables, group: int, index: int) -> int:
    """Number of used blocks for (group, row) on either block-table shape."""
    counts = getattr(tables, "num_blocks", None)
    if counts is not None:
        inner = getattr(counts, "np", None)
        if inner is not None:
            return int(inner[group, index])
        try:
            return int(counts[group][index])  # plain sequence of rows
        except Exception:
            pass
    groups = getattr(tables, "block_tables", None)
    if groups is None:
        raise ValueError("block table exposes neither num_blocks nor block_tables")
    group_table = groups[group]
    per_row = getattr(group_table, "num_blocks_per_row", None)
    if per_row is not None:
        return int(per_row[index])
    # Last resort: a table that only carries the device row (mock/qualification
    # fixtures build exactly this shape) - the used width IS the row width.
    gpu = getattr(group_table, "gpu", None)
    if gpu is not None:
        return int(gpu.shape[-1])
    raise ValueError("block table exposes neither num_blocks, num_blocks_per_row nor a gpu row")


def optional_runner(owner):
    """`owner.runner` when set, else None (upstream tolerates None downstream)."""
    return getattr(owner, "runner", None)
