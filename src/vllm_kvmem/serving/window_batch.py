"""Translate working positions without changing native request/SSM history.

One per-request offset maps the *next dense window position* to logical history.
Historical selected blocks may be noncontiguous; assembly owns that mapping.
This adapter only maps newly appended tokens after an acknowledged refresh.
"""
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ..core.metadata import WindowLengths, WorkingSetMetadata
from ..core import settings as _settings

_active_lengths = ContextVar("kvmem_attention_window_lengths", default=None)


@contextmanager
def attention_windows(groups):
    token = _active_lengths.set(groups)
    try:
        yield
    finally:
        _active_lengths.reset(token)


def contextual_metadata(native):
    groups = _active_lengths.get()
    return WorkingSetMetadata(native, groups) if groups else native


@triton.jit
def _translate(Positions, QueryStart, Offsets, SeqLens, WindowPos, WindowLens,
               NUM_REQS: tl.constexpr, NUM_TOKENS: tl.constexpr,
               PAD_TOKENS: tl.constexpr, IN_STRIDE: tl.constexpr,
               OUT_STRIDE: tl.constexpr, AXES: tl.constexpr, BLOCK: tl.constexpr):
    req = tl.program_id(0)
    begin = tl.load(QueryStart + req)
    end = tl.load(QueryStart + req + 1)
    offset = tl.load(Offsets + req)
    length = tl.load(SeqLens + req)
    tl.store(WindowLens + req, length - offset)
    if AXES == 1:
        # 1D [T]: flat index = token index
        for start in range(begin, end, BLOCK):
            ids = start + tl.arange(0, BLOCK)
            pos = tl.load(Positions + ids, ids < end, other=0)
            tl.store(WindowPos + ids, pos - offset, ids < end)
    else:
        # 2D [AXES, T] (axis-major, vLLM MRoPE layout). Both tensors are slices
        # of wider buffers, so the row stride is the buffer width, not T.
        for a in range(AXES):
            in_base = a * IN_STRIDE
            out_base = a * OUT_STRIDE
            for start in range(begin, end, BLOCK):
                ids = start + tl.arange(0, BLOCK)
                pos = tl.load(Positions + in_base + ids, ids < end, other=0)
                tl.store(WindowPos + out_base + ids, pos - offset, ids < end)
    if req == 0:
        if AXES == 1:
            for start in range(NUM_TOKENS, PAD_TOKENS, BLOCK):
                ids = start + tl.arange(0, BLOCK)
                tl.store(WindowPos + ids, 0, ids < PAD_TOKENS)
        else:
            for a in range(AXES):
                out_base = a * OUT_STRIDE
                for start in range(NUM_TOKENS, PAD_TOKENS, BLOCK):
                    ids = start + tl.arange(0, BLOCK)
                    tl.store(WindowPos + out_base + ids, 0, ids < PAD_TOKENS)


@triton.jit(do_not_specialize=['NUM_TOKENS', 'PAD_TOKENS', 'IN_STRIDE', 'OUT_STRIDE'])
def _translate_dynamic(Positions, QueryStart, Offsets, SeqLens, WindowPos, WindowLens,
                       NUM_TOKENS, PAD_TOKENS, IN_STRIDE, OUT_STRIDE,
                       AXES: tl.constexpr, BLOCK: tl.constexpr):
    # Same program as _translate. The token counts and row strides are runtime
    # arguments, so a new batch shape reuses the one compiled kernel.
    req = tl.program_id(0)
    begin = tl.load(QueryStart + req)
    end = tl.load(QueryStart + req + 1)
    offset = tl.load(Offsets + req)
    length = tl.load(SeqLens + req)
    tl.store(WindowLens + req, length - offset)
    if AXES == 1:
        for start in range(begin, end, BLOCK):
            ids = start + tl.arange(0, BLOCK)
            pos = tl.load(Positions + ids, ids < end, other=0)
            tl.store(WindowPos + ids, pos - offset, ids < end)
    else:
        for a in range(AXES):
            in_base = a * IN_STRIDE
            out_base = a * OUT_STRIDE
            for start in range(begin, end, BLOCK):
                ids = start + tl.arange(0, BLOCK)
                pos = tl.load(Positions + in_base + ids, ids < end, other=0)
                tl.store(WindowPos + out_base + ids, pos - offset, ids < end)
    if req == 0:
        if AXES == 1:
            for start in range(NUM_TOKENS, PAD_TOKENS, BLOCK):
                ids = start + tl.arange(0, BLOCK)
                tl.store(WindowPos + ids, 0, ids < PAD_TOKENS)
        else:
            for a in range(AXES):
                out_base = a * OUT_STRIDE
                for start in range(NUM_TOKENS, PAD_TOKENS, BLOCK):
                    ids = start + tl.arange(0, BLOCK)
                    tl.store(WindowPos + out_base + ids, 0, ids < PAD_TOKENS)


@triton.jit(do_not_specialize=['NUM_TOKENS', 'PAD_TOKENS', 'IN_STRIDE', 'OUT_STRIDE'])
def _translate_axes(Positions, QueryStart, Offsets, WindowPos,
                    NUM_TOKENS, PAD_TOKENS, IN_STRIDE, OUT_STRIDE,
                    AXES: tl.constexpr, BLOCK: tl.constexpr):
    """Shift native M-RoPE positions [AXES, T] into the dense window, per axis.

    Both tensors are slices of wider buffers, so each row's start is a stride
    multiple rather than a token-count multiple.
    """
    req = tl.program_id(0)
    begin = tl.load(QueryStart + req)
    end = tl.load(QueryStart + req + 1)
    offset = tl.load(Offsets + req)
    for a in range(AXES):
        in_base = a * IN_STRIDE
        out_base = a * OUT_STRIDE
        for start in range(begin, end, BLOCK):
            ids = start + tl.arange(0, BLOCK)
            pos = tl.load(Positions + in_base + ids, ids < end, other=0)
            tl.store(WindowPos + out_base + ids, pos - offset, ids < end)
    if req == 0:
        for a in range(AXES):
            out_base = a * OUT_STRIDE
            for start in range(NUM_TOKENS, PAD_TOKENS, BLOCK):
                ids = start + tl.arange(0, BLOCK)
                tl.store(WindowPos + out_base + ids, 0, ids < PAD_TOKENS)


def translate_dynamic():
    """KVMEM_WINDOW_TRANSLATE_DYNAMIC=1: one position kernel for every batch shape.

    _translate takes the request and token counts as compile-time constants, so
    each new (requests, tokens, padded tokens) triple compiles a kernel on the
    engine thread before the forward (40-76 ms). Prefill chunks that share a
    step with decoders have a new shape nearly every step.
    """
    value = _settings.get('KVMEM_WINDOW_TRANSLATE_DYNAMIC', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_WINDOW_TRANSLATE_DYNAMIC must be 0 or 1')
    return value == '1'


@dataclass(frozen=True)
class WindowBatch:
    positions: torch.Tensor
    groups: dict[int, WindowLengths]


class WindowBatchAdapter:
    def __init__(self, attention_groups: tuple[int, ...], capacity: int):
        if not attention_groups or len(set(attention_groups)) != len(attention_groups) or capacity <= 0:
            raise ValueError("Invalid full-attention window configuration")
        self.attention_groups = attention_groups
        self.capacity = capacity
        self.private_capacities = {}
        self.offsets: dict[str, int] = {}
        self.history_ends: dict[str, int] = {}
        self.generations: dict[str, int] = {}
        self._generation = 0
        self.buffers = None
        self.position_axes = 1
        self.dynamic_translate = translate_dynamic()

    def reserve_graph_buffers(self, max_tokens, max_requests, device, position_axes=1):
        """Capture-time attention lengths and RoPE positions keep stable addresses.

        `buffers[0]` always holds the dense per-token window positions, which are
        the native batch positions shifted by each request's window offset. With
        `position_axes > 1` a second `[position_axes, max_tokens]` buffer carries
        window positions for M-RoPE models: vLLM hands those to the model as a
        2D axis-major tensor, and each axis needs the same offset applied.
        """
        if position_axes < 1:
            raise ValueError(f"position_axes must be positive, got {position_axes}")
        self.buffers = (
            torch.zeros(max_tokens, device=device, dtype=torch.int64),
            torch.zeros(max_requests, device=device, dtype=torch.int32),
            torch.zeros(max_requests, device=device, dtype=torch.int32),
        )
        self.position_axes = position_axes
        self.model_positions = (
            None if position_axes == 1 else
            torch.zeros(position_axes, max_tokens, device=device, dtype=torch.int64))
        self.offsets_cpu = torch.empty(max_requests, dtype=torch.int32, pin_memory=True)
        self.offsets_ready = None
        self.uploaded_offsets = None

    def _stage_offsets(self, batch, cpu_offsets):
        """Upload the per-request window offsets once per unchanged mapping."""
        if self.buffers is None:
            return cpu_offsets.to(batch.positions.device)
        gpu_offsets = self.buffers[2][:batch.num_reqs_after_padding]
        offset_key = tuple(cpu_offsets.tolist())
        if self.uploaded_offsets != offset_key:
            # A synchronous CPU copy stalls behind the preceding native draft
            # kernels. Offsets usually stay fixed throughout decode. Retain the
            # pinned source storage until its DMA has completed.
            if self.offsets_ready is not None and not self.offsets_ready.query():
                self.offsets_ready.synchronize()
            source = self.offsets_cpu[:batch.num_reqs_after_padding]
            source.copy_(cpu_offsets)
            gpu_offsets.copy_(source, non_blocking=True)
            self.offsets_ready = torch.cuda.Event()
            self.offsets_ready.record(torch.cuda.current_stream(batch.positions.device))
            self.uploaded_offsets = offset_key
        return gpu_offsets

    def translate_axes(self, native, batch):
        """Shift native M-RoPE positions into the dense window, axis by axis.

        vLLM builds the 2D axis-major tensor from the request's own media grid,
        so a multimodal prompt has per-axis values that a 1D expansion cannot
        represent. Every axis advances with the same window offset, so shifting
        the native tensor preserves the M-RoPE geometry exactly.
        """
        if self.model_positions is None:
            raise ValueError("Window batch reserved no M-RoPE position buffer")
        if native.ndim != 2 or native.shape[0] != self.position_axes:
            raise ValueError(
                f"Native positions carry {native.shape[0] if native.ndim == 2 else 1} "
                f"M-RoPE axes but the window buffer reserved {self.position_axes}")
        if native.shape[1] > self.model_positions.shape[1]:
            raise ValueError("Native M-RoPE positions exceed the reserved window buffer")
        t = native.shape[1]
        out = self.model_positions[:, :t]
        cpu_offsets = self._cpu_offsets(batch)
        gpu_offsets = self._stage_offsets(batch, cpu_offsets)
        _translate_axes[(batch.num_reqs,)](
            native, batch.query_start_loc, gpu_offsets, out,
            batch.num_tokens, t, native.stride(0), out.stride(0),
            self.position_axes, BLOCK=128)
        return out

    def _cpu_offsets(self, batch):
        offsets = [self.offsets.get(req, 0) for req in batch.req_ids]
        cpu_offsets = torch.zeros(batch.num_reqs_after_padding, dtype=torch.int32)
        cpu_offsets[:batch.num_reqs] = torch.tensor(offsets, dtype=torch.int32)
        return cpu_offsets

    def capture_window(self, batch):
        if self.buffers is None:
            raise RuntimeError('Graph window buffers were not reserved')
        positions, seq_lens, _ = self.buffers
        if batch.positions.ndim != 1:
            raise ValueError("Native batch positions must be dense 1D per-token positions")
        n, t = batch.num_reqs_after_padding, batch.num_tokens_after_padding
        positions[:t].copy_(batch.positions[:t])
        window_pos = positions[:t]
        seq_lens[:n].copy_(batch.seq_lens[:n])
        bounds = batch.seq_lens_cpu_upper_bound[:n]
        # Native FULL capture uses a worst-case extent. Do not bake a short
        # bootstrap window into the attention kernel's graph configuration.
        lens = WindowLengths(seq_lens[:n], bounds, max(self.capacity, int(bounds.max())), window_pos)
        return WindowBatch(window_pos, {g: lens for g in self.attention_groups})

    def commit(self, request_id: str, history_end: int, window_tokens: int):
        if not 0 < window_tokens <= min(history_end, self.capacity):
            raise ValueError("Invalid committed working window")
        offset = history_end - window_tokens
        if history_end < self.history_ends.get(request_id, 0):
            raise ValueError("Working window cannot precede the committed history")
        self.offsets[request_id] = offset
        self.history_ends[request_id] = history_end
        self._generation += 1
        self.generations[request_id] = self._generation

    def free(self, request_id):
        self.private_capacities.pop(request_id, None)
        self.offsets.pop(request_id, None)
        self.history_ends.pop(request_id, None)
        self.generations.pop(request_id, None)

    def prepare(self, batch) -> WindowBatch | None:
        offsets = [self.offsets.get(req, 0) for req in batch.req_ids]
        if not any(offsets) and self.buffers is None:
            return None
        if batch.positions.device.type != "cuda" or batch.positions.ndim != 1:
            raise ValueError("Working positions require native CUDA 1D per-token positions")
        if len(offsets) != batch.num_reqs:
            raise ValueError("Request order does not match native batch")
        if any(offset < 0 or offset > computed for offset, computed in
               zip(offsets, batch.num_computed_tokens_np)):
            raise ValueError("Native logical position precedes working-set commit")
        cpu_offsets = self._cpu_offsets(batch)
        bounds = batch.seq_lens_cpu_upper_bound - cpu_offsets
        caps = torch.full_like(bounds, self.capacity)
        for i, rid in enumerate(batch.req_ids):
            caps[i] = self.private_capacities.get(rid, self.capacity)
        if torch.any(bounds < 0).item() or torch.any(bounds > caps).item():
            raise ValueError("Native batch exceeds the admitted working-set extent")
        gpu_offsets = self._stage_offsets(batch, cpu_offsets)
        if self.buffers is None:
            positions = torch.empty_like(batch.positions)
            seq_lens = torch.zeros_like(batch.seq_lens)
        else:
            positions = self.buffers[0][:batch.num_tokens_after_padding]
            seq_lens = self.buffers[1][:batch.num_reqs_after_padding]
            seq_lens.zero_()
        if getattr(self, 'dynamic_translate', False):
            _translate_dynamic[(batch.num_reqs,)](
                batch.positions, batch.query_start_loc, gpu_offsets, batch.seq_lens,
                positions, seq_lens, batch.num_tokens, batch.num_tokens_after_padding,
                0, 0, 1, BLOCK=128)
        else:
            _translate[(batch.num_reqs,)](
                batch.positions, batch.query_start_loc, gpu_offsets, batch.seq_lens,
                positions, seq_lens, batch.num_reqs, batch.num_tokens,
                batch.num_tokens_after_padding, 0, 0, 1, BLOCK=128)
        lengths = WindowLengths(seq_lens, bounds, int(bounds.max().item()), positions)
        return WindowBatch(positions, {group: lengths for group in self.attention_groups})
