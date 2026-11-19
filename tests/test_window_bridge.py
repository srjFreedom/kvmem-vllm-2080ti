from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from vllm_kvmem.serving.window_batch import WindowBatch, contextual_metadata
from vllm_kvmem.core.metadata import WindowLengths
from vllm_kvmem.serving.window_bridge import install_window_bridge


def test_instance_bridge_preserves_recurrent_inputs_and_restores_context():
    # Text-only: 1D per-token batch positions.
    batch = SimpleNamespace(positions=torch.tensor([100, 200]))
    window_positions = torch.tensor([4, 5])
    lens = torch.tensor([5, 6], dtype=torch.int32)
    window = WindowBatch(window_positions, {0: WindowLengths(lens, lens, 6, window_positions)})

    class Native:
        def prepare_attn(self, received):
            assert received is batch
            assert received.positions.tolist() == [100, 200]
            return contextual_metadata(None)

        def prepare_inputs(self, received, state):
            return {"positions": received.positions, "marker": state}

    native = Native()
    translated = []

    def translate(positions, received):
        translated.append((positions, received))
        return torch.tensor([4, 5])

    install_window_bridge(native, lambda _: window, translate)
    metadata = native.prepare_attn(batch)
    assert metadata.get_extra_common_attn_kwargs(0, 2)["positions"] is window_positions
    assert metadata.get_extra_common_attn_kwargs(1, 2) == {}
    result = native.prepare_inputs(batch, "unchanged")
    # 1D RoPE keeps the dense window positions; the axis translator is unused.
    assert result["positions"].tolist() == [4, 5]
    assert result["marker"] == "unchanged"
    assert translated == []
    assert contextual_metadata(None) is None
    with pytest.raises(ValueError, match="already"):
        install_window_bridge(native, lambda _: window, translate)


def test_instance_bridge_multimodal_mrope_positions_are_translated_per_axis():
    # M-RoPE: the native batch carries dense 1D positions, and prepare_inputs
    # returns the axis-major [3, T] tensor built from the prompt's media grid.
    batch = SimpleNamespace(positions=torch.tensor([100, 200]))
    native_positions = torch.tensor([[100, 200], [101, 201], [102, 202]])
    window_positions = torch.tensor([4, 5])
    lens = torch.tensor([5, 6], dtype=torch.int32)
    window = WindowBatch(window_positions, {0: WindowLengths(lens, lens, 6, window_positions)})

    class Native:
        def prepare_attn(self, received):
            return contextual_metadata(None)

        def prepare_inputs(self, received, state):
            return {"positions": native_positions, "marker": state}

    seen = []

    def translate(positions, received):
        seen.append(received)
        return torch.tensor([[4, 5], [5, 6], [6, 7]])

    native = Native()
    install_window_bridge(native, lambda _: window, translate)
    result = native.prepare_inputs(batch, "unchanged")
    assert seen == [batch]
    assert result["positions"].tolist() == [[4, 5], [5, 6], [6, 7]]
    assert result["marker"] == "unchanged"
