"""Instance-scoped bridge to native model-state metadata and position inputs."""
from types import MethodType

from .window_batch import attention_windows


def install_window_bridge(model_state, current_window, translate_positions):
    """The runner supplies its current batch; native recurrent metadata is intact.

    `translate_positions(native, batch)` shifts native M-RoPE positions [AXES, T]
    into the dense window. It is resolved lazily because graph buffers are
    reserved after this bridge is installed.
    """
    if hasattr(model_state, "_kvmem_original_prepare_attn"):
        raise ValueError("Window bridge already installed")
    model_state._kvmem_original_prepare_attn = model_state.prepare_attn
    model_state._kvmem_original_prepare_inputs = model_state.prepare_inputs

    def prepare_attn(self, input_batch, *args, **kwargs):
        window = current_window(input_batch)
        with attention_windows(window.groups if window is not None else None):
            return self._kvmem_original_prepare_attn(input_batch, *args, **kwargs)

    def prepare_inputs(self, input_batch, req_states):
        result = self._kvmem_original_prepare_inputs(input_batch, req_states)
        window = current_window(input_batch)
        if window is None:
            return result
        from ..adapters.original_rope import enabled
        if enabled():
            # Attention lengths/slots still use window.groups. Only model
            # RoPE inputs retain the native original-history positions.
            return result
        positions = result.get("positions")
        if positions is not None and positions.ndim == 2:
            # vLLM built this axis-major tensor from the request's own media
            # grids. Shifting every axis by the request's window offset keeps
            # the M-RoPE geometry that a 1D expansion would collapse.
            return {**result, "positions": translate_positions(positions, input_batch)}
        # 1D RoPE keeps the dense per-token window positions.
        return {**result, "positions": window.positions}

    model_state.prepare_attn = MethodType(prepare_attn, model_state)
    model_state.prepare_inputs = MethodType(prepare_inputs, model_state)
