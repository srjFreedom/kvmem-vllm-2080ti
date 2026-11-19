"""FULL decode graphs with original KVMem authority and explicit refresh batches.

No semantic transaction is captured in a graph. Refresh commands dispatch an
ordinary uncaptured control batch (with compiled model forward when enabled),
so MeanK query/probe, GDN rollback, KV assembly and
connector acknowledgments retain the original ownership order.
"""
from types import MethodType

from .window_batch import WindowBatchAdapter
from ..adapters.cache_groups import full_attention_groups
from ..core.options import RuntimeOptions


def position_axes(runner):
    """Number of RoPE position axes the runtime actually produces.

    vLLM creates a RopeState, and therefore 2D `[num_dims, T]` mrope positions,
    only when `model_config.uses_mrope`. A model class can implement
    SupportsMRoPE while this deployment still runs 1D positions (for example
    `language_model_only` or a checkpoint without mrope sections), so the
    runtime state decides. Reserving more axes than the native layout would
    hand wrongly shaped positions to the RoPE path.
    """
    state = getattr(runner, 'model_state', None)
    rope_state = getattr(state, 'rope_state', None)
    if rope_state is not None:
        return int(rope_state.num_dims)
    model_config = runner.vllm_config.model_config
    if model_config.uses_mrope:
        return int(model_config.mrope_num_dims)
    return 1


def initialize_graph_window(runner):
    cfg = RuntimeOptions.from_vllm(runner.vllm_config).memory
    groups = full_attention_groups(runner.kv_cache_config)
    adapter = WindowBatchAdapter(groups, cfg.select_budget + cfg.gen_budget)
    adapter.reserve_graph_buffers(runner.max_num_tokens, runner.max_num_reqs, runner.device,
                                  position_axes=position_axes(runner))
    runner._kvmem_window_adapter = adapter
    # Archive headers carry one position per token, which is the dense 1D window
    # buffer; the view shares its storage, so it tracks each translated batch.
    for record in runner._kvmem_graph_captures:
        record.positions = adapter.buffers[0]
    manager = runner.cudagraph_manager
    if not hasattr(manager, '_kvmem_original_fullgraph'):
        manager._kvmem_original_fullgraph = manager.run_fullgraph
        def run_fullgraph(self, descriptor):
            if (runner._kvmem_probe_output / 'armed').is_file():
                runner._kvmem_graph_decode_calls = getattr(runner, '_kvmem_graph_decode_calls', 0) + 1
            return self._kvmem_original_fullgraph(descriptor)
        manager.run_fullgraph = MethodType(run_fullgraph, manager)
    state = runner.model_state
    # Native graph capture builds inputs outside prepare_inputs(). Use the same
    # stable dense-position buffer in both capture and subsequent execution.
    if not hasattr(state, '_kvmem_original_dummy_inputs'):
        state._kvmem_original_dummy_inputs = state.prepare_dummy_inputs
        def prepare_dummy_inputs(self, num_reqs, num_tokens):
            result = self._kvmem_original_dummy_inputs(num_reqs, num_tokens)
            adapter = runner._kvmem_window_adapter
            if adapter.model_positions is not None:
                # M-RoPE capture replays from the same axis-major window buffer
                # that prepare_inputs fills, so graph replay reads live values.
                positions = adapter.model_positions[:, :num_tokens]
            else:
                positions = adapter.buffers[0][:num_tokens]
            return {**result, 'positions': positions}
        state.prepare_dummy_inputs = MethodType(prepare_dummy_inputs, state)


def needs_refresh_batch(scheduler_output):
    from .connector import ControlMetadata
    control = scheduler_output.kv_connector_metadata
    # Request swaps (SwapCommand) copy pages around the forward like an
    # asynchronous restore; only synchronous refreshes need an eager batch.
    return isinstance(control, ControlMetadata) and any(
        not getattr(c, 'asynchronous', True) for c in control.commands)
