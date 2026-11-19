import pytest

from vllm_kvmem.serving.admission import QueryPolicy


def test_planned_batch_can_span_multiple_g_windows_without_growing_any_owner():
    from vllm_kvmem.serving.admission import validate_target_batch_budget
    from vllm_kvmem.serving.request_plan import RequestPlan
    validate_target_batch_budget(32,4096,16424,8,5,request_planning=True)
    with pytest.raises(ValueError):validate_target_batch_budget(32,4096,16424,8,5)
    with pytest.raises(ValueError):validate_target_batch_budget(32,4096,32809,8,5,request_planning=True)
    plan=RequestPlan(None,block_tokens=32,gen_budget=4096,output_tokens=16384,rolling_active_window=True)
    admitted=[]
    for history in (16384,17408,18432,19456,20000,20128,20256,20416):
        n=plan.admit(history,16424,110000,0,20480)
        assert 0<n<=4096 and history+n<=20480
        admitted.append(n)
    assert sum(admitted)>4096  # Aggregate progress is not a per-owner allowance.


def test_mainline_prefill_stops_at_k_and_each_query_has_a_block_checkpoint():
    policy = QueryPolicy(128, 131072, 65536)
    assert policy.admit(130944, 1024, 197110, 0) == 128
    assert policy.refresh(131072, 197110, 0)
    assert policy.admit(131072, 1024, 197110, 0) == 1024
    assert policy.admit(132096, 1000, 197110, 1024) == 896
    assert policy.admit(196608, 1024, 197110, 65536) == 502
    assert policy.admit(0, 15, 15, 0) == 15
    assert policy.admit(0, 63, 10000, 0) == 0  # Let ready decodes use the small remainder.


def test_target_decode_refreshes_before_capacity_and_rejects_a_missed_boundary():
    policy = QueryPolicy(128, 131072, 65536)
    offset = 65536
    history = offset + policy.capacity - 128
    assert not policy.refresh(history - 1, 197110, offset)
    assert policy.refresh(history, 197110, offset)
    assert policy.admit(history, 1, 197110, offset) == 1
    with pytest.raises(RuntimeError, match="without"):
        policy.admit(offset + policy.capacity, 1, 197110, offset)
    with pytest.raises(ValueError):
        policy.admit(100, 1, 200, 101)


def test_repeated_prefill_queries_keep_live_extent_bounded_without_shortening_history():
    policy = QueryPolicy(128, 131072, 65536)
    prompt = 300123
    history = offset = 0
    refreshes = 0
    while history < prompt:
        tokens = policy.admit(history, 1024, prompt, offset)
        assert tokens > 0
        assert history - offset + tokens <= policy.capacity
        if policy.refresh(history, prompt, offset):
            offset = history - policy.select_budget
            refreshes += 1
        history += tokens
    assert history == prompt and refreshes > 100
    assert history - offset <= policy.select_budget + 1024


def test_planless_prefill_keeps_recency_until_the_last_chunk():
    from vllm_kvmem.serving.admission import refresh_selection
    # Mid-prefill: no decode query exists yet, so the window slides by recency.
    assert refresh_selection(planned=False, remaining=100000, action=None, max_batched_tokens=2048) == 'pressure'
    # The refresh inside the last chunk is the one retrieval that fills B.
    assert refresh_selection(planned=False, remaining=1024, action=None, max_batched_tokens=2048) == 'chunk'
    # Decode: nothing left to prefill, keep the semantic selection.
    assert refresh_selection(planned=False, remaining=0, action=None, max_batched_tokens=2048) == 'chunk'
    # Planned requests keep their explicit policy.
    assert refresh_selection(planned=True, remaining=100000, action='pressure', max_batched_tokens=2048) == 'pressure'
    assert refresh_selection(planned=True, remaining=1024, action=None, max_batched_tokens=2048) == 'chunk'


def test_prefill_capacity_argument_triggers_a_slide_earlier():
    policy = QueryPolicy(128, 8192, 4096)               # capacity 12,288
    assert policy.refresh(12288, 20000, 0) is True       # default capacity: full window
    assert policy.refresh(8192, 20000, 0) is False
    # A capped prefill extent slides at the same history earlier, and a wider
    # one later; the default stays untouched.
    assert policy.refresh(8192, 20000, 0, capacity=8192) is True
    assert policy.refresh(12288, 20000, 0, capacity=16384) is False


def test_semantic_prefill_keeps_the_query_picker_mid_prefill():
    from vllm_kvmem.serving.admission import refresh_selection
    assert refresh_selection(planned=False, remaining=80000, action=None,
                             max_batched_tokens=2048) == 'pressure'
    assert refresh_selection(planned=False, remaining=80000, action=None,
                             max_batched_tokens=2048, semantic_prefill=True) == 'chunk'


def test_selection_config_bands_can_be_zeroed_for_a_prefill_slide():
    from dataclasses import make_dataclass
    from vllm_kvmem.serving.request_plan import selection_config
    Memory = make_dataclass('Memory', [('select_budget', int), ('sink_blocks', int),
                                       ('recent_blocks', int), ('block_tokens', int),
                                       ('gen_budget', int)])
    cfg = Memory(select_budget=53248, sink_blocks=512, recent_blocks=32,
                 block_tokens=32, gen_budget=4096)
    step = selection_config(cfg, 16384, selection='query', sink_blocks=0, recent_blocks=0)
    assert (step.sink_blocks, step.recent_blocks, step.select_budget) == (0, 0, 16384)
    final = selection_config(cfg, 53248, selection='chunk')
    assert (final.sink_blocks, final.recent_blocks) == (512, 32)
