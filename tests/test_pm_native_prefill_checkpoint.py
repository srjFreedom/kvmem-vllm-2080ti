"""The pm policy keeps a checkpoint below the prompt for native chat requests.

A native chat request carries no request plan, so the plan-anchored prefill point does not
exist for it. Before this fallback the pm policy captured nothing below the prompt, and any
later prompt that did not repeat the previous answer (an aborted turn, an empty answer, an
identical retry) was shorter than the only checkpoint and had to prefill again.
"""
from vllm_kvmem.serving.session_control import SessionLedger, SessionNotice
from test_session_control import request


def ledger(**kwargs):
    events = []
    control = SessionLedger(2, 4, lambda *args: events.append(('prepare', args)),
        lambda *args: events.append(('abort', args)),
        lambda *args: events.append(('commit', args)), **kwargs)
    return control, events


def test_pm_policy_captures_the_final_prefill_boundary_without_a_request_plan():
    control, _ = ledger(checkpoint_policy='pm')
    req = request('chat')                                   # computed 16, prompt 25, block 4
    assert control.lookup(req, 0) == 0
    control.allocated(req, 0)
    control.capture_before_final_prefill(req, 9)            # 16 + 9 == 25: the tail is the last chunk
    assert [capture.position for capture in control.take_commands().captures] == [16]


def test_a_partial_prefill_tail_captures_nothing():
    control, _ = ledger(checkpoint_policy='pm')
    req = request('chat')
    req.num_computed_tokens = 8
    assert control.lookup(req, 0) == 0
    control.allocated(req, 0)
    control.capture_before_final_prefill(req, 1)            # 8 + 1 < 25: prefill continues
    assert control.take_commands().captures == []


def test_a_position_must_be_aligned_and_strictly_inside_the_prompt():
    control, _ = ledger(checkpoint_policy='pm')
    for computed, prompt in ((18, 25), (0, 25)):            # unaligned / nothing computed yet
        req = request(f'chat{computed}', session=f'session{computed}')
        req.num_computed_tokens, req.num_prompt_tokens = computed, prompt
        assert control.lookup(req, 0) == 0
        control.allocated(req, 0)
        control.capture_before_final_prefill(req, prompt - computed)
        assert control.take_commands().captures == []


def test_an_identical_prompt_reuses_the_native_capture():
    control, _ = ledger(checkpoint_policy='pm')
    first = request('turn1')
    assert control.lookup(first, 0) == 0
    control.allocated(first, 0)
    control.capture_before_final_prefill(first, 9)
    commands = control.take_commands()
    binding, capture = commands.bindings[0], commands.captures[0]
    assert capture.position == 16
    control.finished(first)
    control.consume(SessionNotice('retired', binding.namespace, binding.generation,
                                  first.request_id, capture.position, 12))
    retry = request('turn2')                                 # identical prompt, no new answer
    assert control.lookup(retry, 0) == 16                    # import 16 tokens, prefill the last 9
