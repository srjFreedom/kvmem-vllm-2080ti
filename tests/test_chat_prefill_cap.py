"""A native chat prefill step is sized by chat_prefill_blocks, not by the planned chunk.

Native chat carries no request plan, so KVMEM_PREFILL_CHUNK_TOKENS never sized it: the
scheduler hardcoded eight blocks (256 tokens at block 32), which is what a 134K chat
prompt actually ran as (525 forwards at ~363 ms each).
"""
import pytest

from vllm_kvmem.serving.prefill_control import chat_prefill_cap
from vllm_kvmem.serving.serve import build_launch
from test_serving_launcher import profile as serving_profile


def test_the_default_is_eight_blocks(monkeypatch):
    monkeypatch.delenv('KVMEM_CHAT_PREFILL_BLOCKS', raising=False)
    assert chat_prefill_cap(32, 4096) == 256


@pytest.mark.parametrize('blocks,expected', [(8, 256), (32, 1024), (64, 2048)])
def test_the_cap_follows_the_configured_block_count(blocks, expected, monkeypatch):
    monkeypatch.setenv('KVMEM_CHAT_PREFILL_BLOCKS', str(blocks))
    assert chat_prefill_cap(32, 4096) == expected


def test_the_cap_never_eats_the_native_mtp_lookahead():
    for blocks in (128, 1000):
        assert chat_prefill_cap(32, 4096, blocks) == 4064     # G minus one block
    assert chat_prefill_cap(32, 4096, 4096) == 4064


def test_invalid_values_are_refused(monkeypatch):
    for bad in ('0', '-1'):
        monkeypatch.setenv('KVMEM_CHAT_PREFILL_BLOCKS', bad)
        with pytest.raises(ValueError):
            chat_prefill_cap(32, 4096)
    with pytest.raises(ValueError):
        chat_prefill_cap(0, 4096, 8)


def test_the_launch_records_and_validates_the_block_count(tmp_path):
    profile = serving_profile(tmp_path)
    plan = build_launch(profile, tmp_path, tmp_path/'run')
    assert plan['environment']['KVMEM_CHAT_PREFILL_BLOCKS'] == '8'
    profile['chat_prefill_blocks'] = 64
    plan = build_launch(profile, tmp_path, tmp_path/'run64')
    assert plan['environment']['KVMEM_CHAT_PREFILL_BLOCKS'] == '64'
    assert plan['profile']['chat_prefill_blocks'] == 64
    profile['chat_prefill_blocks'] = 1024                     # 1024 blocks do not fit G
    with pytest.raises(ValueError, match='fit G'):
        build_launch(profile, tmp_path, tmp_path/'too_big')
