"""A generation stream that goes silent mid-request fails fast instead of waiting for the client."""
import asyncio
import json
import logging
import time

import pytest

from vllm_kvmem.serving.http_stream_watchdog import StreamWatchdogMiddleware, validate_config
from vllm_kvmem.serving.serve import build_launch
from test_serving_launcher import profile as serving_profile

SCOPE = {'type': 'http', 'method': 'POST', 'path': '/v1/chat/completions'}
FIRST = b'data: {"id":"chatcmpl-probe","choices":[{"delta":{"role":"assistant"}}]}\n\n'


class App:
    """A streaming app that can stall, before its first byte or after it."""

    def __init__(self, *, before_first=0.0, stall=0.0, chunks=3, gap=0.0):
        self.before_first, self.stall, self.chunks, self.gap = before_first, stall, chunks, gap
        self.cancelled = self.finished = False

    async def __call__(self, scope, receive, send):
        if self.before_first:
            await asyncio.sleep(self.before_first)
        await send({'type': 'http.response.start', 'status': 200,
                    'headers': [(b'content-type', b'text/event-stream')]})
        try:
            await send({'type': 'http.response.body', 'body': FIRST, 'more_body': True})
            if self.stall:
                await asyncio.sleep(self.stall)
            for _ in range(self.chunks):
                if self.gap:
                    await asyncio.sleep(self.gap)
                await send({'type': 'http.response.body', 'body': b'data: {}\n\n', 'more_body': True})
            await send({'type': 'http.response.body', 'body': b'data: [DONE]\n\n', 'more_body': True})
            await send({'type': 'http.response.body', 'body': b'', 'more_body': False})
            self.finished = True
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def run(app, idle, *, monkeypatch, path='/v1/chat/completions'):
    monkeypatch.setenv('KVMEM_STREAM_WATCHDOG', json.dumps({'idle_seconds': idle}))
    middleware = StreamWatchdogMiddleware(app)
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {'type': 'http.request'}

    started = time.monotonic()
    asyncio.run(middleware(dict(SCOPE, path=path), receive, send))
    return sent, time.monotonic() - started


def bodies(sent):
    return b''.join(message.get('body') or b'' for message in sent
                    if message['type'] == 'http.response.body')


def test_a_silent_stream_is_aborted_fast(monkeypatch, caplog):
    app = App(stall=30)
    with caplog.at_level(logging.WARNING):
        sent, elapsed = run(app, 0.2, monkeypatch=monkeypatch)
    assert elapsed < 2.0
    assert app.cancelled and not app.finished
    assert sent[0]['type'] == 'http.response.start'
    # The body ends without the application's terminal chunk: the client sees a truncated
    # stream (a transport failure it retries) instead of a completed answer.
    assert sent[-1] == {'type': 'http.response.body', 'body': b'', 'more_body': False}
    assert b'[DONE]' not in bodies(sent)
    assert 'stream watchdog' in caplog.text
    assert 'chatcmpl-probe' in caplog.text
    assert 'silent for' in caplog.text


def test_a_healthy_stream_passes_through(monkeypatch, caplog):
    app = App(chunks=4, gap=0.05)
    with caplog.at_level(logging.WARNING):
        sent, _ = run(app, 1.0, monkeypatch=monkeypatch)
    assert app.finished and not app.cancelled
    assert b'[DONE]' in bodies(sent)
    assert sent[-1] == {'type': 'http.response.body', 'body': b'', 'more_body': False}
    assert 'stream watchdog' not in caplog.text


def test_silence_before_the_first_byte_is_prefill_not_a_stall(monkeypatch):
    """A chunked prefill of a long prompt produces nothing for minutes; that is not a stall."""
    app = App(before_first=0.5, chunks=1)
    sent, elapsed = run(app, 0.1, monkeypatch=monkeypatch)
    assert app.finished and not app.cancelled
    assert elapsed >= 0.5
    assert b'[DONE]' in bodies(sent)


def test_other_routes_are_not_watched(monkeypatch):
    app = App(stall=0.4)
    sent, elapsed = run(app, 0.05, monkeypatch=monkeypatch, path='/v1/models')
    assert app.finished and not app.cancelled
    assert elapsed >= 0.4


def test_idle_zero_disables_the_watchdog(monkeypatch):
    app = App(stall=0.4)
    sent, _ = run(app, 0, monkeypatch=monkeypatch)
    assert app.finished and not app.cancelled


def test_config_validation_rejects_a_typo_and_accepts_a_disabled_limit():
    assert validate_config({'idle_seconds': 12.5}) == {'idle_seconds': 12.5}
    assert validate_config({}) == {'idle_seconds': 60.0}
    assert validate_config({'idle_seconds': 0}) == {'idle_seconds': 0.0}
    for bad in ({'idle': 1}, {'idle_seconds': -1}, {'idle_seconds': True},
                {'idle_seconds': '60'}, []):
        with pytest.raises(ValueError):
            validate_config(bad)


def test_the_launch_wires_the_watchdog_and_its_configuration(tmp_path):
    cfg = serving_profile(tmp_path)
    argv = build_launch(cfg, tmp_path, tmp_path/'run')['vllm_argv']
    wired = [argv[index + 1] for index, value in enumerate(argv) if value == '--middleware']
    assert 'vllm_kvmem.serving.http_stream_watchdog.StreamWatchdogMiddleware' in wired
    assert 'vllm_kvmem.serving.http_sessions.validate_prompt_session' in wired
    # The default limit needs no environment entry of its own.
    assert 'KVMEM_STREAM_WATCHDOG' not in build_launch(cfg, tmp_path, tmp_path/'def')['environment']
    cfg['stream_watchdog'] = {'idle_seconds': 15}
    plan = build_launch(cfg, tmp_path, tmp_path/'explicit')
    assert json.loads(plan['environment']['KVMEM_STREAM_WATCHDOG']) == {'idle_seconds': 15}
    assert plan['profile']['stream_watchdog'] == {'idle_seconds': 15.0}
    cfg['stream_watchdog'] = {'idle': 15}
    with pytest.raises(ValueError, match='idle_seconds'):
        build_launch(cfg, tmp_path, tmp_path/'typo')
