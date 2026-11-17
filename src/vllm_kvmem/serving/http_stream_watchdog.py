"""Abort a generation stream that went silent while its request is still running.

The engine can stop emitting SSE bytes for a live request: the request's output
parser (reasoning or tool-call structure) holds text until a marker closes, and a
model that fell into a repetition loop inside such a structure never closes it.
Measured on this machine: one 134K-token turn delivered 23 chunks (~30 tokens),
then 26,772 tokens were decoded over 300 s without a single byte reaching the
client.

From the client that is indistinguishable from a hung server, so its own body
timeout fires first (300 s in DSH's transport), and the client then retries,
which re-prefills the whole prompt (208 s here). This pure-ASGI middleware turns
that into a fast failure: once a response has started streaming and produced no
body bytes for ``idle_seconds``, it logs what it knows, cancels the downstream
application (which aborts the engine request) and ends the body without a
terminal chunk. The client sees a truncated stream, classifies it as a transport
failure, and retries immediately.

The silence before the first body byte is deliberately not watched: a chunked
prefill of a 134K-token prompt legitimately takes minutes, and requests that do
not stream (one JSON body) never wait for a later byte at all.

Configuration comes from the launch environment (see serving/serve.py,
"stream_watchdog"): ``KVMEM_STREAM_WATCHDOG={"idle_seconds": 60}``; 0 disables it.
"""
import asyncio
import contextlib
import json
import logging
import re
import time

from ..core import settings as _settings

logger = logging.getLogger(__name__)

DEFAULTS = {'idle_seconds': 60.0}
_GENERATE_PREFIXES = ('/v1/chat/completions', '/v1/completions')
_REQUEST_ID = re.compile(rb'"id"\s*:\s*"([^"]{1,200})"')


def validate_config(config):
    """Only the documented knob; a typo must fail the launch, not the first request."""
    if not isinstance(config, dict) or set(config) - set(DEFAULTS):
        raise ValueError('stream_watchdog accepts only idle_seconds')
    idle = config.get('idle_seconds', DEFAULTS['idle_seconds'])
    if isinstance(idle, bool) or not isinstance(idle, (int, float)) or idle < 0:
        raise ValueError('stream_watchdog.idle_seconds must be a non-negative number')
    return {'idle_seconds': float(idle)}


def load_config():
    raw = _settings.get('KVMEM_STREAM_WATCHDOG')
    if not raw:
        return dict(DEFAULTS)
    try:
        return validate_config(json.loads(raw))
    except (TypeError, ValueError) as error:
        raise ValueError('KVMEM_STREAM_WATCHDOG: %s' % error) from None


class _Tracker:
    """The response as the client sees it: timestamps, counts, and the terminal state."""

    def __init__(self, send):
        self._send = send
        self.started = False
        self.finished = False
        self.chunks = 0
        self.bytes_sent = 0
        self.last_body_at = 0.0
        self.content_type = ''
        self.request_id = '-'

    async def send(self, message):
        kind = message['type']
        if kind == 'http.response.start':
            self.started = True
            headers = {name.lower(): value for name, value in message.get('headers') or ()}
            self.content_type = headers.get(b'content-type', b'').decode('latin-1')
        elif kind == 'http.response.body':
            body = message.get('body') or b''
            if body:
                now = time.monotonic()
                if not self.chunks:
                    match = _REQUEST_ID.search(body)
                    if match:
                        self.request_id = match.group(1).decode('latin-1')
                self.chunks += 1
                self.bytes_sent += len(body)
                self.last_body_at = now
            if not message.get('more_body', False):
                self.finished = True
        await self._send(message)

    def wait_budget(self, idle_limit):
        """Seconds to wait before the stream counts as silent.

        None means the watchdog has nothing to watch: no body yet (prefill, or a
        response that is not streaming) or the body already ended.
        """
        if not self.chunks or self.finished:
            return None
        return idle_limit - (time.monotonic() - self.last_body_at)


class StreamWatchdogMiddleware:
    """Pure ASGI; watches generation responses for a silent stream body."""

    def __init__(self, app):
        self.app = app
        self.config = load_config()

    async def __call__(self, scope, receive, send):
        idle_limit = self.config['idle_seconds']
        if (idle_limit <= 0 or scope.get('type') != 'http'
                or not str(scope.get('path') or '').startswith(_GENERATE_PREFIXES)):
            await self.app(scope, receive, send)
            return
        tracker = _Tracker(send)
        # Polling keeps the two "nothing to watch yet" cases apart - before the first byte
        # (prefill) a stream may legitimately be silent for minutes, so the watchdog only
        # starts counting after the response body began.
        poll = max(min(idle_limit / 8.0, 1.0), 0.01)
        task = asyncio.ensure_future(self.app(scope, receive, tracker.send))
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=poll)
                if task in done:
                    await task      # re-raise the application's own exception
                    return
                budget = tracker.wait_budget(idle_limit)
                if budget is not None and budget <= 0:
                    await self._abort(scope, tracker, task)
                    return
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task

    async def _abort(self, scope, tracker, task):
        logger.warning(
            'stream watchdog: %s %s request=%s silent for %.1fs after %d chunks / %d bytes; '
            'aborting (the client sees a truncated stream and retries instead of timing out)',
            scope.get('method'), scope.get('path'), tracker.request_id,
            time.monotonic() - tracker.last_body_at, tracker.chunks, tracker.bytes_sent)
        task.cancel()
        with contextlib.suppress(BaseException):
            await task
        if not tracker.finished:
            tracker.finished = True
            with contextlib.suppress(BaseException):
                await tracker.send({'type': 'http.response.body', 'body': b'', 'more_body': False})
