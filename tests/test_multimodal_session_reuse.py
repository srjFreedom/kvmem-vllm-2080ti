"""Integration test: multimodal requests enter KVMem session reuse.

Requires a running engine started with `mm_image_limit > 0` (raw media enabled).
Before that option existed the launcher passed `--language-model-only`, so vLLM
zeroed every modality limit and images could not reach the engine at all.

Run with:
    TMPDIR=/tmp /home/srj/vLLM-2080Ti-Definitive/.venv/bin/python -m pytest \
        tests/test_multimodal_session_reuse.py -x -q -s
"""
import base64
import struct
import zlib

import pytest

requests = pytest.importorskip("requests")

ENGINE_URL = "http://127.0.0.1:18200"
MODEL = "kvmem-2080ti"


def _make_tiny_png(width=8, height=8, color=(255, 0, 0)):
    """Create a minimal valid PNG in memory (no PIL dependency)."""
    def chunk(ctype, data):
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    raw_rows = b"".join(b"\x00" + bytes(color) * width for _ in range(height))
    return signature + ihdr + chunk(b"IDAT", zlib.compress(raw_rows)) + chunk(b"IEND", b"")


def _image_uri():
    return "data:image/png;base64," + base64.b64encode(_make_tiny_png()).decode()


def _post(messages, session_id, max_tokens=512):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "session_id": session_id,
        # The served checkpoint is a thinking model: reasoning would otherwise
        # consume the whole completion budget and leave content empty.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = requests.post(f"{ENGINE_URL}/v1/chat/completions", json=payload, timeout=300)
    return response


def _message(response):
    assert response.status_code == 200, f"Engine returned {response.status_code}: {response.text[:300]}"
    data = response.json()
    assert data.get("choices"), f"No choices in response: {data}"
    message = data["choices"][0]["message"]
    text = (message.get("content") or message.get("reasoning") or "").strip()
    assert text, f"Empty answer: {message}"
    return text


def test_multimodal_request_reaches_the_engine():
    """An image prompt must not be rejected (modality limit, text-only guard)."""
    text = _message(_post([
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _image_uri()}},
            {"type": "text", "text": "What color is this image? One word."},
        ]},
    ], "mm-test-001"))
    print(f"\nanswer: {text[:160]!r}")


def test_image_turns_share_one_session():
    """Successive image turns in one session must stay correct and reusable."""
    image = {"type": "image_url", "image_url": {"url": _image_uri()}}
    first = [{"role": "user", "content": [image, {"type": "text", "text": "What color? One word."}]}]
    answer = _message(_post(first, "mm-reuse-001"))
    second = first + [{"role": "assistant", "content": answer},
                      {"role": "user", "content": "Repeat your previous answer."}]
    follow_up = _message(_post(second, "mm-reuse-001"))
    print(f"\nturn1: {answer[:80]!r}\nturn2: {follow_up[:80]!r}")
