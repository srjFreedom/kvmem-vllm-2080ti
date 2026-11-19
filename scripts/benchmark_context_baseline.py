"""Context-length baseline: prefill and decode throughput on the live engine.

One cold streaming request per length, 256 forced output tokens. Prefill and
decode time are the engine's own per-request histograms; client timestamps are
kept as a cross-check. Prompts are real repository text (documentation and
source), sliced to the requested token count and offset per run so no run
replays a previous prompt.

    python3 scripts/benchmark_context_baseline.py --lengths 8192 32768 65536 204800
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
METRIC = re.compile(r'^vllm:(\w+)(\{.*?\})?\s+([0-9.e+-]+)$', re.MULTILINE)


def snapshot(base):
    text = requests.get(base + '/metrics', timeout=30).text
    values = {}
    for name, labels, value in METRIC.findall(text):
        if labels and 'model_name="' not in labels:
            continue
        key, labels = name, labels or ''
        if 'by_source' in name:
            source = re.search(r'source="([^"]+)"', labels)
            key = f'{name}[{source.group(1) if source else "?"}]'
        values[key] = values.get(key, 0.0) + float(value)
    return values


def corpus():
    parts = []
    for relative in ('docs', 'src'):
        for path in sorted((ROOT / relative).rglob('*')):
            if path.suffix in ('.py', '.md') and path.is_file():
                parts.append(path.read_text(errors='ignore'))
    return '\n\n'.join(parts)


def text_prompt(blob, tokens, chars_per_token, nonce):
    need = int(tokens * chars_per_token)
    start = ((nonce + 1) * 104729 * 7919) % max(1, len(blob) - need)
    return blob[start:start + need]


def token_prompt(tokens, nonce):
    rng = random.Random(tokens * 1_000_003 + nonce)
    return [1000 + rng.randrange(4000) for _ in range(tokens)]


def token_count(base, model, text):
    response = requests.post(base + '/tokenize', json={'model': model, 'prompt': text}, timeout=120)
    response.raise_for_status()
    return len(response.json()['tokens'])


def fit_text(base, model, blob, tokens, nonce):
    """Slice the corpus until its token count matches the target within 0.2%."""
    start = ((nonce + 1) * 104729 * 7919 + tokens * 977) % (len(blob) // 2)
    width = int(tokens * 4.45)
    best = None
    for _ in range(8):
        text = blob[start:start + width]
        count = token_count(base, model, text)
        if best is None or abs(count - tokens) < abs(best[1] - tokens):
            best = (text, count)
        if abs(count - tokens) <= max(1, tokens // 500):
            break
        width = int(width * tokens / max(1, count))
    return best[0]


def run(base, model, prompt, tokens, max_tokens, timeout):
    before = snapshot(base)
    payload = {
        'model': model,
        'prompt': prompt,
        'max_tokens': max_tokens,
        'min_tokens': max_tokens,
        'ignore_eos': True,
        'temperature': 0.0,
        'top_p': 1.0,
        'stream': True,
        'stream_options': {'include_usage': True},
    }
    started = time.monotonic()
    first = last = None
    usage = None
    with requests.post(base + '/v1/completions', json=payload, stream=True,
                       timeout=(30, timeout)) as response:
        if response.status_code != 200:
            raise RuntimeError(f'{tokens}: HTTP {response.status_code} {response.text[:400]}')
        for line in response.iter_lines():
            if line.startswith(b'data: '):
                body = line[6:]
                if body == b'[DONE]':
                    break
                chunk = json.loads(body)
                now = time.monotonic()
                if chunk.get('usage'):
                    usage = chunk['usage']
                for choice in chunk.get('choices', []):
                    if (choice.get('text') or ''):
                        if first is None:
                            first = now
                        last = now
    ended = time.monotonic()
    after = snapshot(base)

    def delta(name):
        return after.get(name, 0.0) - before.get(name, 0.0)

    prefill = delta('request_prefill_time_seconds_sum')
    decode = delta('request_decode_time_seconds_sum')
    prompt_tokens = usage['prompt_tokens'] if usage else tokens
    gen_tokens = usage['completion_tokens'] if usage else 0
    return {
        'target_tokens': tokens,
        'prompt_tokens': prompt_tokens,
        'generation_tokens': gen_tokens,
        'prefill_seconds_engine': round(prefill, 3),
        'decode_seconds_engine': round(decode, 3),
        'prefill_tokens_per_s': round(prompt_tokens / prefill, 1) if prefill else None,
        'decode_tokens_per_s': round(gen_tokens / decode, 1) if decode else None,
        'ttft_seconds_engine': round(delta('time_to_first_token_seconds_sum'), 3),
        'ttft_seconds_client': round(first - started, 3) if first else None,
        'decode_tokens_per_s_stream': (
            round((gen_tokens - 1) / (last - first), 1)
            if first and last and last > first else None),
        'total_seconds_client': round(ended - started, 3),
        'external_kv_tokens': int(delta('prompt_tokens_by_source_total[external_kv_transfer]')),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='http://127.0.0.1:18200')
    parser.add_argument('--model', default='kvmem-2080ti')
    parser.add_argument('--lengths', type=int, nargs='+', default=[8192, 32768, 65536, 204800])
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--mode', choices=('text', 'tokens'), default='text')
    parser.add_argument('--nonce', type=int, default=0,
                        help='selects a different slice of the corpus; 0 may replay an earlier prompt')
    parser.add_argument('--out', default='')
    args = parser.parse_args()

    blob = corpus() if args.mode == 'text' else ''
    print(f'mode={args.mode} corpus={len(blob)} chars', flush=True)

    rows = []
    for tokens in args.lengths:
        for attempt in range(4):
            nonce = args.nonce + attempt * 1000
            prompt = (fit_text(args.base, args.model, blob, tokens, nonce)
                      if args.mode == 'text' else token_prompt(tokens, nonce))
            row = run(args.base, args.model, prompt, tokens, args.max_tokens, args.timeout)
            if not row['external_kv_tokens']:
                break
            print(f'{tokens}: replayed {row["external_kv_tokens"]} prefix tokens, refitting', flush=True)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2) + '\n')
        print('wrote', args.out)


if __name__ == '__main__':
    main()
