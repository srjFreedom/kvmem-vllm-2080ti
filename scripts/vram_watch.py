#!/usr/bin/env python3
"""Watch GPU memory: live table, or one-shot, with a running peak.

`nvidia-smi` alone samples on demand, so a sub-second transient (the one that
decides whether a long prefill OOMs) is easy to miss. This samples fast, keeps
the worst value it saw, and can add the engine's own counters.

    python3 scripts/vram_watch.py                # 1 Hz live table until Ctrl-C
    python3 scripts/vram_watch.py --hz 10        # 10 Hz
    python3 scripts/vram_watch.py --json         # one sample, JSON
    python3 scripts/vram_watch.py --hz 10 --seconds 300 > prefill.csv

Exit code is 0; the summary goes to stderr so `> file.csv` keeps clean CSV.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request

SESSION_COLUMNS = ("running", "waiting", "kv_perc")


def sample():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.strip().splitlines():
        index, total, used, util = (part.strip() for part in line.split(","))
        rows.append(dict(index=int(index), total=int(total), used=int(used),
                         free=int(total) - int(used), util=int(util)))
    return rows


def engine(base_url, timeout=2.0):
    """Engine counters, or None when the endpoint is not serving."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/metrics", timeout=timeout) as fh:
            text = fh.read().decode("utf-8", "replace")
    except Exception:
        return None
    found = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for name, key in (("num_requests_running", "running"),
                          ("num_requests_waiting", "waiting"),
                          ("kv_cache_usage_perc", "kv_perc")):
            if line.startswith(f"vllm:{name}{{"):
                try:
                    found[key] = float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return found or None


def free_mib(base_url):
    """One-shot: the numbers you actually compare against a pool size."""
    rows = sample()
    data = engine(base_url) or {}
    return dict(gpus=rows, min_free_mib=min(r["free"] for r in rows),
                peak_used_mib=max(r["used"] for r in rows), **data)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hz", type=float, default=1.0, help="samples per second (default 1)")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = until Ctrl-C)")
    ap.add_argument("--base-url", default="http://127.0.0.1:18200", help="engine endpoint for the counters")
    ap.add_argument("--json", action="store_true", help="print one JSON sample and exit")
    args = ap.parse_args()

    if args.json:
        print(json.dumps(free_mib(args.base_url), indent=2))
        return 0

    period = 1.0 / max(args.hz, 0.05)
    started = time.time()
    worst = None
    header = ["t", "gpu", "used_mib", "free_mib", "util%", *SESSION_COLUMNS]
    print(",".join(header), flush=True)
    try:
        while True:
            rows = sample()
            data = engine(args.base_url) or {}
            now = time.time() - started
            for row in rows:
                print(",".join(str(x) for x in (
                    f"{now:.2f}", row["index"], row["used"], row["free"], row["util"],
                    data.get("running", ""), data.get("waiting", ""),
                    f"{data.get('kv_perc', ''):.3f}" if "kv_perc" in data else "")), flush=True)
            low = min(row["free"] for row in rows)
            high = max(row["used"] for row in rows)
            worst = (low, high) if worst is None else (min(worst[0], low), max(worst[1], high))
            if args.seconds and now >= args.seconds:
                break
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    if worst:
        seen = time.time() - started
        print(f"\n# {seen:.1f}s, {seen and int(seen / period) or 0} samples: "
              f"min_free={worst[0]} MiB, peak_used={worst[1]} MiB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
