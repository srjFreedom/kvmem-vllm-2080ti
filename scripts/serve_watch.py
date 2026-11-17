#!/usr/bin/env python3
"""Watchdog for the KVMem 2080Ti service: start/stop/status/req with live GPU sampling.

No fixed sleeps in the hot paths: readiness and completion are polled on observable signals
(/health, /v1/models, the process table, nvidia-smi) and a GPU-idle window while a request is
in flight is reported as a stall instead of being waited out blindly.

  serve_watch.py start [--config PATH] [--run-dir DIR] [--set KEY=VALUE ...] [--timeout S]
  serve_watch.py stop
  serve_watch.py status [--run-dir DIR]
  serve_watch.py logs [--run-dir DIR] [--lines N]
  serve_watch.py params [--run-dir DIR] [--config PATH]
  serve_watch.py req --file REQ.json [--idle-stall S] [--repeat N]

The engine log lives in the run directory (serve.log). `logs` and `status` print the path;
`params` prints the effective serving knobs (window, history, sessions, MTP depth).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _engine_venv():
    """Locate the engine virtualenv that provides ``vllm-kvmem``.

    Order: $KVMEM_ENGINE_VENV, $KVMEM_ENGINE_DIR/.venv/bin, then PATH lookup.
    Nothing is hard-coded to one machine.
    """
    explicit = os.environ.get("KVMEM_ENGINE_VENV")
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    engine = os.environ.get("KVMEM_ENGINE_DIR")
    if engine:
        return os.path.join(os.path.abspath(os.path.expanduser(engine)), ".venv", "bin")
    found = shutil.which("vllm-kvmem")
    if found:
        return os.path.dirname(os.path.abspath(found))
    raise SystemExit("set KVMEM_ENGINE_VENV (or KVMEM_ENGINE_DIR) to the engine virtualenv")


_VENV = None


def venv():
    """Resolve the engine virtualenv lazily; only launching needs it."""
    global _VENV
    if _VENV is None:
        _VENV = _engine_venv()
    return _VENV
BASE = os.environ.get("KVMEM_BASE_URL", "http://127.0.0.1:18200")
DEFAULT_CONFIG = os.path.join(ROOT, "local", "serve-2080ti-smoke.json")
DEFAULT_RUN = os.environ.get("RUN_DIR", "/tmp/kvmem-run-watch")
MISS_LOG = os.environ.get("KVMEM_SESSION_MISS_LOG", "/tmp/kvmem_session_miss.jsonl")

FATAL = re.compile(r"(ValueError: |RuntimeError: |EngineDeadError|engine core initialization failed"
                   r"|CUDA out of memory|Traceback \(most recent call last\))")

def http(path, timeout=5.0, data=None):
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status, response.read()

def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=5).stdout.strip().splitlines()
        rows = []
        for line in out:
            idx, util, mem = (part.strip() for part in line.split(","))
            rows.append((int(idx), int(util), int(mem)))
        return rows
    except Exception:
        return []

def serve_pids():
    out = subprocess.run(["pgrep", "-f", "vllm-kvmem serve"], capture_output=True, text=True).stdout
    return [int(p) for p in out.split()]

def ready():
    try:
        status, body = http("/v1/models", timeout=3)
        if status != 200:
            return None
        model = json.loads(body)["data"][0]
        return model
    except Exception:
        return None

def cmd_start(args):
    if serve_pids():
        print("already running: pids=%s" % serve_pids())
        return 0
    # The launcher refuses an existing run-dir. Only ever remove one that is ours (it holds
    # capture/launch.json or is empty); otherwise ask for --fresh.
    if os.path.isdir(args.run_dir):
        entries = set(os.listdir(args.run_dir))
        ours = not entries or entries <= {"capture", "launch.json", "serve.log"}
        if not ours and not args.fresh:
            print("run-dir %s has unrelated content; pass --fresh to replace it" % args.run_dir)
            return 1
        import shutil
        shutil.rmtree(args.run_dir)
    elif os.path.exists(args.run_dir):
        print("run-dir %s exists and is not a directory" % args.run_dir)
        return 1
    log_path = args.log or os.path.join(args.run_dir, "serve.log")
    if os.path.dirname(os.path.abspath(log_path)) == os.path.abspath(args.run_dir):
        os.makedirs(args.run_dir, exist_ok=True)   # the launcher owns serve.log only
    log = open(log_path, "w")
    started = time.time()
    command = [os.path.join(venv(), "vllm-kvmem"), "serve", "--config", args.config,
               "--run-dir", args.run_dir]
    for assignment in args.passthrough:
        command += ["--set", assignment]
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print("launch pid=%d log=%s" % (proc.pid, log_path))
    marker = "Application startup complete"
    last_note = 0.0
    while True:
        if proc.poll() is not None:
            print("FAILED: process exited with %s after %.1fs" % (proc.returncode, time.time() - started))
            _tail(log_path, FATAL)
            return 1
        model = ready()
        if model is not None:
            print("READY in %.1fs  model=%s ctx=%s" % (time.time() - started, model.get("id"),
                                                       model.get("max_model_len")))
            _pool_line(log_path)
            for idx, util, mem in gpu():
                print("  gpu%d util=%d%% mem=%d MiB" % (idx, util, mem))
            return 0
        text = _tail(log_path, FATAL, quiet=True)
        if text:
            print("FAILED at %.1fs: %s" % (time.time() - started, text[-300:]))
            return 1
        if os.path.exists(log_path):
            with open(log_path, "r", errors="replace") as fh:
                if marker in fh.read() and time.time() - last_note > 5:
                    last_note = time.time()
        if time.time() - started > args.timeout:
            print("TIMEOUT after %.1fs (no /health 200)" % (time.time() - started))
            _tail(log_path, FATAL)
            return 1
        time.sleep(0.5)

def _tail(path, pattern, quiet=False):
    try:
        with open(path, "r", errors="replace") as fh:
            lines = [l for l in fh.read().splitlines() if pattern.search(l)]
        if not lines:
            return ""
        if quiet:
            return lines[-1]
        for line in lines[-6:]:
            print("  " + line[:220])
        return lines[-1]
    except FileNotFoundError:
        return ""

def _pool_line(path):
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                if "GPU KV cache size" in line or "skipping the KV capacity" in line:
                    print("  " + line.split("] ")[-1].strip()[:200])
    except FileNotFoundError:
        pass

def _log_path(run_dir, explicit=None):
    return explicit or os.path.join(run_dir, "serve.log")


def cmd_logs(args):
    run_dir = args.run_dir
    log_path = _log_path(run_dir, args.log)
    capture = os.path.join(run_dir, "capture")
    print("run-dir:", run_dir)
    print("serve log:", log_path, "(exists)" if os.path.exists(log_path) else "(missing)")
    if os.path.isdir(capture):
        for name in sorted(os.listdir(capture)):
            path = os.path.join(capture, name)
            if os.path.isfile(path):
                print("  capture/%s %d bytes" % (name, os.path.getsize(path)))
    print("session misses:", MISS_LOG, "(exists)" if os.path.exists(MISS_LOG) else "(missing)")
    if os.path.exists(log_path):
        with open(log_path, "r", errors="replace") as fh:
            lines = fh.read().splitlines()
        print("--- last %d lines of serve.log ---" % args.lines)
        for line in lines[-args.lines:]:
            print(line[:240])
    return 0


PARAM_GROUPS = (
    ("model", ("model", "served_model_name", "tensor_parallel_size", "dtype",
               "kv_cache_dtype", "max_model_len", "execution_mode")),
    ("batch", ("max_num_seqs", "max_num_batched_tokens", "prefill_chunk_tokens",
               "gpu_memory_utilization", "cudagraph_capture_sizes")),
    ("mtp", ("draft_method", "draft_tokens", "sm75_spec_sync_mode",
             "allow_mamba_spec_full_cudagraph")),
    ("window", ("kvmem.select_budget", "kvmem.gen_budget", "kvmem.block_tokens",
                "kvmem.sink_blocks", "kvmem.recent_blocks", "query_replay_budget")),
    ("history", ("kvmem.cpu_read_cache_bytes", "kvmem.repository_blocks",
                 "kvmem.archive_backend")),
    ("sessions", ("kvmem.session_capacity", "kvmem.session_checkpoint_bytes",
                  "kvmem.session_checkpoint_policy", "kvmem.session_capture_terminal",
                  "auto_session")),
)


def _dig(profile, dotted):
    node = profile
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return "-"
        node = node[part]
    return node


def cmd_params(args):
    profile = None
    launch = os.path.join(args.run_dir, "launch.json")
    if os.path.exists(launch):
        profile = json.load(open(launch)).get("profile")
        print("source:", launch)
    if profile is None and os.path.exists(args.config):
        profile = json.load(open(args.config))
        print("source:", args.config)
    if profile is None:
        print("no launch.json in the run dir and no readable config; is the service started?")
        return 1
    for title, keys in PARAM_GROUPS:
        print("%s:" % title)
        for key in keys:
            print("  %-32s %s" % (key, _dig(profile, key)))
    argv = json.load(open(launch)).get("vllm_argv") if os.path.exists(launch) else None
    if argv and "--kv-cache-dtype" in argv:
        print("  %-32s %s" % ("kv cache dtype (argv)", argv[argv.index("--kv-cache-dtype") + 1]))
    return 0


def cmd_stop(_args):
    pids = serve_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    extra = subprocess.run(["pgrep", "-f", "VLLM::"], capture_output=True, text=True).stdout.split()
    for pid in extra:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except Exception:
            pass
    deadline = time.time() + 30
    while time.time() < deadline:
        if not serve_pids():
            break
        time.sleep(0.3)
    print("stopped (was %s)" % (pids or "none"))
    for idx, util, mem in gpu():
        print("  gpu%d util=%d%% mem=%d MiB" % (idx, util, mem))
    return 0

def cmd_status(args):
    model = ready()
    print("ready:", model.get("id") if model else False)
    if model:
        print("ctx:", model.get("max_model_len"))
    for idx, util, mem in gpu():
        print("gpu%d util=%d%% mem=%d MiB" % (idx, util, mem))
    print("serve pids:", serve_pids())
    print("run-dir:", args.run_dir)
    print("serve log:", _log_path(args.run_dir, args.log))
    for name in ("scheduler_resources.json", "archive.json"):
        path = os.path.join(args.run_dir, "capture", name)
        if os.path.exists(path):
            data = json.load(open(path))
            if "sessions" in data:
                print("sessions:", data["sessions"])
            if "session_restored_tokens" in data:
                print("restored_tokens:", data["session_restored_tokens"])
    return 0

def cmd_req(args):
    body = open(args.file, "rb").read()
    result = {}
    def worker():
        t0 = time.time()
        try:
            status, raw = http("/v1/chat/completions", timeout=args.timeout, data=body)
            result["elapsed"] = time.time() - t0
            result["status"] = status
            result["body"] = json.loads(raw)
        except urllib.error.HTTPError as error:
            result["elapsed"] = time.time() - t0
            result["status"] = error.code
            result["body"] = {"error": error.read().decode()[:300]}
        except Exception as error:                     # noqa: BLE001 - report, never hang
            result["elapsed"] = time.time() - t0
            result["status"] = "error"
            result["body"] = {"error": str(error)[:300]}
    import threading
    thread = threading.Thread(target=worker, daemon=True)
    t0 = time.time()
    thread.start()
    idle_since = None
    stall_reported = False
    peak_mem = 0
    samples = 0
    busy = 0
    while thread.is_alive():
        rows = gpu()
        samples += 1
        hottest = max((r[1] for r in rows), default=0)
        peak_mem = max(peak_mem, max((r[2] for r in rows), default=0))
        if hottest > 5:
            busy += 1
            idle_since = None
        else:
            idle_since = idle_since or time.time()
            if (not stall_reported and args.idle_stall > 0
                    and time.time() - idle_since > args.idle_stall):
                print("STALL: GPU idle %.0fs while the request is in flight (t=%.1fs)"
                      % (time.time() - idle_since, time.time() - t0))
                stall_reported = True
        time.sleep(0.4)
    thread.join(timeout=5)
    elapsed = result.get("elapsed", time.time() - t0)
    usage = (result.get("body", {}).get("usage") or {})
    prompt = usage.get("prompt_tokens") or 0
    completion = usage.get("completion_tokens") or 0
    print("request status=%s wall=%.2fs prompt=%s completion=%s"
          % (result.get("status"), elapsed, prompt, completion))
    if prompt and elapsed:
        print("  prefill+decode throughput ~%.0f tok/s (prompt/s)" % (prompt / elapsed))
    if completion and elapsed:
        print("  decode throughput ~%.1f tok/s (completion/s over the whole wall)"
              % (completion / elapsed))
    print("  gpu busy on %d/%d samples, peak mem %d MiB, stalled=%s"
          % (busy, samples, peak_mem, stall_reported))
    choices = result.get("body", {}).get("choices")
    if choices:
        answer = choices[0]["message"].get("content") or ""
        print("  answer:", repr(answer.strip()[:60]))
    else:
        print("  error:", str(result.get("body"))[:300])
    return 0

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    start = sub.add_parser("start")
    start.add_argument("--config", default=DEFAULT_CONFIG)
    start.add_argument("--run-dir", default=DEFAULT_RUN)
    start.add_argument("--log", default=None, help="engine log (default: RUN_DIR/serve.log)")
    start.add_argument("--timeout", type=float, default=420)
    start.add_argument("--fresh", action="store_true",
                       help="replace an existing run-dir that holds unrelated content")
    start.add_argument("--set", dest="passthrough", action="append", default=[], metavar="KEY=VALUE",
                       help="profile override forwarded to vllm-kvmem serve (repeatable)")
    sub.add_parser("stop")
    status = sub.add_parser("status")
    status.add_argument("--run-dir", default=DEFAULT_RUN)
    status.add_argument("--log", default=None)
    logs = sub.add_parser("logs")
    logs.add_argument("--run-dir", default=DEFAULT_RUN)
    logs.add_argument("--log", default=None)
    logs.add_argument("--lines", type=int, default=40)
    params = sub.add_parser("params")
    params.add_argument("--run-dir", default=DEFAULT_RUN)
    params.add_argument("--config", default=DEFAULT_CONFIG)
    req = sub.add_parser("req")
    req.add_argument("--file", required=True)
    req.add_argument("--timeout", type=float, default=3600)
    req.add_argument("--idle-stall", type=float, default=25)
    args = parser.parse_args()
    handlers = {"start": cmd_start, "stop": cmd_stop, "status": cmd_status,
                "logs": cmd_logs, "params": cmd_params, "req": cmd_req}
    return handlers[args.cmd](args)

if __name__ == "__main__":
    sys.exit(main())
