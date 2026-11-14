#!/usr/bin/env bash
# Run one chat request with a GPU watchdog: if the engine stops using the GPU while the
# request is still in flight, abort instead of waiting blind.
#   usage: scripts/run_with_watch.sh <payload.json> [out.json] [max_seconds] [idle_samples]
set -u
PAYLOAD=${1:?payload json}
OUT=${2:-/tmp/watch_out.json}
MAXS=${3:-900}
IDLE_SAMPLES=${4:-6}          # 6 samples * 5s = 30s of idle GPU = stall
URL=${WATCH_URL:-http://127.0.0.1:18200/v1/chat/completions}
LOG=$(ls -t /home/srj/src/kvmem-vllm-adapter/engines/vllm/run-logs/vllm-*.log 2>/dev/null | head -1)

rm -f "$OUT"
start=$(date +%s)
curl -s -m "$MAXS" "$URL" -H 'Content-Type: application/json' -d @"$PAYLOAD" -o "$OUT" &
CPID=$!
idle=0; ticks=0
while kill -0 "$CPID" 2>/dev/null; do
  sleep 5; ticks=$((ticks+1))
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
  if [ "${util:-0}" -lt 3 ]; then idle=$((idle+1)); else idle=0; fi
  if [ "$idle" -ge "$IDLE_SAMPLES" ]; then
    kill -9 "$CPID" 2>/dev/null
    echo "STALL: GPU idle ${idle} samples (~$((idle*5))s) after ${ticks} ticks; request aborted" >&2
    if [ -n "$LOG" ]; then tail -3 "$LOG" | cut -c1-150 >&2; fi
    echo "--- adapter events (last):" >&2
    python3 - <<'PYEOF' >&2 2>/dev/null || true
import json, os
p="/tmp/kvmem_sparse.jsonl"
if os.path.exists(p):
    size=os.path.getsize(p)
    with open(p,"rb") as f:
        f.seek(max(0,size-300_000)); t=f.read().decode("utf-8","ignore")
    rows=[json.loads(l) for l in t.splitlines() if l.strip().startswith("{")]
    for r in rows[-4:]:
        print({k:r.get(k) for k in ("kind","req","tokens","n_new","error")})
PYEOF
    exit 3
  fi
  if [ $((ticks % 12)) -eq 0 ]; then echo "  ...${ticks}0s util=${util}%" >&2; fi
done
wait "$CPID" 2>/dev/null; rc=$?
elapsed=$(( $(date +%s) - start ))
if [ "$rc" -ne 0 ]; then echo "curl rc=$rc after ${elapsed}s" >&2; exit "$rc"; fi
echo "done in ${elapsed}s -> $OUT"
