#!/usr/bin/env bash
# three-tree liveness check: the SERVICE imports vllm from the site tree;
# fork changes in engines/vllm only matter after a file-level sync.
set -u
SITE=/home/srj/vLLM-2080Ti-Definitive
SRC=/home/srj/src/kvmem-vllm-adapter/engines/vllm
fail=0
for f in vllm/v1/worker/gpu/model_runner.py; do
  if ! diff -q "$SRC/$f" "$SITE/$f" >/dev/null 2>&1; then
    echo "DESYNC: $f"; fail=1
  fi
done
python3 - << 'PYEOF' || fail=1
import sys
sys.path.insert(0, "/home/srj/vLLM-2080Ti-Definitive")
try:
    import inspect
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    src = inspect.getsource(GPUModelRunner.prepare_attn)
    assert "kvmem_view_rows" in src, "fork override NOT live in site"
    print("fork override LIVE")
except AssertionError as e:
    sys.exit(f"FAIL: {e}")
except Exception as e:
    sys.exit(f"FAIL import: {e}")
PYEOF
[ $fail -eq 0 ] && echo "site sync check: OK"
