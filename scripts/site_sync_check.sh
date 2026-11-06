#!/usr/bin/env bash
# three-tree liveness check: the SERVICE imports vllm from the site tree;
# fork changes under engines/vllm only matter after a file-level sync.
# CONTRACTS lists required tokens (empty = none). DESYNC fails.
set -u
SITE=/home/srj/vLLM-2080Ti-Definitive
SRC=/home/srj/src/kvmem-vllm-adapter/engines/vllm
CONTRACTS=""
fail=0
for f in vllm/v1/worker/gpu/model_runner.py; do
  if ! diff -q "$SRC/$f" "$SITE/$f" >/dev/null 2>&1; then
    echo "DESYNC: $f"; fail=1
  fi
  for tok in $CONTRACTS; do
    if ! grep -q "$tok" "$SITE/$f"; then
      echo "CONTRACT MISSING: $tok not in site $f"; fail=1
    fi
  done
done
if [ $fail -eq 0 ]; then
  echo "site sync check: OK ($(md5sum "$SRC/vllm/v1/worker/gpu/model_runner.py" | cut -c1-8) == site)"
fi
exit $fail
