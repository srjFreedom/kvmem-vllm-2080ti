#!/usr/bin/env bash
# Atomic arm switch: dedupe keys BEFORE appending, assert, restart.
# usage: sudo scripts/arm_boot.sh "K=V K=V ..."
set -eu
FILE=/etc/default/kvmem-vllm
# Bench baseline: every arm boot starts from the clean file so an experimental key
# (KVMEM_SLIDING_WINDOW / MASK_WIN / ...) can never leak into the next measurement.
# Measured cost of leaking: three rounds of "request hangs" that were actually a stale
# sliding-window arm. Maintain the baseline here; `KVMEM_ENV_BASELINE=/path` overrides.
BASELINE=${KVMEM_ENV_BASELINE:-/tmp/kvmem_env_baseline}
if [[ -f "$BASELINE" ]]; then cp "$BASELINE" "$FILE"; fi
declare -A FINAL; order=()
while IFS= read -r l; do
  [[ "$l" == *"="* && "$l" != \#* ]] || continue
  k=${l%%=*}; v=${l#*=}
  [[ " ${order[*]-} " == *" $k "* ]] || order+=("$k")
  FINAL[$k]=$v
done < "$FILE"
for pair in $1; do
  k=${pair%%=*}; v=${pair#*=}
  [[ " ${order[*]-} " == *" $k "* ]] || order+=("$k")
  FINAL[$k]=$v
done
{ for k in "${order[@]}"; do echo "${k}=${FINAL[$k]}"; done; } > "$FILE"
for pair in $1; do
  grep -qx "$pair" "$FILE" || { echo "ARM-ASSERT-FAIL: $pair"; exit 1; }
done
systemctl restart kvmem-vllm
echo "arm-boot issued: $1"
