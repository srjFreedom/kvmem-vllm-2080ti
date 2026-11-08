#!/usr/bin/env bash
# Atomic arm switch: dedupe keys BEFORE appending, assert, restart.
# usage: sudo scripts/arm_boot.sh "K=V K=V ..."
set -eu
FILE=/etc/default/kvmem-vllm
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
