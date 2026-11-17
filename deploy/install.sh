#!/bin/bash
# Check the host, apply the engine patch and install the plugin.
#   ./deploy/install.sh [--print-unit]
# Environment:
#   KVMEM_ENGINE_DIR  vLLM 0.2.2.post3 SM75 checkout (required, has .venv)
#   KVMEM_MODEL_DIR   directory holding the served checkpoint (required by profiles)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENGINE=${KVMEM_ENGINE_DIR:?set KVMEM_ENGINE_DIR to the engine checkout (contains .venv)}
VENV=$ENGINE/.venv/bin
MODEL=${KVMEM_MODEL_DIR:?set KVMEM_MODEL_DIR to the checkpoint directory}

if [ "${1:-}" = "--print-unit" ]; then
  LAUNCHER=${KVMEM_LAUNCHER:-$REPO/deploy/run-kvmem-vllm.sh}
  sed -e "s|@LAUNCHER@|$LAUNCHER|g" -e "s|@USER@|${SUDO_USER:-$USER}|g" \
      -e "s|@HOME@|$HOME|g" "$REPO/deploy/kvmem-vllm.service"
  exit 0
fi

command -v python3 >/dev/null || { echo "python3 is required"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found (NVIDIA driver)"; exit 1; }
[ -x "$VENV/python" ] || { echo "no virtualenv at $VENV (create it in the engine checkout)"; exit 1; }
[ -f "$MODEL/config.json" ] || { echo "no config.json under $MODEL"; exit 1; }

echo "[1/3] engine patch"
if git -C "$ENGINE" apply --check "$REPO/patches/0001-engine-skip-kv-check.patch" 2>/dev/null; then
  git -C "$ENGINE" apply "$REPO/patches/0001-engine-skip-kv-check.patch"
  echo "      applied 0001-engine-skip-kv-check.patch"
else
  echo "      already applied (or the tree differs) — check with: git -C $ENGINE diff --stat"
fi

echo "[2/3] plugin install (editable)"
if command -v uv >/dev/null; then
  uv pip install --python "$VENV/python" -e "$REPO"
else
  "$VENV/python" -m pip install -e "$REPO"
fi

echo "[3/3] smoke"
KVMEM_ENGINE_DIR="$ENGINE" KVMEM_MODEL_DIR="$MODEL" \
  CONFIG="$REPO/local/serve-2080ti-default.json" "$REPO/deploy/run-kvmem-vllm.sh" status || true
cat <<MSG

Done. Export these in your shell before running the launcher:
  export KVMEM_ENGINE_DIR=$ENGINE
  export KVMEM_MODEL_DIR=$MODEL

Start serving:            ./deploy/run-kvmem-vllm.sh start
Status:                   ./deploy/run-kvmem-vllm.sh status
Send a request:           ./scripts/serve_watch.py req --file request.json
Install as a service:     ./deploy/install.sh --print-unit > /etc/systemd/system/kvmem-vllm.service
MSG
