#!/usr/bin/env bash
# ==============================================================================
# KINETIC-X // Linux Server Auto-Deploy Script
# Automatically detects NVIDIA GPU (CUDA) vs CPU-only server environment,
# installs matching PyTorch runtime, and launches on 0.0.0.0:${PORT:-8000}.
# ==============================================================================
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

PORT="${KINETIC_PORT:-8000}"
HOST="${KINETIC_HOST:-0.0.0.0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "============================================================"
echo "  KINETIC-X // Server Auto-Provisioning & Launcher"
echo "============================================================"

# 1. Auto-detect NVIDIA GPU
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  COMPUTE_MODE="cuda"
  TORCH_INDEX="https://download.pytorch.org/whl/cu126"
  echo "[DETECT] NVIDIA GPU found -> Selecting PyTorch CUDA 12.6"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
  COMPUTE_MODE="cpu"
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
  echo "[DETECT] No NVIDIA GPU found -> Selecting lightweight PyTorch CPU"
fi

# 2. Create virtual environment if missing
if [ ! -d ".venv" ]; then
  echo "[SETUP] Creating Python virtual environment (.venv)..."
  "$PYTHON_BIN" -m venv .venv
fi

PIP=".venv/bin/pip"
PY=".venv/bin/python"

# 3. Install PyTorch matching detected hardware
if ! "$PY" -c "import torch" >/dev/null 2>&1; then
  echo "[INSTALL] Installing PyTorch ($COMPUTE_MODE)..."
  "$PIP" install --upgrade pip
  "$PIP" install torch torchvision --index-url "$TORCH_INDEX"
fi

# 4. Install application dependencies
echo "[INSTALL] Installing server & vision dependencies..."
"$PIP" install -r requirements.txt

# 5. Launch server (auto-detects CUDA/CPU at runtime as well)
echo "[START] Launching KINETIC-X on ${HOST}:${PORT} (mode=auto)"
exec "$PY" server.py --host "$HOST" --port "$PORT" --device auto
