#!/usr/bin/env bash
# Octopus launcher for macOS and Linux. First run installs everything (a few GB).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  echo "🐙 First run: setting up Octopus (this takes a few minutes)…"
  $PY -m venv .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU found, installing PyTorch with CUDA"
    pip install torch --index-url https://download.pytorch.org/whl/cu124
  else
    pip install torch
  fi
  pip install -r requirements.txt
else
  . .venv/bin/activate
fi
( sleep 3; (command -v open >/dev/null && open "http://localhost:${OCTOPUS_PORT:-8000}") || (command -v xdg-open >/dev/null && xdg-open "http://localhost:${OCTOPUS_PORT:-8000}") || true ) &
python -m octopus
