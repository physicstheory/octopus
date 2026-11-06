#!/usr/bin/env bash
# Octopus launcher for macOS and Linux. The first run installs what your computer needs.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
PORT=${OCTOPUS_PORT:-8000}
if [ ! -d .venv ]; then
  echo "🐙 First run: setting up Octopus…"
  $PY -m venv .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU found: installing PyTorch with CUDA and the training libraries (a few GB)"
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install -r requirements-gpu.txt
  else
    echo "No NVIDIA GPU: installing the light version. Chat runs through Ollama; training and 3D use the free cloud notebook."
    pip install -r requirements.txt
  fi
else
  . .venv/bin/activate
fi
if ! command -v nvidia-smi >/dev/null 2>&1 && ! command -v ollama >/dev/null 2>&1; then
  echo
  echo "⚠️  Octopus chats through Ollama, which isn't installed yet."
  echo "    Download it from https://ollama.com/download, open it once, then run ./start.sh again."
  (command -v open >/dev/null && open "https://ollama.com/download") || true
fi
( sleep 3; (command -v open >/dev/null && open "http://localhost:$PORT") || (command -v xdg-open >/dev/null && xdg-open "http://localhost:$PORT") || true ) &
python -m octopus
