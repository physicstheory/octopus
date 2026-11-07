#!/usr/bin/env bash
# Runs once, when the codespace is created: installs Ollama, Octopus's libraries and the default chat model.
set -e
echo "🐙 Setting up Octopus (only the first time, about 5 minutes)…"
mkdir -p "$OCTOPUS_HOME" "$OLLAMA_MODELS"
if ! command -v ollama >/dev/null; then
  sudo apt-get update -q && sudo apt-get install -y -q zstd libgl1 >/dev/null
  curl -fsSL https://ollama.com/install.sh | sh
fi
pip install -q --user -r requirements.txt
(ollama serve >/tmp/ollama.log 2>&1 &)
for i in $(seq 1 30); do curl -sf http://127.0.0.1:11434/api/version >/dev/null && break; sleep 1; done
ollama pull qwen2.5-coder:1.5b
echo "🐙 Setup finished."
