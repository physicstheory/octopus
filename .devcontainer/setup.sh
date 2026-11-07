#!/usr/bin/env bash
# Runs when the codespace is created (and again from start.sh if anything is missing):
# installs Ollama, Octopus's libraries and the default chat model.
set -e
cd "$(dirname "$0")/.."
echo "🐙 Setting up Octopus (only the first time, about 5 minutes)…"
mkdir -p "$OCTOPUS_HOME" "$OLLAMA_MODELS"
if ! command -v zstd >/dev/null; then
  sudo apt-get update -q >/dev/null && sudo apt-get install -y -q zstd libgl1 >/dev/null
fi
# Ollama's installer sometimes stops halfway on a slow connection; check the chat engine is really there.
for try in 1 2 3; do
  { command -v ollama >/dev/null && [ -x /usr/local/lib/ollama/llama-server ]; } && break
  echo "🐙 Installing Ollama (try $try)…"
  curl -fsSL https://ollama.com/install.sh | sh || sleep 5
done
[ -x /usr/local/lib/ollama/llama-server ] || { echo "❌ Ollama didn't install. Run: bash .devcontainer/setup.sh"; exit 1; }
pip install -q --user -r requirements.txt
pgrep -x ollama >/dev/null || (nohup ollama serve >/tmp/ollama-setup.log 2>&1 &)
for i in $(seq 1 30); do curl -sf http://127.0.0.1:11434/api/version >/dev/null && break; sleep 1; done
ollama pull qwen2.5-coder:1.5b
echo "🐙 Setup finished."
