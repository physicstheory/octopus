#!/usr/bin/env bash
# Octopus launcher for macOS and Linux. The first run installs everything it needs
# (Python if missing, Octopus's libraries, and Ollama for chat), then opens Octopus.
set -e
cd "$(dirname "$0")"
PORT=${OCTOPUS_PORT:-8000}
OS=$(uname -s)
echo
echo "  🐙 Octopus is starting. The first run installs everything it needs (5 to 15 minutes)."
echo

ok_python() { "$1" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' >/dev/null 2>&1; }
find_python() {
  PY=""
  for c in ${PYTHON:-} python3.12 python3.11 python3.10 python3.13 python3 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12 \
           /Library/Frameworks/Python.framework/Versions/3.12/bin/python3; do
    if command -v "$c" >/dev/null 2>&1 && ok_python "$c"; then PY=$(command -v "$c"); return 0; fi
  done
  return 1
}

# ---------- 1. Python ----------
if ! find_python; then
  echo "Python 3.10 to 3.13 isn't installed. Installing Python 3.12…"
  if [ "$OS" = "Darwin" ]; then
    if command -v brew >/dev/null 2>&1; then
      brew install python@3.12
    else
      curl -fL -o /tmp/octopus-python.pkg https://www.python.org/ftp/python/3.12.8/python-3.12.8-macos11.pkg
      echo "Your Mac will ask for your password to install Python."
      sudo installer -pkg /tmp/octopus-python.pkg -target /
    fi
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y python3 python3-pip
  elif command -v pacman >/dev/null 2>&1; then
    sudo pacman -S --noconfirm python python-pip
  fi
  find_python || { echo "Couldn't install Python automatically. Get Python 3.12 from https://www.python.org/downloads/ and run this again."; exit 1; }
fi

# ---------- 2. Octopus's own Python environment ----------
NVIDIA=0
command -v nvidia-smi >/dev/null 2>&1 && NVIDIA=1
if [ ! -x .venv/bin/python ]; then
  echo "Setting up Octopus's private Python environment…"
  if ! "$PY" -m venv .venv; then
    command -v apt-get >/dev/null 2>&1 && sudo apt-get install -y python3-venv && "$PY" -m venv .venv
  fi
fi
VPY="$PWD/.venv/bin/python"
REQS=$(cat requirements.txt requirements-gpu.txt)
if [ "$REQS" != "$(cat .venv/octopus-reqs.txt 2>/dev/null)" ]; then
  echo "Installing Octopus's libraries…"
  "$VPY" -m pip install --upgrade pip --quiet --disable-pip-version-check
  if [ "$NVIDIA" = 1 ]; then
    echo "NVIDIA GPU found: installing PyTorch with CUDA and the training libraries (a few GB)"
    "$VPY" -m pip install torch --index-url https://download.pytorch.org/whl/cu124
    "$VPY" -m pip install -r requirements-gpu.txt
  else
    "$VPY" -m pip install -r requirements.txt
  fi
  printf '%s' "$REQS" > .venv/octopus-reqs.txt
fi

# ---------- 3. Ollama runs chat on computers without an NVIDIA GPU ----------
if [ "$NVIDIA" = 0 ]; then
  if ! command -v ollama >/dev/null 2>&1 && [ ! -d /Applications/Ollama.app ]; then
    echo "Installing Ollama, which runs Octopus's chat model on your computer…"
    if [ "$OS" = "Darwin" ]; then
      { curl -fL -o /tmp/Ollama-darwin.zip https://ollama.com/download/Ollama-darwin.zip && ditto -x -k /tmp/Ollama-darwin.zip /Applications; } \
        || echo "Couldn't install Ollama. Chat needs it: get it from https://ollama.com/download"
    else
      curl -fsSL https://ollama.com/install.sh | sh || echo "Couldn't install Ollama. Chat needs it: get it from https://ollama.com/download"
    fi
  fi
  if ! curl -s "http://${OLLAMA_HOST:-127.0.0.1:11434}/api/version" >/dev/null 2>&1; then
    if [ "$OS" = "Darwin" ] && [ -d /Applications/Ollama.app ]; then
      open -g -a Ollama
    elif command -v ollama >/dev/null 2>&1; then
      (ollama serve >/dev/null 2>&1 &)
    fi
  fi
fi

echo
echo "  Opening Octopus in your browser. Keep this window open while you use it."
echo
( sleep 5; (command -v open >/dev/null && open "http://localhost:$PORT") || (command -v xdg-open >/dev/null && xdg-open "http://localhost:$PORT") || true ) &
exec "$VPY" -m octopus
