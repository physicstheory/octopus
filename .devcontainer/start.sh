#!/usr/bin/env bash
# Runs every time the codespace starts: starts Ollama and Octopus in the background.
mkdir -p "$OCTOPUS_HOME"
# Accounts after the first need this invite code (the owner signs up without it). It stays the same.
[ -f "$OCTOPUS_HOME/invite-code" ] || tr -dc 'a-z0-9' </dev/urandom | head -c 8 > "$OCTOPUS_HOME/invite-code"
export OCTOPUS_PASSWORD="${OCTOPUS_PASSWORD:-$(cat "$OCTOPUS_HOME/invite-code")}"
pgrep -x ollama >/dev/null || nohup setsid ollama serve >/tmp/ollama.log 2>&1 &
for i in $(seq 1 30); do curl -sf http://127.0.0.1:11434/api/version >/dev/null && break; sleep 1; done
pkill -f "python -m octopus" 2>/dev/null
nohup setsid python -m octopus >/tmp/octopus.log 2>&1 &
for i in $(seq 1 30); do curl -sf http://127.0.0.1:8000/healthz >/dev/null && break; sleep 1; done
echo "🐙 Octopus is running. Invite code for other people: $OCTOPUS_PASSWORD"
