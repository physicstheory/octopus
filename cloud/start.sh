#!/usr/bin/env bash
# Starts Ollama and Octopus inside the cloud container.
if [ -z "$OCTOPUS_PASSWORD" ]; then
  echo "Add a secret called OCTOPUS_PASSWORD in the Space's Settings (it's the invite code for new accounts), then restart the Space."
  exit 1
fi
[ -z "$HF_TOKEN" ] && echo "No HF_TOKEN secret: Octopus will work, but accounts and chats are lost when the Space restarts."
ollama serve >/tmp/ollama.log 2>&1 &
for i in $(seq 1 30); do curl -sf http://127.0.0.1:11434/api/version >/dev/null && break; sleep 1; done
exec python -m octopus
