#!/usr/bin/env bash
# Keeps Ollama and Octopus running inside the codespace, restarting either one if it stops.
LOG="$OCTOPUS_HOME/logs"
mkdir -p "$LOG"
while true; do
  pgrep -x ollama >/dev/null || (nohup ollama serve >>"$LOG/ollama.log" 2>&1 &)
  for i in $(seq 1 30); do curl -sf http://127.0.0.1:11434/api/version >/dev/null && break; sleep 1; done
  echo "--- Octopus starting $(date)" >>"$LOG/octopus.log"
  python -m octopus >>"$LOG/octopus.log" 2>&1
  echo "--- Octopus stopped (exit $?) $(date), restarting" >>"$LOG/octopus.log"
  sleep 2
done
