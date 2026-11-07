#!/usr/bin/env bash
# Runs every time the codespace starts: updates Octopus, then keeps Ollama and Octopus running in the background.
# Everything is inside main so bash reads the whole file before git pull can change it.
main() {
cd "$(dirname "$0")/.."
mkdir -p "$OCTOPUS_HOME/logs"
# Pick up the newest Octopus from GitHub (skipped if you've changed files here).
git pull -q --ff-only 2>/dev/null && pip install -q --user -r requirements.txt >/dev/null 2>&1
# Finish the setup if it didn't complete when the codespace was made.
[ -x /usr/local/lib/ollama/llama-server ] && python -c "import fastapi, trimesh" 2>/dev/null || bash .devcontainer/setup.sh
# Accounts after the first need this invite code (the owner signs up without it). It stays the same.
[ -f "$OCTOPUS_HOME/invite-code" ] || tr -dc 'a-z0-9' </dev/urandom | head -c 8 > "$OCTOPUS_HOME/invite-code"
export OCTOPUS_PASSWORD="${OCTOPUS_PASSWORD:-$(cat "$OCTOPUS_HOME/invite-code")}"
# Listen on IPv4 and IPv6, so the codespace's port forwarding reaches Octopus whichever it uses.
if [ -f /proc/net/if_inet6 ]; then export OCTOPUS_HOST="::"; else export OCTOPUS_HOST="0.0.0.0"; fi

# Stop anything left from an earlier start, then run both under a watchdog that restarts them if they stop.
pkill -f ".devcontainer/watchdog.sh" 2>/dev/null
pkill -f "python -m octopus" 2>/dev/null
nohup setsid bash .devcontainer/watchdog.sh >/dev/null 2>&1 &

for i in $(seq 1 45); do curl -sf http://127.0.0.1:8000/healthz >/dev/null && break; sleep 1; done
if curl -sf http://127.0.0.1:8000/healthz >/dev/null; then
  echo "🐙 Octopus is running. Invite code for other people: $OCTOPUS_PASSWORD"
else
  echo "❌ Octopus didn't start. The last lines of its log are below; send a screenshot of them to Claude."
  tail -n 15 "$OCTOPUS_HOME/logs/octopus.log"
fi
}
main "$@"
