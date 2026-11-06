#!/usr/bin/env bash
# Sets up Octopus on a fresh Ubuntu server (made for Oracle Cloud's free Ampere server, works on any Ubuntu).
# Run as root with OCTOPUS_PASSWORD set to the invite code people need to create an account.
# It installs Ollama and Octopus, downloads the everyday chat models, and serves Octopus over HTTPS
# at https://<server-ip>.sslip.io with a free certificate. Everything starts again by itself after a reboot.
# Safe to run again: it updates Octopus to the newest version.
set -euo pipefail

OCTOPUS_REPO="${OCTOPUS_REPO:-https://github.com/physicstheory/octopus}"
OCTOPUS_REF="${OCTOPUS_REF:-main}"
CHAT_MODELS="${CHAT_MODELS:-qwen2.5-coder:1.5b qwen3:4b-instruct-2507-q4_K_M}"
APP=/opt/octopus
log() { echo "[octopus] $*"; }

if [ -z "${OCTOPUS_PASSWORD:-}" ]; then
  if [ -f /etc/octopus.env ]; then . /etc/octopus.env; fi
fi
[ -n "${OCTOPUS_PASSWORD:-}" ] || { echo "Set OCTOPUS_PASSWORD (the invite code) first."; exit 1; }

export DEBIAN_FRONTEND=noninteractive
log "Installing system packages…"
apt-get update -q
apt-get install -y -q git curl ca-certificates python3 python3-venv python3-pip python3-dev build-essential cmake zstd libgl1 libglib2.0-0 iptables-persistent

log "Opening the web ports (80 and 443) in the server's firewall…"
for port in 443 80; do
  iptables -C INPUT -p tcp --dport "$port" -j ACCEPT 2>/dev/null || iptables -I INPUT 1 -p tcp --dport "$port" -j ACCEPT \
    || log "Couldn't open port $port in the firewall; Octopus may not be reachable from outside."
done
netfilter-persistent save >/dev/null 2>&1 || true

if ! command -v ollama >/dev/null; then
  log "Installing Ollama…"
  curl -fsSL https://ollama.com/install.sh | sh
fi
systemctl enable --now ollama

if ! command -v caddy >/dev/null; then
  log "Installing Caddy (HTTPS)…"
  arch=$(dpkg --print-architecture)
  curl -fsSL "https://caddyserver.com/api/download?os=linux&arch=${arch}" -o /usr/local/bin/caddy
  chmod +x /usr/local/bin/caddy
fi

log "Installing Octopus…"
id octopus >/dev/null 2>&1 || useradd --system --create-home --home-dir /var/lib/octopus --shell /usr/sbin/nologin octopus
if [ -d "$APP/.git" ]; then
  git -C "$APP" fetch -q --depth 1 origin "$OCTOPUS_REF" && git -C "$APP" reset -q --hard FETCH_HEAD
else
  git clone -q --depth 1 -b "$OCTOPUS_REF" "$OCTOPUS_REPO" "$APP"
fi
[ -x "$APP/.venv/bin/python" ] || python3 -m venv "$APP/.venv"
"$APP/.venv/bin/pip" install -q --upgrade pip
"$APP/.venv/bin/pip" install -q -r "$APP/requirements.txt"
mkdir -p /var/lib/octopus/workspace
chown -R octopus:octopus /var/lib/octopus

umask 077
cat > /etc/octopus.env <<EOF
OCTOPUS_PASSWORD=${OCTOPUS_PASSWORD}
EOF
umask 022

cat > /etc/systemd/system/octopus.service <<EOF
[Unit]
Description=Octopus AI assistant
After=network-online.target ollama.service
Wants=network-online.target ollama.service

[Service]
User=octopus
EnvironmentFile=/etc/octopus.env
Environment=OCTOPUS_HOME=/var/lib/octopus/workspace OCTOPUS_HOST=127.0.0.1 OCTOPUS_PORT=8000 PYTHONUNBUFFERED=1
WorkingDirectory=$APP
ExecStart=$APP/.venv/bin/python -m octopus
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

ip=$(curl -fsS --max-time 10 https://api.ipify.org || curl -fsS --max-time 10 https://ifconfig.me)
host="$(echo "$ip" | tr . -).sslip.io"
mkdir -p /etc/caddy
cat > /etc/caddy/Caddyfile <<EOF
$host {
	encode gzip
	reverse_proxy 127.0.0.1:8000 {
		flush_interval -1
	}
}
EOF
cat > /etc/systemd/system/caddy.service <<'EOF'
[Unit]
Description=Caddy (HTTPS for Octopus)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/local/bin/caddy run --config /etc/caddy/Caddyfile
ExecReload=/usr/local/bin/caddy reload --config /etc/caddy/Caddyfile
Environment=XDG_DATA_HOME=/var/lib/caddy XDG_CONFIG_HOME=/var/lib/caddy
Restart=always
AmbientCapabilities=CAP_NET_BIND_SERVICE

[Install]
WantedBy=multi-user.target
EOF

cat > /usr/local/bin/octopus-update <<EOF
#!/usr/bin/env bash
# Update Octopus to the newest version from GitHub.
set -e
curl -fsSL https://raw.githubusercontent.com/physicstheory/octopus/main/cloud/oracle-setup.sh | sudo bash
EOF
chmod +x /usr/local/bin/octopus-update

systemctl daemon-reload
systemctl enable --now octopus caddy
systemctl restart octopus caddy

log "Downloading the everyday chat models…"
for i in $(seq 1 30); do curl -sf http://127.0.0.1:11434/api/version >/dev/null && break; sleep 2; done
for m in $CHAT_MODELS; do ollama pull "$m" >/dev/null || log "Couldn't download $m now; Octopus downloads it the first time it's picked."; done

log "Done. Octopus is at https://$host"
echo "https://$host" > /var/lib/octopus/url
