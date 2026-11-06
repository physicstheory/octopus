#!/usr/bin/env bash
# Puts Octopus online on Oracle Cloud's Always Free tier. Run it in Oracle Cloud Shell:
#   bash <(curl -fsSL https://raw.githubusercontent.com/physicstheory/octopus/main/cloud/oracle-launch.sh)
# It creates a network and a free Ampere server (4 cores, 24 GB), installs Octopus on it with
# cloud/oracle-setup.sh, and prints your web address. Running it again reuses what it already made.
set -uo pipefail

say() { printf '\n🐙 %s\n' "$*"; }
die() { printf '\n❌ %s\n' "$*"; exit 1; }
q() { "$@" 2>/dev/null | tr -d '"'; }

command -v oci >/dev/null || die "Run this in Oracle Cloud Shell (the >_ button at the top right of cloud.oracle.com)."
C="${OCI_TENANCY:-}"
[ -n "$C" ] || die "Couldn't find your Oracle account (OCI_TENANCY). Run this in Oracle Cloud Shell."

if [ -z "${OCTOPUS_PASSWORD:-}" ]; then
  while :; do
    read -r -p "Make up an invite code for Octopus (letters and numbers, 4 or more): " OCTOPUS_PASSWORD
    [[ "$OCTOPUS_PASSWORD" =~ ^[A-Za-z0-9_-]{4,}$ ]] && break
    echo "Use only letters, numbers, - or _, at least 4 of them."
  done
fi

# --- already running? ---
INST=$(q oci compute instance list -c "$C" --display-name octopus --lifecycle-state RUNNING --query 'data[0].id' --raw-output)
if [ -n "$INST" ] && [ "$INST" != "null" ]; then
  IP=$(q oci compute instance list-vnics --instance-id "$INST" --query 'data[0]."public-ip"' --raw-output)
  say "Octopus's server already exists. It's at https://$(echo "$IP" | tr . -).sslip.io"
  exit 0
fi

# --- network ---
say "Setting up the network…"
VCN=$(q oci network vcn list -c "$C" --display-name octopus-vcn --lifecycle-state AVAILABLE --query 'data[0].id' --raw-output)
if [ -z "$VCN" ] || [ "$VCN" = "null" ]; then
  VCN=$(q oci network vcn create -c "$C" --cidr-blocks '["10.0.0.0/16"]' --display-name octopus-vcn --dns-label octopus \
        --wait-for-state AVAILABLE --query 'data.id' --raw-output) || true
  [ -n "$VCN" ] || die "Oracle wouldn't create the network. Check that your account finished signing up, then run this again."
  IGW=$(q oci network internet-gateway create -c "$C" --vcn-id "$VCN" --is-enabled true --display-name octopus-internet \
        --wait-for-state AVAILABLE --query 'data.id' --raw-output)
  RT=$(q oci network vcn get --vcn-id "$VCN" --query 'data."default-route-table-id"' --raw-output)
  oci network route-table update --rt-id "$RT" --force \
    --route-rules "[{\"destination\":\"0.0.0.0/0\",\"destinationType\":\"CIDR_BLOCK\",\"networkEntityId\":\"$IGW\"}]" >/dev/null
  SL=$(q oci network vcn get --vcn-id "$VCN" --query 'data."default-security-list-id"' --raw-output)
  rule() { echo "{\"source\":\"0.0.0.0/0\",\"protocol\":\"6\",\"isStateless\":false,\"tcpOptions\":{\"destinationPortRange\":{\"min\":$1,\"max\":$1}}}"; }
  oci network security-list update --security-list-id "$SL" --force \
    --ingress-security-rules "[$(rule 22),$(rule 80),$(rule 443)]" \
    --egress-security-rules '[{"destination":"0.0.0.0/0","protocol":"all","isStateless":false}]' >/dev/null
  oci network subnet create -c "$C" --vcn-id "$VCN" --cidr-block 10.0.0.0/24 --display-name octopus-subnet --dns-label main \
    --wait-for-state AVAILABLE >/dev/null || die "Oracle wouldn't create the subnet. Run this again in a minute."
fi
SUBNET=$(q oci network subnet list -c "$C" --vcn-id "$VCN" --display-name octopus-subnet --query 'data[0].id' --raw-output)
[ -n "$SUBNET" ] && [ "$SUBNET" != "null" ] || die "Couldn't find Octopus's subnet. Run this again."

# --- server ---
IMG=$(q oci compute image list -c "$C" --operating-system "Canonical Ubuntu" --operating-system-version "24.04" \
      --shape VM.Standard.A1.Flex --sort-by TIMECREATED --sort-order DESC --limit 1 --query 'data[0].id' --raw-output)
[ -n "$IMG" ] && [ "$IMG" != "null" ] || die "Couldn't find Ubuntu 24.04 for the free Ampere server in your region."

[ -f ~/.ssh/id_rsa.pub ] || ssh-keygen -q -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa
INIT=$(mktemp)
cat > "$INIT" <<EOF
#!/bin/bash
export OCTOPUS_PASSWORD='$OCTOPUS_PASSWORD'
for i in 1 2 3 4 5; do
  curl -fsSL https://raw.githubusercontent.com/physicstheory/octopus/main/cloud/oracle-setup.sh -o /root/octopus-setup.sh && break
  sleep 10
done
bash /root/octopus-setup.sh > /var/log/octopus-setup.log 2>&1
EOF

ADS=$(q oci iam availability-domain list -c "$C" --query 'data[].name' --raw-output | tr -d '[],' | xargs)
say "Creating your free server (4 cores, 24 GB of memory)…"
echo "   Oracle's free servers are popular. If your region is full, this keeps trying every minute (for up to 3 hours)."
echo "   You can leave this tab open and come back later."
INST=""
for attempt in $(seq 1 180); do
  for AD in $ADS; do
    OUT=$(oci compute instance launch -c "$C" --availability-domain "$AD" --display-name octopus \
          --shape VM.Standard.A1.Flex --shape-config '{"ocpus":4,"memoryInGBs":24}' \
          --image-id "$IMG" --subnet-id "$SUBNET" --assign-public-ip true --boot-volume-size-in-gbs 100 \
          --ssh-authorized-keys-file ~/.ssh/id_rsa.pub --user-data-file "$INIT" \
          --wait-for-state RUNNING --query 'data.id' --raw-output 2>&1)
    if [[ "$OUT" =~ ocid1\.instance\.[A-Za-z0-9._-]+ ]]; then INST="${BASH_REMATCH[0]}"; break 2; fi
    if echo "$OUT" | grep -qi "capacity"; then
      continue
    elif echo "$OUT" | grep -qi "LimitExceeded\|limit"; then
      die "Oracle says you've reached a free-tier limit. Delete any other Ampere server in Compute → Instances, then run this again."
    elif echo "$OUT" | grep -qi "TooManyRequests"; then
      sleep 30
    else
      echo "$OUT" | tail -5
      die "Oracle didn't create the server (see above). Send a screenshot of this to Claude."
    fi
  done
  printf '   %s  Oracle is full right now, trying again in a minute (try %s of 180)…\n' "$(date +%H:%M)" "$attempt"
  sleep 60
done
rm -f "$INIT"
[ -n "$INST" ] || die "Oracle stayed full for 3 hours. Run this again later; nights and weekends are usually quieter."

IP=$(q oci compute instance list-vnics --instance-id "$INST" --query 'data[0]."public-ip"' --raw-output)
URL="https://$(echo "$IP" | tr . -).sslip.io"
say "Your server is running. It's now installing Octopus (about 10 minutes)…"
for i in $(seq 1 60); do
  if curl -fsS --max-time 5 "$URL/healthz" >/dev/null 2>&1; then
    say "🎉 Octopus is online: $URL"
    echo "   Open it, create your account with your invite code ($OCTOPUS_PASSWORD), and bookmark it."
    exit 0
  fi
  sleep 20
done
say "The server is still setting up. Try $URL in 10 minutes."
echo "   (To check on it: ssh ubuntu@$IP 'tail /var/log/octopus-setup.log')"
