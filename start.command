#!/usr/bin/env bash
# Double-click this on a Mac to start Octopus.
cd "$(dirname "$0")"
chmod +x start.sh install_studio.sh 2>/dev/null
exec ./start.sh
