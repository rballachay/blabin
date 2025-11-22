#!/bin/bash
set -euo pipefail

mkdir -p /app/logs

# Set timezone if TZ provided
if [[ -n "${TZ:-}" && -f "/usr/share/zoneinfo/${TZ}" ]]; then
  sudo ln -sf "/usr/share/zoneinfo/${TZ}" /etc/localtime || true
  echo "${TZ}" | sudo tee /etc/timezone >/dev/null || true
fi

echo "Starting scheduler with supercronic..."
exec /usr/local/bin/supercronic -quiet /app/scripts/blabin.cron
