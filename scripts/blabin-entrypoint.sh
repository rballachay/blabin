#!/bin/bash
set -euo pipefail

# Recreate creds from env
if [[ -n "${GOOGLE_BASE_64_KEY:-}" && ! -f "/app/$GOOGLE_APPLICATION_CREDENTIALS" ]]; then
  mkdir -p "$(dirname "$GOOGLE_APPLICATION_CREDENTIALS")"
  echo "$GOOGLE_BASE_64_KEY" | base64 -d > /app/"$GOOGLE_APPLICATION_CREDENTIALS"
  chmod 600 /app/"$GOOGLE_APPLICATION_CREDENTIALS"
fi

mkdir -p /app/logs
echo "Starting scheduler with supercronic..."
exec /usr/local/bin/supercronic -quiet /app/scripts/blabin.cron
