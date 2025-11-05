#!/bin/bash
set -euo pipefail

# Load .env if present
ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
ENV_FILE="$ROOT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

REGION="${GCP_REGION:-us-central1}"
SERVICE="${SERVICE:-mlflow}"

gcloud config set project "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID in .env or env}"

# Start local proxy (injects auth), then open in browser
PORT="${PORT:-8081}"
echo "Proxying Cloud Run service '$SERVICE' in region '$REGION' to http://localhost:${PORT}"
gcloud run services proxy "$SERVICE" --region "$REGION" --port "$PORT"
