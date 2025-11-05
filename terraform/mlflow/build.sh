#!/bin/bash
set -euo pipefail

# Load variables from repo .env if present (exports all vars)
ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
ENV_FILE="$ROOT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

# Resolve config (precedence: exported env > .env > defaults)
PROJECT="${PROJECT:-${GOOGLE_CLOUD_PROJECT:-}}"
REGION="${REGION:-${GCP_REGION:-us-central1}}"
OWNER_EMAIL="${OWNER_EMAIL:-}"

echo "Using PROJECT=${PROJECT} REGION=${REGION} OWNER_EMAIL=${OWNER_EMAIL}"

gcloud config set project "$PROJECT"

# Enable required APIs on the project (idempotent)
gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com --project "$PROJECT"

# Create (or reuse) the Artifact Registry repo
gcloud artifacts repositories describe mlflow --location="$REGION" --project "$PROJECT" >/dev/null 2>&1 || \
gcloud artifacts repositories create mlflow \
  --repository-format=DOCKER \
  --location="$REGION" \
  --description="Custom MLflow images" \
  --project "$PROJECT"

# IAM for builds
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="user:${OWNER_EMAIL}" \
  --role="roles/cloudbuild.builds.editor" >/dev/null

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
gcloud artifacts repositories add-iam-policy-binding mlflow \
  --location="$REGION" \
  --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  --role="roles/artifactregistry.writer" \
  --project "$PROJECT" >/dev/null

# Build and push
cd /workspaces/blabin/terraform/mlflow
IMAGE_REF="${REGION}-docker.pkg.dev/${PROJECT}/mlflow/mlflow:v3.4.0"
gcloud builds submit --tag "$IMAGE_REF" .
