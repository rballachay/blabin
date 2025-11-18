#!/bin/bash
set -euo pipefail

if [[ ! -f "${GCP_SA_KEY_FILE}" ]]; then
  echo "Service account key not found at ${GCP_SA_KEY_FILE}" >&2
  exit 1
fi

gcloud auth activate-service-account --key-file "${GCP_SA_KEY_FILE}"
gcloud config set project "${GOOGLE_CLOUD_PROJECT}"
exec gcloud run services proxy "${SERVICE}" --region "${GCP_REGION}" --port "${PORT}"
