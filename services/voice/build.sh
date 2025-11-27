#!/usr/bin/env bash

IMAGE_NAME=blabin-voice
IMAGE_TAG=latest
REGISTRY_REPO=fastapi-containers
CREDS_FILE=../../.creds/gcp-sa-key.json
CR_SERVICE_NAME=fastapi-service

# Config via env vars
GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:?set GOOGLE_CLOUD_PROJECT}"
REGION="${REGION:-us-central1}"

# tagging
LOCAL_TAG="${IMAGE_NAME}:${IMAGE_TAG}"
REMOTE_TAG="${REGION}-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/${REGISTRY_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

# auth
gcloud auth activate-service-account --key-file "${CREDS_FILE}" --quiet
gcloud config set project "${GOOGLE_CLOUD_PROJECT}" --quiet
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

echo "[build] ${LOCAL_TAG} from ${SERVICE_DIR}"
docker build -t "${LOCAL_TAG}" .

echo "[tag] ${LOCAL_TAG} -> ${REMOTE_TAG}"
docker tag "${LOCAL_TAG}" "${REMOTE_TAG}"

echo "[push] ${REMOTE_TAG}"
docker push "${REMOTE_TAG}"

# Resolve the pushed tag to an immutable digest
echo "[digest] resolving digest for ${REMOTE_TAG}"
DIGEST="$(gcloud artifacts docker images describe "${REMOTE_TAG}" --format='value(image_summary.digest)')"
IMAGE_WITH_DIGEST="${REGION}-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/${REGISTRY_REPO}/${IMAGE_NAME}@${DIGEST}"
echo "[update] deploying ${IMAGE_WITH_DIGEST} to Cloud Run service ${CR_SERVICE_NAME}"

gcloud run services update "${CR_SERVICE_NAME}" \
  --region "${REGION}" \
  --image "${IMAGE_WITH_DIGEST}" \
  --quiet

URL="$(gcloud run services describe "${CR_SERVICE_NAME}" --region "${REGION}" --format='value(status.url)')"
echo "[done] updated ${CR_SERVICE_NAME} -> ${IMAGE_WITH_DIGEST}"
echo "[url] ${URL}"
