#!/usr/bin/env bash

IMAGE_NAME=blabin-voice
IMAGE_TAG=latest
REGISTRY_REPO=fastapi-containers
CREDS_FILE=../../.creds/gcp-sa-key.json

# Config via env vars
GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:?set GOOGLE_CLOUD_PROJECT}"
REGION="${REGION:-us-central1}"

# tagging
LOCAL_TAG="${IMAGE_NAME}:${IMAGE_TAG}"
REMOTE_TAG="${REGION}-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/${REGISTRY_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

# auth
gcloud auth activate-service-account --key-file "${CREDS_FILE}" --quiet
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

echo "[build] ${LOCAL_TAG} from ${SERVICE_DIR}"
docker build -t "${LOCAL_TAG}" .

echo "[tag] ${LOCAL_TAG} -> ${REMOTE_TAG}"
docker tag "${LOCAL_TAG}" "${REMOTE_TAG}"

echo "[push] ${REMOTE_TAG}"
docker push "${REMOTE_TAG}"

echo "[done] pushed ${REMOTE_TAG}"
