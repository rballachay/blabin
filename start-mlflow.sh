#!/usr/bin/env bash

HOST="127.0.0.1"
PORT="${PORT:-8080}"
LOG_DIR="/workspaces/blabin/data/logs"
LOG_FILE="${LOG_DIR}/mlflow.log"
ARTIFACT_DIR="/workspaces/blabin/mlruns"

mkdir -p "$(dirname "${LOG_FILE}")" "${ARTIFACT_DIR}"

# If port is already open, assume MLflow is running
if command -v ss >/dev/null 2>&1 && ss -lnt | grep -q ":${PORT}\b"; then
  echo "[mlflow] Already running on ${HOST}:${PORT}"
  exit 0
fi

mlflow server \
    --host $HOST --port $PORT \
    --backend-store-uri sqlite:///data/mlflow.db \
    --default-artifact-root "file:${ARTIFACT_DIR}" \
    --serve-artifacts   \
    >"${LOG_FILE}" 2>&1 &

echo "[mlflow] Started on ${HOST}:${PORT}, logs: ${LOG_FILE}"
