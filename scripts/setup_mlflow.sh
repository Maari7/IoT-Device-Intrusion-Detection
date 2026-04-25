#!/usr/bin/env bash
set -euo pipefail

MLFLOW_HOST="${MLFLOW_HOST:-0.0.0.0}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-sqlite:///mlflow.db}"
MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-./mlruns}"

echo "Starting MLflow server on ${MLFLOW_HOST}:${MLFLOW_PORT}"
mlflow server \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}" \
  --host "${MLFLOW_HOST}" \
  --port "${MLFLOW_PORT}"
