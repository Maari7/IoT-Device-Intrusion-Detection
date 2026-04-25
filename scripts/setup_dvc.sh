#!/usr/bin/env bash
set -euo pipefail

REMOTE_NAME="${DVC_REMOTE_NAME:-storage}"
REMOTE_URL="${DVC_REMOTE_URL:-}"

if ! command -v dvc >/dev/null 2>&1; then
  echo "dvc command not found. Install DVC first."
  exit 1
fi

if [ ! -d ".dvc" ]; then
  dvc init
fi

if [ -n "$REMOTE_URL" ]; then
  dvc remote add -f "$REMOTE_NAME" "$REMOTE_URL"
  dvc remote default "$REMOTE_NAME"
  echo "Configured DVC remote '$REMOTE_NAME'"
else
  echo "DVC remote not configured. Set DVC_REMOTE_URL to configure remote storage."
fi

dvc add data/raw data/processed data/features

echo "DVC setup complete."
