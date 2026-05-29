#!/usr/bin/env bash
set -euo pipefail

# User-level cleanup only. Safe to run from cron.

HOME_DIR="${HOME:-/home/ubuntu}"
TMP_BASE="/mnt/ML_projects/tmp"

mkdir -p "${TMP_BASE}"

cleanup_path() {
  local p="$1"
  if [[ -e "${p}" ]]; then
    rm -rf "${p}" || true
  fi
}

# Browser/app caches.
cleanup_path "${HOME_DIR}/.cache/microsoft-edge"
cleanup_path "${HOME_DIR}/.cache/google-chrome"
cleanup_path "${HOME_DIR}/.cache/chat-gpt"
cleanup_path "${HOME_DIR}/.cache/node-gyp"

# Trash.
cleanup_path "${HOME_DIR}/.local/share/Trash/files"
cleanup_path "${HOME_DIR}/.local/share/Trash/info"

# NPM cache.
if command -v npm >/dev/null 2>&1; then
  npm cache clean --force >/dev/null 2>&1 || true
fi

# Pip cache.
if command -v pip >/dev/null 2>&1; then
  pip cache purge >/dev/null 2>&1 || true
fi

# Temporary files created by local experiments.
cleanup_path "/tmp/lgbm_fallback_smoke"
cleanup_path "/tmp/baselines_fallback_smoke"
cleanup_path "/tmp/lgbm_fallback_params.yaml"
cleanup_path "/tmp/baselines_gpu_overrides.yaml"

df -h / | sed -n '1,3p'
