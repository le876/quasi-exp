#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${1:-}"
PY_BASENAME="${2:-python3.10}"

if [[ -z "${ENV_PREFIX}" ]]; then
  echo "usage: $0 <env_prefix> [python_binary_name]"
  echo "example: $0 /mnt/ML_projects/conda_envs/dante_env python3.10"
  exit 2
fi

BIN_DIR="${ENV_PREFIX%/}/bin"
TARGET="${BIN_DIR}/${PY_BASENAME}"

if [[ ! -d "${BIN_DIR}" ]]; then
  echo "ERROR: bin dir not found: ${BIN_DIR}" >&2
  exit 1
fi
if [[ ! -x "${TARGET}" ]]; then
  echo "ERROR: target python not executable: ${TARGET}" >&2
  ls -la "${BIN_DIR}" | sed -n '1,60p' >&2 || true
  exit 1
fi

backup_if_needed() {
  local p="$1"
  if [[ -e "${p}" && ! -L "${p}" ]]; then
    mv -f "${p}" "${p}.bak.$(date +%Y%m%d_%H%M%S)"
  elif [[ -L "${p}" ]]; then
    rm -f "${p}"
  fi
}

for name in python python3 python3.1; do
  backup_if_needed "${BIN_DIR}/${name}"
done

ln -s "${PY_BASENAME}" "${BIN_DIR}/python"
ln -s "${PY_BASENAME}" "${BIN_DIR}/python3"

echo "OK: linked ${BIN_DIR}/python -> ${PY_BASENAME}"
echo "OK: linked ${BIN_DIR}/python3 -> ${PY_BASENAME}"

