#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="dante_env"
DATASET_PATH=""
ROBOT_CONFIG=""
OUT_DIR=""
MODELS="mlp,mlp_large,rf,lgbm,knn"

if [[ $# -ge 5 ]]; then
  ENV_NAME="${1}"
  DATASET_PATH="${2}"
  ROBOT_CONFIG="${3}"
  OUT_DIR="${4}"
  MODELS="${5}"
elif [[ $# -eq 4 ]]; then
  DATASET_PATH="${1}"
  ROBOT_CONFIG="${2}"
  OUT_DIR="${3}"
  MODELS="${4}"
elif [[ $# -eq 3 ]]; then
  DATASET_PATH="${1}"
  ROBOT_CONFIG="${2}"
  OUT_DIR="${3}"
fi

if [[ -z "${DATASET_PATH}" || -z "${ROBOT_CONFIG}" || -z "${OUT_DIR}" ]]; then
  echo "usage(new): $0 <conda_env_name> <dataset.parquet> <robot_config.yaml> <out_dir> [models_csv]"
  echo "usage(compat): $0 <dataset.parquet> <robot_config.yaml> <out_dir> [models_csv]"
  exit 2
fi

echo "Waiting for dataset: ${DATASET_PATH}"
while [[ ! -f "${DATASET_PATH}" ]]; do
  sleep 60
done

mkdir -p "${OUT_DIR}"
echo "Found dataset. Running baselines in env=${ENV_NAME} models=${MODELS}"

REQ_PKGS="pyarrow,sklearn,numpy,pandas"
if [[ "${MODELS}" == *"lgbm"* ]]; then
  REQ_PKGS="${REQ_PKGS},lightgbm"
fi
conda run -n "${ENV_NAME}" python scripts/check_ml_env.py \
  --require-packages "${REQ_PKGS}" \
  --check-tf-gpu \
  --output "${OUT_DIR}/env_check.json"

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
conda run -n "${ENV_NAME}" python scripts/baselines/run_baselines.py \
  --dataset "${DATASET_PATH}" \
  --robot-config "${ROBOT_CONFIG}" \
  --out-dir "${OUT_DIR}" \
  --seed 20260207 --val-size 0.1 --test-size 0.1 \
  --models "${MODELS}" |& tee "${OUT_DIR}/run.log"
