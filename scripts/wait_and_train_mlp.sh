#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="dante_env"
DATASET_PATH=""
ROBOT_CONFIG=""
OUT_DIR=""

if [[ $# -ge 4 ]]; then
  ENV_NAME="${1}"
  DATASET_PATH="${2}"
  ROBOT_CONFIG="${3}"
  OUT_DIR="${4}"
elif [[ $# -eq 3 ]]; then
  DATASET_PATH="${1}"
  ROBOT_CONFIG="${2}"
  OUT_DIR="${3}"
fi

if [[ -z "${DATASET_PATH}" || -z "${ROBOT_CONFIG}" || -z "${OUT_DIR}" ]]; then
  echo "usage(new): $0 <conda_env_name> <dataset.parquet> <robot_config.yaml> <out_dir>"
  echo "usage(compat): $0 <dataset.parquet> <robot_config.yaml> <out_dir>"
  exit 2
fi

echo "Waiting for dataset: ${DATASET_PATH}"
while [[ ! -f "${DATASET_PATH}" ]]; do
  sleep 30
done
echo "Found dataset, checking env=${ENV_NAME} ..."

mkdir -p "${OUT_DIR}"
conda run -n "${ENV_NAME}" python scripts/check_ml_env.py \
  --require-packages tensorflow,pyarrow,sklearn,numpy,pandas \
  --require-tf-gpu \
  --output "${OUT_DIR}/env_check.json"

echo "Env check passed, starting training in env=${ENV_NAME} ..."
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  nice -n 10 \
  conda run -n "${ENV_NAME}" python scripts/train_mlp.py \
    --dataset "${DATASET_PATH}" \
    --robot-config "${ROBOT_CONFIG}" \
    --out-dir "${OUT_DIR}" \
    --early-stopping |& tee "${OUT_DIR}/train.log"
