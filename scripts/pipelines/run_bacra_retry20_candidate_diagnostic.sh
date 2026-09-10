#!/usr/bin/env bash
set -euo pipefail
SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
exec timeout --signal=TERM --kill-after=30s 12h /mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python \
  "$SOURCE_ROOT/scripts/analysis/run_bacra_retry20_candidate_diagnostic.py" --config "$SOURCE_ROOT/configs/bacra_retry20_candidate_diagnostic.yaml" \
  --output-root "${1:?output root required}" --binding-sha "${2:?binding SHA required}"
