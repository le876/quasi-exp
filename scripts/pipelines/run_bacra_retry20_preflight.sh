#!/usr/bin/env bash
set -euo pipefail
SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python
OUTPUT_ROOT=${1:?output root required}
BINDING_SHA=${2:?binding SHA required}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec timeout --signal=TERM --kill-after=30s 12h "$PYTHON" "$SOURCE_ROOT/scripts/analysis/run_bacra_retry20_preflight.py" \
  --config "$SOURCE_ROOT/configs/bacra_retry20_preflight.yaml" --output-root "$OUTPUT_ROOT" --binding-sha "$BINDING_SHA"
