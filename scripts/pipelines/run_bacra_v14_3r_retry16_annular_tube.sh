#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python
CONFIG="$SOURCE_ROOT/configs/bacra_v14_3r_retry16_omega600_annular_tube_preflight.yaml"
OUTPUT_ROOT=${1:-/mnt/ML_projects/quasi_exp/runs/bacra_v14_3r_retry16_omega600_annular_tube_preflight_attempt1}
BINDING_SHA=${2:-${BACRA_RETRY16_ANNULAR_BINDING_SHA:-}}
RUNNER="$SOURCE_ROOT/scripts/analysis/run_bacra_v14_3r_retry16_annular_tube.py"

if [[ -z "$BINDING_SHA" ]]; then
  echo "retry16 annular preflight requires binding SHA as argument 2 or BACRA_RETRY16_ANNULAR_BINDING_SHA" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

exec "$PYTHON" "$RUNNER" \
  --config "$CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --binding-sha "$BINDING_SHA"
