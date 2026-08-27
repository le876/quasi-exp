#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python
CONFIG="$SOURCE_ROOT/configs/bacra_v14_3r_retry11_dual_track_zero_bridge.yaml"
OUTPUT_ROOT=${1:-/mnt/ML_projects/quasi_exp/runs/bacra_v14_3r_retry11_dual_track_zero_bridge_attempt1}
BINDING_SHA=${2:-${BACRA_RETRY11_BINDING_SHA:-}}
RUNNER="$SOURCE_ROOT/scripts/analysis/run_bacra_v14_3r_retry11.py"

if [[ -z "$BINDING_SHA" ]]; then
  echo "retry11 launcher requires binding SHA as argument 2 or BACRA_RETRY11_BINDING_SHA" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export PYTHONHASHSEED=20260881
export MPLCONFIGDIR=/tmp/bacra_retry11_matplotlib_cache

stages=(
  inventory
  outer_sparse_wide
  outer_student
  outer_theta
  outer_tension_pilot
  outer_tension_materialization
  zero_seed
  zero_frontier_poc
  zero_outer_bridge
  summary
)

cd "$SOURCE_ROOT"
for stage in "${stages[@]}"; do
  if "$PYTHON_BIN" "$RUNNER" \
    --config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --binding-sha "$BINDING_SHA" \
    --validate-stage "$stage" >/dev/null 2>&1; then
    continue
  fi
  "$PYTHON_BIN" "$RUNNER" \
    --config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --binding-sha "$BINDING_SHA" \
    --stage "$stage"
done
