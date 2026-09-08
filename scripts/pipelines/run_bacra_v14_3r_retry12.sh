#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python
CONFIG="$SOURCE_ROOT/configs/bacra_v14_3r_retry12_symmetry_quotient_zero_centered.yaml"
OUTPUT_ROOT=${1:-/mnt/ML_projects/quasi_exp/runs/bacra_v14_3r_retry12_symmetry_quotient_attempt1}
BINDING_SHA=${2:-${BACRA_RETRY12_BINDING_SHA:-}}
RUNNER="$SOURCE_ROOT/scripts/analysis/run_bacra_v14_3r_retry12.py"

if [[ -z "$BINDING_SHA" ]]; then
  echo "retry12 launcher requires binding SHA as argument 2 or BACRA_RETRY12_BINDING_SHA" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export PYTHONHASHSEED=20260882
export MPLCONFIGDIR=/tmp/bacra_retry12_matplotlib_cache

stages=(
  symmetry_contract
  quotient_target_registry
  zero_rooted_outward_atlas
  symmetry_fixed_seams
  orbit_selection_expansion_split
  full_graph_connectivity_audit
  freeze_kinematic_dataset
  quotient_student
  zero_to_target_trajectory_audit
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
