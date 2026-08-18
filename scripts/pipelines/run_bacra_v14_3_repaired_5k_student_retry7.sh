#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${SOURCE_ROOT%%/.worktrees/*}
PYTHON_BIN=/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python
RUNNER="$SOURCE_ROOT/scripts/analysis/run_bacra_v14_3_repaired_5k_student.py"
CONFIG="$SOURCE_ROOT/configs/bacra_v14_3_repaired_5k_student_retry7.yaml"
OUTPUT_ROOT="$PROJECT_ROOT/runs/bacra_v14_3_repaired_5k_student_retry7"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export MPLCONFIGDIR=/tmp/mpl-bacra-v14-3-retry7

run_stage() {
  local stage_name=$1
  if "$PYTHON_BIN" "$RUNNER" --config "$CONFIG" --output-root "$OUTPUT_ROOT" --validate-stage "$stage_name" >/dev/null 2>&1; then
    echo "stage=$stage_name status=sealed_skip"
    return
  fi
  echo "stage=$stage_name status=starting"
  "$PYTHON_BIN" "$RUNNER" --config "$CONFIG" --output-root "$OUTPUT_ROOT" --stage "$stage_name"
  echo "stage=$stage_name status=completed"
}

run_stage inventory
run_stage pilot_registry
run_stage root_charts
run_stage stitched_atlas
run_stage fixed_budget_dataset
run_stage students
run_stage representation_decision
run_stage reach_update
run_stage formal_admission
run_stage summary
