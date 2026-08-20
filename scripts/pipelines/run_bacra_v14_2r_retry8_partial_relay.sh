#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${SOURCE_ROOT%%/.worktrees/*}
PYTHON_BIN=/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python
RUNNER="$SOURCE_ROOT/scripts/analysis/run_bacra_v14_2r_retry8.py"
CONFIG="$SOURCE_ROOT/configs/bacra_v14_2r_retry8_partial_relay.yaml"
OUTPUT_ROOT="$PROJECT_ROOT/runs/bacra_v14_2r_retry8_partial_relay_retry3"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export MPLCONFIGDIR=/tmp/mpl-bacra-v14-2r-retry8

run_stage() {
  local stage_name=$1
  echo "stage=$stage_name status=starting_or_validated_resume"
  "$PYTHON_BIN" "$RUNNER" \
    --config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --stage "$stage_name"
  echo "stage=$stage_name status=completed"
}

run_stage inventory
run_stage frozen_partial
run_stage relay_registry
run_stage growth_funnel
run_stage sampled_screening
run_stage partial_certificate
run_stage seed_labels
run_stage authorization
run_stage summary

echo "retry8_status=operationally_complete downstream_smoke=separate_authorization"
