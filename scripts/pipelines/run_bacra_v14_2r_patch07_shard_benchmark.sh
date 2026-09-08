#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${SOURCE_ROOT%%/.worktrees/*}
PYTHON_BIN=/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python
RUNNER="$SOURCE_ROOT/scripts/analysis/run_bacra_v14_2r_stitched_atlas.py"
REPORTER="$SOURCE_ROOT/scripts/analysis/report_bacra_v14_2r_patch_benchmark.py"
CONFIG="$SOURCE_ROOT/configs/bacra_v14_2r_stitched_atlas.yaml"
OUTPUT_ROOT="$PROJECT_ROOT/runs/bacra_v14_2r_patch07_shard_benchmark_retry1"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export MPLCONFIGDIR=/tmp/mpl-bacra-v14-2r-benchmark

run_stage() {
  local stage_name=$1
  local stage_dir=$2
  if [[ -f "$OUTPUT_ROOT/$stage_dir/gate.json" ]]; then
    echo "stage=$stage_name status=sealed_skip"
    return
  fi
  echo "stage=$stage_name status=starting"
  "$PYTHON_BIN" "$RUNNER" --config "$CONFIG" --output-root "$OUTPUT_ROOT" --stage "$stage_name"
  echo "stage=$stage_name status=completed"
}

run_stage inventory 00_inventory
run_stage replacement_confirmation 01_replacement_confirmation
run_stage artifact_diagnostics 02_artifact_diagnostics

if [[ ! -f "$OUTPUT_ROOT/03_rooted_baseline/patch_07/baseline/report.json" ]]; then
  echo "stage=patch07_baseline_benchmark status=starting audit_shards=12"
  "$PYTHON_BIN" "$RUNNER" \
    --config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --stage rooted_baseline \
    --patch-id patch_07 \
    --variant baseline \
    --audit-shards-per-patch 12
  echo "stage=patch07_baseline_benchmark status=completed"
fi

"$PYTHON_BIN" "$REPORTER" --config "$CONFIG" --output-root "$OUTPUT_ROOT"
