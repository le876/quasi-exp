#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${SOURCE_ROOT%%/.worktrees/*}
PYTHON_BIN=/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python
RUNNER="$SOURCE_ROOT/scripts/analysis/run_bacra_v14_2r_stitched_atlas.py"
CONFIG="$SOURCE_ROOT/configs/bacra_v14_2r_stitched_atlas.yaml"
OUTPUT_ROOT="$PROJECT_ROOT/runs/bacra_v14_2r_stitched_atlas"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export MPLCONFIGDIR=/tmp/mpl-bacra-v14-2r

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

reach_pid=""
if [[ ! -f "$OUTPUT_ROOT/08_reach_round7/gate.json" ]]; then
  (
    run_stage reach_round7 08_reach_round7
  ) >"$OUTPUT_ROOT/reach_round7.concurrent.log" 2>&1 &
  reach_pid=$!
fi

cleanup() {
  if [[ -n "$reach_pid" ]] && kill -0 "$reach_pid" 2>/dev/null; then
    kill "$reach_pid" 2>/dev/null || true
    wait "$reach_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

run_stage rooted_baseline 03_rooted_baseline
run_stage registered_retry 04_registered_retry
run_stage patch07_local_audit 05_patch07_local_audit
run_stage search_stability 06_search_stability
run_stage mechanism_gate 07_mechanism_gate
run_stage confirmation 09_twelve_patch_confirmation
run_stage meso_bridge 10_meso_bridge

if [[ -n "$reach_pid" ]]; then
  wait "$reach_pid"
  reach_pid=""
fi

run_stage summary 11_summary
trap - EXIT INT TERM
