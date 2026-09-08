#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${SOURCE_ROOT%%/.worktrees/*}
PYTHON_BIN=/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python
RETRY5_RUNNER="$SOURCE_ROOT/scripts/analysis/run_bacra_v14_2r_stitched_atlas.py"
RETRY5_CONFIG="$SOURCE_ROOT/configs/bacra_v14_2r_stitched_atlas_retry5.yaml"
RETRY5_OUTPUT="$PROJECT_ROOT/runs/bacra_v14_2r_stitched_atlas_retry5"
V143_PIPELINE="$SOURCE_ROOT/scripts/pipelines/run_bacra_v14_3_repaired_5k_student.sh"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export MPLCONFIGDIR=/tmp/mpl-bacra-v14-2r-retry5

run_stage() {
  local stage_name=$1
  echo "stage=$stage_name status=starting_or_validated_resume"
  "$PYTHON_BIN" "$RETRY5_RUNNER" \
    --config "$RETRY5_CONFIG" \
    --output-root "$RETRY5_OUTPUT" \
    --stage "$stage_name"
  echo "stage=$stage_name status=completed"
}

run_stage inventory
run_stage replacement_confirmation
run_stage artifact_diagnostics
run_stage reach_round7
run_stage rooted_baseline
run_stage registered_retry
run_stage patch07_local_audit
run_stage search_stability
run_stage mechanism_gate
run_stage confirmation
run_stage meso_bridge
run_stage summary

"$PYTHON_BIN" - "$RETRY5_OUTPUT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "11_summary/gate.json").read_text(encoding="utf-8"))
if not bool(summary.get("deployment_authorized", False)):
    raise SystemExit("retry5 did not authorize repaired 5k; V14.3 remains blocked")
PY

exec "$V143_PIPELINE"
