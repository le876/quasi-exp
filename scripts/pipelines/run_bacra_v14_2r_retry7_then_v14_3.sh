#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${SOURCE_ROOT%%/.worktrees/*}
PYTHON_BIN=/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python
RUNNER="$SOURCE_ROOT/scripts/analysis/run_bacra_v14_2r_retry7.py"
CONFIG="$SOURCE_ROOT/configs/bacra_v14_2r_stitched_atlas_retry7.yaml"
OUTPUT_ROOT="$PROJECT_ROOT/runs/bacra_v14_2r_stitched_atlas_retry7_abstention_retry4"
V143_PIPELINE="$SOURCE_ROOT/scripts/pipelines/run_bacra_v14_3_repaired_5k_student_retry7.sh"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export MPLCONFIGDIR=/tmp/mpl-bacra-v14-2r-retry7

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
run_stage lineage_audit
run_stage kr_ablation
run_stage holonomy_diagnostics
run_stage gauge_kernel_selection
run_stage patch07_repair
run_stage four_patch_gate
run_stage reach_round8
run_stage confirmation
run_stage meso_bridge
run_stage summary

if "$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary = json.loads(
    (root / "10_summary/retry7_next_stage_authorization.json").read_text(
        encoding="utf-8"
    )
)
raise SystemExit(0 if bool(summary.get("repaired_5k_authorized", False)) else 1)
PY
then
  exec "$V143_PIPELINE"
fi

echo "retry7_scientific_gate=failed downstream_v14_3=not_started"
exit 0
