#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python
CONFIG="$SOURCE_ROOT/configs/bacra_v14_3r_retry13_zero_rooted_connected_roadmap.yaml"
OUTPUT_ROOT=${1:-/mnt/ML_projects/quasi_exp/runs/bacra_v14_3r_retry13_connected_roadmap_attempt1}
BINDING_SHA=${2:-${BACRA_RETRY13_BINDING_SHA:-}}
RUNNER="$SOURCE_ROOT/scripts/analysis/run_bacra_v14_3r_retry13.py"

if [[ -z "$BINDING_SHA" ]]; then
  echo "retry13 launcher requires binding SHA as argument 2 or BACRA_RETRY13_BINDING_SHA" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export PYTHONHASHSEED=20260883
export MPLCONFIGDIR=/tmp/bacra_retry13_matplotlib_cache

stages=(
  input_lineage_audit
  connected_backbone
  core_axis_spokes
  connected_fill
  orbit_expansion_prune_split
  three_graph_audit
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
