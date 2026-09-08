#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python
CONFIG="$SOURCE_ROOT/configs/bacra_v14_3r_retry10_frontier_expanded_atlas.yaml"
OUTPUT_ROOT=${1:-/mnt/ML_projects/quasi_exp/runs/bacra_v14_3r_retry10_frontier_expanded_atlas_attempt1}
RUNNER="$SOURCE_ROOT/scripts/analysis/run_bacra_v14_3r_retry10.py"

export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export PYTHONHASHSEED=20260881
export MPLCONFIGDIR=/tmp/bacra_retry10_matplotlib_cache

stages=(
  inventory
  atlas_poc
  seed_datasets
  seed_students
  atlas_expansion
  taxonomy_rescue
  atlas_freeze
  full_diagnostic
  preflight_50k
  dataset_50k
  student_50k
  tension_pilot
  preflight_100k
  dataset_100k
  student_100k
  preflight_200k
  dataset_200k
  student_200k
  tension_materialization
  summary
)

cd "$SOURCE_ROOT"
for stage in "${stages[@]}"; do
  if "$PYTHON_BIN" "$RUNNER" \
    --config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --validate-stage "$stage" >/dev/null 2>&1; then
    continue
  fi
  "$PYTHON_BIN" "$RUNNER" \
    --config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --stage "$stage"
done
