#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="$(cd "$SOURCE_ROOT/../.." && pwd)"
if [[ "$(basename "$(dirname "$SOURCE_ROOT")")" != ".worktrees" ]]; then
  PROJECT_ROOT="$SOURCE_ROOT"
fi
OUTPUT_ROOT="${1:-$PROJECT_ROOT/runs/bacra_v14_3r_retry9_coverage_first_attempt1}"
PYTHON=/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python
export CUDA_VISIBLE_DEVICES=-1

cd "$SOURCE_ROOT"
exec "$PYTHON" scripts/analysis/run_bacra_v14_3r_retry9.py \
  --config configs/bacra_v14_3r_retry9_stage0_3.yaml \
  --output-root "$OUTPUT_ROOT"
