#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python
CONFIG="$SOURCE_ROOT/configs/bacra_v14_3r_retry15_omega600_domain_discovery.yaml"
OUTPUT_ROOT=${1:-/mnt/ML_projects/quasi_exp/runs/bacra_v14_3r_retry15_omega600_domain_discovery_attempt1}
BINDING_SHA=${2:-${BACRA_RETRY15_DISCOVERY_BINDING_SHA:-}}
RUNNER="$SOURCE_ROOT/scripts/analysis/run_bacra_v14_3r_retry15_discovery.py"

if [[ -z "$BINDING_SHA" ]]; then
  echo "retry15 discovery requires binding SHA as argument 2 or BACRA_RETRY15_DISCOVERY_BINDING_SHA" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

exec "$PYTHON_BIN" "$RUNNER" \
  --config "$CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --binding-sha "$BINDING_SHA"
