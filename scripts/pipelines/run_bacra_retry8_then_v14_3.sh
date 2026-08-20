#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${SOURCE_ROOT%%/.worktrees/*}
PYTHON_BIN=/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python
RETRY8="$SOURCE_ROOT/scripts/pipelines/run_bacra_v14_2r_retry8_partial_relay.sh"
SMOKE="$SOURCE_ROOT/scripts/pipelines/run_bacra_v14_3_retry8_smoke.sh"
FIVE_K="$SOURCE_ROOT/scripts/pipelines/run_bacra_v14_3_retry8_5k.sh"
SMOKE_AUTH="$PROJECT_ROOT/runs/bacra_v14_3_retry8_balanced_smoke_retry2/09_summary/retry8_next_stage_authorization.json"

"$RETRY8"
"$SMOKE"

if "$PYTHON_BIN" - "$SMOKE_AUTH" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(
    0 if bool(payload.get("five_k_teacher_execution_authorized", False)) else 1
)
PY
then
  exec "$FIVE_K"
fi

echo "retry8_smoke_complete five_k_teacher_execution_authorized=false"
exit 0
