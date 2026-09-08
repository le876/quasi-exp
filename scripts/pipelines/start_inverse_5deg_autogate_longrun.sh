#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT_DIR}"

ENV_NAME="${ENV_NAME:-quasi_exp}"
CFG_2K="${CFG_2K:-configs/robot_rods_only_5deg_inverse_2k_paper_tf_cpu.yaml}"
CFG_100K="${CFG_100K:-configs/robot_rods_only_5deg_inverse_100k_paper_tf_cpu.yaml}"
OUT_2K_DIR="${OUT_2K_DIR:-data/accept_inverse_cont_2k_5deg}"
OUT_100K_DIR="${OUT_100K_DIR:-data/accept_inverse_cont_100k_5deg}"
REPORT_JSON="${REPORT_JSON:-runs/acceptance/inverse_5deg_2k_continuity_gate.json}"

TASK_NAME="${TASK_NAME:-inverse_5deg_autogate}"
LOG_FILE="${LOG_FILE:-runs/maintenance/${TASK_NAME}.log}"

scripts/pipelines/longrun_tmux.sh start "${TASK_NAME}" "${LOG_FILE}" -- \
  env \
    ENV_NAME="${ENV_NAME}" \
    CFG_2K="${CFG_2K}" \
    CFG_100K="${CFG_100K}" \
    OUT_2K_DIR="${OUT_2K_DIR}" \
    OUT_100K_DIR="${OUT_100K_DIR}" \
    REPORT_JSON="${REPORT_JSON}" \
    bash scripts/pipelines/run_inverse_5deg_autogate.sh

echo ""
echo "[hint] monitor status:"
echo "  bash scripts/pipelines/longrun_tmux.sh status"
echo "[hint] tail logs:"
echo "  bash scripts/pipelines/longrun_tmux.sh logs 120"
