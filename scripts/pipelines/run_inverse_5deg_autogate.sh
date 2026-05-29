#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-quasi_exp}"
CFG_2K="${CFG_2K:-configs/robot_rods_only_5deg_inverse_2k_paper_tf_cpu.yaml}"
CFG_100K="${CFG_100K:-configs/robot_rods_only_5deg_inverse_100k_paper_tf_cpu.yaml}"
OUT_2K_DIR="${OUT_2K_DIR:-data/accept_inverse_cont_2k_5deg}"
OUT_100K_DIR="${OUT_100K_DIR:-data/accept_inverse_cont_100k_5deg}"
REPORT_JSON="${REPORT_JSON:-runs/acceptance/inverse_5deg_2k_continuity_gate.json}"

echo "[pipeline] start $(date '+%F %T')"
echo "[pipeline] env=$ENV_NAME"
echo "[pipeline] cfg2k=$CFG_2K"
echo "[pipeline] cfg100k=$CFG_100K"

rm -rf "$OUT_2K_DIR"
mkdir -p "$OUT_2K_DIR"

echo "[pipeline] generate 2k inverse dataset..."
conda run -n "$ENV_NAME" python scripts/generate_dataset.py \
  --config "$CFG_2K" \
  --num-samples 2000 \
  --max-tried 15000

echo "[pipeline] evaluate continuity..."
set +e
conda run -n "$ENV_NAME" python scripts/analysis/eval_local_continuity.py \
  --dataset "$OUT_2K_DIR/dataset.parquet" \
  --out-json "$REPORT_JSON"
gate_exit=$?
set -e

if [[ $gate_exit -ne 0 ]]; then
  echo "[pipeline] continuity gate FAILED (exit=$gate_exit). stop."
  exit $gate_exit
fi

echo "[pipeline] continuity gate PASSED. start 100k generation..."
rm -rf "$OUT_100K_DIR"
mkdir -p "$OUT_100K_DIR"
conda run -n "$ENV_NAME" python scripts/generate_dataset.py \
  --config "$CFG_100K" \
  --num-samples 100000 \
  --max-tried 800000

echo "[pipeline] done $(date '+%F %T')"
