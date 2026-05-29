#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT_DIR}"

status_text="$(bash scripts/pipelines/longrun_tmux.sh status)"
task="$(echo "${status_text}" | awk -F': ' '/^task:/{print $2}')"
status="$(echo "${status_text}" | awk -F': ' '/^status:/{print $2}')"
started_at="$(echo "${status_text}" | awk -F': ' '/^started_at:/{print $2}')"
log_path="$(echo "${status_text}" | awk -F': ' '/^log:/{print $2}')"

run_root="$(ls -1dt runs/acceptance/inverse_5deg_tune_* 2>/dev/null | head -n1 || true)"
attempt=""
accepted=""
target=""
tried=""
acc_rate=""
p95_10=""
p95_20=""
pairs_10=""
pairs_20=""

if [[ -n "${run_root}" ]]; then
  attempt_log="$(ls -1 "${run_root}"/attempt_*.log 2>/dev/null | sort | tail -n1 || true)"
  if [[ -n "${attempt_log}" && -f "${attempt_log}" ]]; then
    progress_line="$(tr '\r' '\n' < "${attempt_log}" | rg 'accepted:' | tail -n1 || true)"
    if [[ -n "${progress_line}" ]]; then
      pair="$(echo "${progress_line}" | sed -n 's#.* \([0-9]\+\)/\([0-9]\+\) .*#\1/\2#p' | tail -n1)"
      if [[ -n "${pair}" ]]; then
        accepted="${pair%/*}"
        target="${pair#*/}"
      fi
      tried="$(echo "${progress_line}" | sed -n 's#.*tried=\([0-9]\+\).*#\1#p' | tail -n1)"
      acc_rate="$(echo "${progress_line}" | sed -n 's#.*acc_rate=\([0-9.]\+\).*#\1#p' | tail -n1)"
    fi
  fi

  summary_md="${run_root}/summary.md"
  if [[ -f "${summary_md}" ]]; then
    summary_row="$(awk -F'|' 'NF>=10 && $2 ~ /^[[:space:]]*[0-9]+[[:space:]]*$/ {row=$0} END{print row}' "${summary_md}")"
    if [[ -n "${summary_row}" ]]; then
      attempt="$(echo "${summary_row}" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/,"",$2); print $2}')"
      p95_10="$(echo "${summary_row}" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/,"",$5); print $5}')"
      p95_20="$(echo "${summary_row}" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/,"",$6); print $6}')"
      pairs_10="$(echo "${summary_row}" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/,"",$7); print $7}')"
      pairs_20="$(echo "${summary_row}" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/,"",$8); print $8}')"
    fi
  fi
fi

ts="$(date '+%F %T')"
echo "[${ts}] task=${task} status=${status} started_at=${started_at} accepted=${accepted:-na}/${target:-na} tried=${tried:-na} acc_rate=${acc_rate:-na} latest_attempt=${attempt:-na} p95_10mm=${p95_10:-na} p95_20mm=${p95_20:-na} pairs10=${pairs_10:-na} pairs20=${pairs_20:-na} run_root=${run_root:-na} log=${log_path:-na}"
