#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT_DIR}"

BASE_INTERVAL_SEC="${1:-300}"
OUT_LOG="${2:-runs/maintenance/inverse_tune_watch.log}"
MIN_INTERVAL_SEC="${MIN_INTERVAL_SEC:-60}"
MAX_INTERVAL_SEC="${MAX_INTERVAL_SEC:-600}"
FAST_NEAR_END_SEC="${FAST_NEAR_END_SEC:-45}"
PID_FILE="runs/maintenance/inverse_tune_watch.pid"
FINAL_REPORT="runs/maintenance/inverse_tune_final_report.md"

mkdir -p "$(dirname "${OUT_LOG}")"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "[watch] already running pid=${old_pid} log=${OUT_LOG}"
    exit 3
  fi
fi

echo $$ > "${PID_FILE}"
trap 'rm -f "${PID_FILE}"' EXIT

last_accepted=-1
current_interval="${BASE_INTERVAL_SEC}"

finalize_report() {
  local line="$1"
  local ts status task run_root latest_attempt p95_10 p95_20 pairs10 pairs20 accepted target tried acc_rate
  ts="$(date '+%F %T')"
  status="$(echo "${line}" | sed -n 's#.* status=\([^ ]\+\) .*#\1#p')"
  task="$(echo "${line}" | sed -n 's#.* task=\([^ ]\+\) .*#\1#p')"
  run_root="$(echo "${line}" | sed -n 's#.* run_root=\([^ ]\+\) .*#\1#p')"
  latest_attempt="$(echo "${line}" | sed -n 's#.* latest_attempt=\([^ ]\+\) .*#\1#p')"
  p95_10="$(echo "${line}" | sed -n 's#.* p95_10mm=\([^ ]\+\) .*#\1#p')"
  p95_20="$(echo "${line}" | sed -n 's#.* p95_20mm=\([^ ]\+\) .*#\1#p')"
  pairs10="$(echo "${line}" | sed -n 's#.* pairs10=\([^ ]\+\) .*#\1#p')"
  pairs20="$(echo "${line}" | sed -n 's#.* pairs20=\([^ ]\+\) .*#\1#p')"
  accepted="$(echo "${line}" | sed -n 's#.* accepted=\([^/]*\)/\([^ ]*\) .*#\1#p')"
  target="$(echo "${line}" | sed -n 's#.* accepted=\([^/]*\)/\([^ ]*\) .*#\2#p')"
  tried="$(echo "${line}" | sed -n 's#.* tried=\([^ ]\+\) .*#\1#p')"
  acc_rate="$(echo "${line}" | sed -n 's#.* acc_rate=\([^ ]\+\) .*#\1#p')"

  cat > "${FINAL_REPORT}" <<EOF
# Inverse Tune 自动结论报告

- 时间: ${ts}
- 任务: ${task}
- 状态: ${status}
- 最新运行目录: ${run_root}
- 最新轮次: ${latest_attempt}
- 采样进度: ${accepted}/${target}
- tried: ${tried}
- acc_rate: ${acc_rate}
- p95(theta_rms_deg, <=10mm): ${p95_10}
- p95(theta_rms_deg, <=20mm): ${p95_20}
- pairs<=10mm: ${pairs10}
- pairs<=20mm: ${pairs20}

> 该报告由 watcher 在任务状态变化时自动生成。
EOF
}

while true; do
  line="$(bash scripts/pipelines/report_inverse_tune_progress.sh)"
  echo "${line}" | tee -a "${OUT_LOG}"

  status="$(echo "${line}" | sed -n 's#.* status=\([^ ]\+\) .*#\1#p')"
  accepted="$(echo "${line}" | sed -n 's#.* accepted=\([^/]*\)/\([^ ]*\) .*#\1#p')"
  target="$(echo "${line}" | sed -n 's#.* accepted=\([^/]*\)/\([^ ]*\) .*#\2#p')"
  started_at="$(echo "${line}" | sed -n 's#.* started_at=\([^ ]\+ [^ ]\+\) accepted=.*#\1#p')"

  if [[ "${status}" != "running" ]]; then
    finalize_report "${line}"
    echo "[watch] task status=${status}, watcher exit" | tee -a "${OUT_LOG}"
    break
  fi

  delta="na"
  decision_reason="base_interval"
  eta_hms="na"

  if [[ -n "${accepted}" && "${accepted}" =~ ^[0-9]+$ ]]; then
    if (( last_accepted >= 0 )); then
      delta=$((accepted - last_accepted))
      if (( delta <= 0 )); then
        current_interval=$(( BASE_INTERVAL_SEC / 2 ))
        decision_reason="progress_stall"
      elif (( delta >= 8 )); then
        current_interval="${BASE_INTERVAL_SEC}"
        decision_reason="progress_good"
      else
        current_interval=$(( BASE_INTERVAL_SEC * 2 / 3 ))
        decision_reason="progress_slow"
      fi
    else
      current_interval="${BASE_INTERVAL_SEC}"
      decision_reason="first_tick"
    fi
    last_accepted="${accepted}"
  else
    current_interval=$(( BASE_INTERVAL_SEC / 2 ))
    decision_reason="accepted_parse_fail"
  fi

  if [[ -n "${target}" && "${target}" =~ ^[0-9]+$ && "${target}" -gt 0 && -n "${accepted}" && "${accepted}" =~ ^[0-9]+$ ]]; then
    pct=$(( accepted * 100 / target ))
    if (( pct >= 90 )); then
      current_interval="${FAST_NEAR_END_SEC}"
      decision_reason="near_finish"
    fi
  fi

  if (( current_interval < MIN_INTERVAL_SEC )); then
    current_interval="${MIN_INTERVAL_SEC}"
  fi
  if (( current_interval > MAX_INTERVAL_SEC )); then
    current_interval="${MAX_INTERVAL_SEC}"
  fi

  if [[ -n "${started_at}" && -n "${accepted}" && "${accepted}" =~ ^[0-9]+$ && -n "${target}" && "${target}" =~ ^[0-9]+$ ]]; then
    now_epoch="$(date +%s)"
    start_epoch="$(date -d "${started_at}" +%s 2>/dev/null || true)"
    if [[ -n "${start_epoch}" && "${start_epoch}" =~ ^[0-9]+$ && "${now_epoch}" -gt "${start_epoch}" && "${accepted}" -gt 0 ]]; then
      elapsed=$(( now_epoch - start_epoch ))
      remain=$(( target - accepted ))
      if (( remain < 0 )); then
        remain=0
      fi
      eta_sec="$(awk -v a="${accepted}" -v e="${elapsed}" -v r="${remain}" 'BEGIN{if(a<=0||e<=0){print -1}else{printf "%.0f", r*(e/a)}}')"
      if [[ "${eta_sec}" =~ ^-?[0-9]+$ ]] && (( eta_sec >= 0 )); then
        eta_hms="$(printf '%02d:%02d:%02d' $((eta_sec/3600)) $(((eta_sec%3600)/60)) $((eta_sec%60)))"
      fi
    fi
  fi

  echo "[watch] decision=${decision_reason} delta_acc=${delta} eta=${eta_hms} next_report_in=${current_interval}s" | tee -a "${OUT_LOG}"
  sleep "${current_interval}"
done
