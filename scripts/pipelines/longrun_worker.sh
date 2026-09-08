#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <state_dir>" >&2
  exit 2
fi

STATE_DIR="$1"
PID_FILE="${STATE_DIR}/current.pid"
TASK_FILE="${STATE_DIR}/current.task"
LOG_FILE_PTR="${STATE_DIR}/current.log"
CMD_FILE="${STATE_DIR}/current.command.sh"
STATUS_FILE="${STATE_DIR}/current.status"
RC_FILE="${STATE_DIR}/current.rc"
START_FILE="${STATE_DIR}/current.started_at"
END_FILE="${STATE_DIR}/current.finished_at"
HB_FILE="${STATE_DIR}/current.heartbeat"

if [[ ! -f "${CMD_FILE}" ]]; then
  echo "[longrun-worker] missing command file: ${CMD_FILE}" >&2
  exit 3
fi
if [[ ! -f "${LOG_FILE_PTR}" ]]; then
  echo "[longrun-worker] missing log pointer file: ${LOG_FILE_PTR}" >&2
  exit 3
fi

TASK_NAME="$(cat "${TASK_FILE}" 2>/dev/null || echo "unknown_task")"
LOG_FILE="$(cat "${LOG_FILE_PTR}")"
mkdir -p "$(dirname "${LOG_FILE}")"
touch "${LOG_FILE}"

date '+%F %T' > "${START_FILE}"
echo "running" > "${STATUS_FILE}"
echo "" > "${RC_FILE}"
echo $$ > "${PID_FILE}"

{
  echo "[longrun] task=${TASK_NAME}"
  echo "[longrun] start=$(cat "${START_FILE}")"
  echo "[longrun] pid=$$"
  echo "[longrun] cmd_file=${CMD_FILE}"
} | tee -a "${LOG_FILE}"

set +e
stdbuf -oL -eL bash "${CMD_FILE}" 2>&1 | while IFS= read -r line; do
  date '+%F %T' > "${HB_FILE}"
  echo "${line}"
done | tee -a "${LOG_FILE}"
RC=${PIPESTATUS[0]}
set -e

date '+%F %T' > "${END_FILE}"
echo "${RC}" > "${RC_FILE}"
if [[ ${RC} -eq 0 ]]; then
  echo "success" > "${STATUS_FILE}"
else
  echo "failed" > "${STATUS_FILE}"
fi

echo "[longrun] end=$(cat "${END_FILE}") rc=${RC}" | tee -a "${LOG_FILE}"
rm -f "${PID_FILE}"
exit "${RC}"
