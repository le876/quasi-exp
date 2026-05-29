#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
STATE_DIR="${ROOT_DIR}/runs/maintenance/longrun"
SESSION_NAME="${TMUX_LONGRUN_SESSION:-quasi_exp_longrun}"
WINDOW_NAME="${TMUX_LONGRUN_WINDOW:-worker}"
WORKER_SCRIPT="${ROOT_DIR}/scripts/pipelines/longrun_worker.sh"

TASK_FILE="${STATE_DIR}/current.task"
LOG_FILE_PTR="${STATE_DIR}/current.log"
CMD_FILE="${STATE_DIR}/current.command.sh"
CMD_TEXT_FILE="${STATE_DIR}/current.command.txt"
STATUS_FILE="${STATE_DIR}/current.status"
PID_FILE="${STATE_DIR}/current.pid"
RC_FILE="${STATE_DIR}/current.rc"
START_FILE="${STATE_DIR}/current.started_at"
END_FILE="${STATE_DIR}/current.finished_at"
HB_FILE="${STATE_DIR}/current.heartbeat"

usage() {
  cat <<'EOF'
Usage:
  longrun_tmux.sh start <task_name> <log_file> -- <command...>
  longrun_tmux.sh status
  longrun_tmux.sh logs [lines]
  longrun_tmux.sh attach
  longrun_tmux.sh stop
  longrun_tmux.sh reset
EOF
}

ensure_session() {
  if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    tmux new-session -d -s "${SESSION_NAME}" -n "${WINDOW_NAME}" "bash"
    tmux set-option -t "${SESSION_NAME}" remain-on-exit on >/dev/null
  fi
  if ! tmux list-windows -t "${SESSION_NAME}" -F '#W' | rg -x "${WINDOW_NAME}" >/dev/null 2>&1; then
    tmux new-window -t "${SESSION_NAME}" -n "${WINDOW_NAME}" "bash"
  fi
  tmux set-option -t "${SESSION_NAME}:${WINDOW_NAME}" remain-on-exit on >/dev/null
}

read_file() {
  local f="$1"
  if [[ -f "${f}" ]]; then
    cat "${f}"
  fi
}

pid_running() {
  if [[ ! -f "${PID_FILE}" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "${PID_FILE}")"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

prepare_state() {
  mkdir -p "${STATE_DIR}"
  chmod +x "${WORKER_SCRIPT}"
}

render_command() {
  local out=""
  local token
  for token in "$@"; do
    out+=$(printf '%q ' "${token}")
  done
  printf '%s' "${out}"
}

cmd_start() {
  if [[ $# -lt 4 ]]; then
    usage
    exit 2
  fi
  local task_name="$1"
  local log_file="$2"
  shift 2
  if [[ "$1" != "--" ]]; then
    echo "missing -- before command" >&2
    exit 2
  fi
  shift
  if [[ $# -lt 1 ]]; then
    echo "empty command" >&2
    exit 2
  fi

  prepare_state
  ensure_session

  if pid_running; then
    echo "[longrun] task is already running: $(read_file "${TASK_FILE}") (pid=$(read_file "${PID_FILE}"))"
    echo "[longrun] use: $0 status"
    exit 3
  fi

  local rendered
  rendered="$(render_command "$@")"
  local abs_log
  if [[ "${log_file}" = /* ]]; then
    abs_log="${log_file}"
  else
    abs_log="${ROOT_DIR}/${log_file}"
  fi
  mkdir -p "$(dirname "${abs_log}")"
  : > "${abs_log}"

  printf '%s\n' "${task_name}" > "${TASK_FILE}"
  printf '%s\n' "${abs_log}" > "${LOG_FILE_PTR}"
  printf '%s\n' "${rendered}" > "${CMD_TEXT_FILE}"
  cat > "${CMD_FILE}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${ROOT_DIR@Q}
${rendered}
EOF
  chmod +x "${CMD_FILE}"

  tmux respawn-pane -k -t "${SESSION_NAME}:${WINDOW_NAME}" "bash ${WORKER_SCRIPT@Q} ${STATE_DIR@Q}"

  echo "[longrun] started"
  echo "  session=${SESSION_NAME}"
  echo "  window=${WINDOW_NAME}"
  echo "  task=${task_name}"
  echo "  log=${abs_log}"
}

cmd_status() {
  prepare_state
  local session_state="missing"
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    session_state="ready"
  fi

  local task status pid rc started ended hb cmd_text log_path
  task="$(read_file "${TASK_FILE}")"
  status="$(read_file "${STATUS_FILE}")"
  pid="$(read_file "${PID_FILE}")"
  rc="$(read_file "${RC_FILE}")"
  started="$(read_file "${START_FILE}")"
  ended="$(read_file "${END_FILE}")"
  hb="$(read_file "${HB_FILE}")"
  cmd_text="$(read_file "${CMD_TEXT_FILE}")"
  log_path="$(read_file "${LOG_FILE_PTR}")"

  if pid_running; then
    status="running"
  elif [[ -z "${status}" ]]; then
    status="idle"
  fi

  cat <<EOF
session: ${SESSION_NAME}
session_state: ${session_state}
window: ${WINDOW_NAME}
task: ${task:-none}
status: ${status}
pid: ${pid:-none}
rc: ${rc:-none}
started_at: ${started:-none}
finished_at: ${ended:-none}
heartbeat: ${hb:-none}
log: ${log_path:-none}
command: ${cmd_text:-none}
EOF
}

cmd_logs() {
  prepare_state
  local lines="${1:-80}"
  local log_path
  log_path="$(read_file "${LOG_FILE_PTR}")"
  if [[ -z "${log_path}" || ! -f "${log_path}" ]]; then
    echo "[longrun] no log file found"
    exit 4
  fi
  tail -n "${lines}" "${log_path}"
}

cmd_attach() {
  ensure_session
  exec tmux attach -t "${SESSION_NAME}"
}

cmd_stop() {
  prepare_state
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    tmux send-keys -t "${SESSION_NAME}:${WINDOW_NAME}" C-c || true
  fi
  if pid_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    kill -TERM "${pid}" 2>/dev/null || true
    sleep 2
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  fi
  echo "stopped_by_user" > "${STATUS_FILE}"
  date '+%F %T' > "${END_FILE}"
  rm -f "${PID_FILE}"
  echo "[longrun] stop sent"
}

cmd_reset() {
  prepare_state
  if pid_running; then
    echo "[longrun] task is running, stop it first"
    exit 5
  fi
  rm -f "${TASK_FILE}" "${LOG_FILE_PTR}" "${CMD_FILE}" "${CMD_TEXT_FILE}" \
    "${STATUS_FILE}" "${PID_FILE}" "${RC_FILE}" "${START_FILE}" "${END_FILE}" "${HB_FILE}"
  echo "[longrun] state reset"
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 2
  fi
  local cmd="$1"
  shift
  case "${cmd}" in
    start) cmd_start "$@" ;;
    status) cmd_status ;;
    logs) cmd_logs "$@" ;;
    attach) cmd_attach ;;
    stop) cmd_stop ;;
    reset) cmd_reset ;;
    *) usage; exit 2 ;;
  esac
}

main "$@"
