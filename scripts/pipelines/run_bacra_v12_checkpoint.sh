#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LONGRUN="${ROOT_DIR}/scripts/pipelines/longrun_tmux.sh"
STATE_DIR="${ROOT_DIR}/runs/maintenance/longrun"

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <task_name> <log_file> -- <command...>" >&2
  exit 2
fi

"${LONGRUN}" start "$@"

# The caller (normally a fresh long_wait_monitor) keeps this terminal and
# waits in 300-second increments.  The detached tmux process survives terminal
# loss, while this wrapper prevents the monitor from returning prematurely.
while true; do
  status="$(test -f "${STATE_DIR}/current.status" && tr -d '\r\n' < "${STATE_DIR}/current.status" || true)"
  pid="$(test -f "${STATE_DIR}/current.pid" && tr -d '\r\n' < "${STATE_DIR}/current.pid" || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    sleep 300
    continue
  fi
  case "${status}" in
    success|completed)
      rc="$(test -f "${STATE_DIR}/current.rc" && tr -d '\r\n' < "${STATE_DIR}/current.rc" || echo 1)"
      exit "${rc}"
      ;;
    failed|stopped_by_user)
      rc="$(test -f "${STATE_DIR}/current.rc" && tr -d '\r\n' < "${STATE_DIR}/current.rc" || echo 1)"
      if [[ "${rc}" = "0" ]]; then
        rc=1
      fi
      exit "${rc}"
      ;;
  esac
  sleep 300
done
