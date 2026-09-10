from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "pipelines" / "longrun_tmux.sh"


def _write_state(state_dir: Path, *, status: str, rc: int | None, pid: int | None = None) -> None:
    state_dir.mkdir()
    (state_dir / "current.status").write_text(f"{status}\n", encoding="utf-8")
    if rc is not None:
        (state_dir / "current.rc").write_text(f"{rc}\n", encoding="utf-8")
    if pid is not None:
        (state_dir / "current.pid").write_text(f"{pid}\n", encoding="utf-8")


def _run_wait(state_dir: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["TMUX_LONGRUN_STATE_DIR"] = str(state_dir)
    return subprocess.run(
        ["bash", str(LAUNCHER), "wait"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )


def test_wait_returns_recorded_success_exit_code_without_mutating_state(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    _write_state(state_dir, status="success", rc=0)
    before = {path.name: path.read_bytes() for path in state_dir.iterdir()}

    result = _run_wait(state_dir)

    after = {path.name: path.read_bytes() for path in state_dir.iterdir()}
    assert result.returncode == 0
    assert "terminal status=success rc=0" in result.stdout
    assert before == after


def test_wait_returns_recorded_failed_exit_code(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    _write_state(state_dir, status="failed", rc=23)

    result = _run_wait(state_dir)

    assert result.returncode == 23
    assert "terminal status=failed rc=23" in result.stdout


def test_monitor_timeout_leaves_independent_worker_and_state_intact(tmp_path: Path) -> None:
    # A detached launcher worker is outside the foreground wait's process group.
    worker = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        state_dir = tmp_path / "state"
        _write_state(state_dir, status="running", rc=None, pid=worker.pid)
        before = {path.name: path.read_bytes() for path in state_dir.iterdir()}
        env = os.environ.copy()
        env["TMUX_LONGRUN_STATE_DIR"] = str(state_dir)
        result = subprocess.run(
            ["timeout", "--signal=TERM", "--kill-after=1s", "0.2s", "bash", str(LAUNCHER), "wait"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=3,
        )
        assert result.returncode == 124
        assert worker.poll() is None
        assert {path.name: path.read_bytes() for path in state_dir.iterdir()} == before
    finally:
        worker.terminate()
        worker.wait(timeout=3)


def test_wait_treats_stopped_by_user_as_failure(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    _write_state(state_dir, status="stopped_by_user", rc=130)

    result = _run_wait(state_dir)

    assert result.returncode == 130
    assert "terminal status=stopped_by_user rc=130" in result.stdout


def test_wait_fails_closed_for_dead_pid_with_unknown_status(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    _write_state(state_dir, status="unknown_after_crash", rc=None, pid=999_999_999)

    result = _run_wait(state_dir)

    assert result.returncode == 6
    assert "no live worker for non-terminal status" in result.stderr
    assert "status=unknown_after_crash" in result.stderr


def test_start_wait_chain_observes_the_new_worker_state(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_tmux = fake_bin / "tmux"
    fake_tmux.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case "$1" in
  has-session)
    exit 0
    ;;
  list-windows)
    echo worker
    ;;
  set-option)
    exit 0
    ;;
  respawn-pane)
    command="${!#}"
    bash -c "${command}" &
    ;;
  *)
    echo "unexpected fake tmux command: $*" >&2
    exit 99
    ;;
esac
""",
        encoding="utf-8",
    )
    fake_tmux.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["TMUX_LONGRUN_STATE_DIR"] = str(state_dir)
    env["TMUX_LONGRUN_SESSION"] = "test_longrun"
    log_file = tmp_path / "task.log"

    started = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "start",
            "quick_success",
            str(log_file),
            "--",
            "bash",
            "-c",
            "exit 0",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    waited = subprocess.run(
        ["bash", str(LAUNCHER), "wait"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )

    assert started.returncode == 0, started.stderr
    assert waited.returncode == 0, waited.stderr
    assert (state_dir / "current.status").read_text(encoding="utf-8").strip() == "success"
    assert (state_dir / "current.rc").read_text(encoding="utf-8").strip() == "0"


def test_start_accepts_tmux_pane_pid_when_host_pid_is_not_namespace_visible(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_tmux = fake_bin / "tmux"
    fake_tmux.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case "$1" in
  has-session|set-option)
    exit 0
    ;;
  list-windows)
    echo worker
    ;;
  respawn-pane)
    mkdir -p "${TMUX_LONGRUN_STATE_DIR}"
    echo running > "${TMUX_LONGRUN_STATE_DIR}/current.status"
    echo 999999999 > "${TMUX_LONGRUN_STATE_DIR}/current.pid"
    ;;
  display-message)
    echo "999999999 0"
    ;;
  *)
    echo "unexpected fake tmux command: $*" >&2
    exit 99
    ;;
esac
""",
        encoding="utf-8",
    )
    fake_tmux.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["TMUX_LONGRUN_STATE_DIR"] = str(state_dir)
    env["TMUX_LONGRUN_SESSION"] = "test_longrun"

    result = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "start",
            "namespace_hidden_pid",
            str(tmp_path / "task.log"),
            "--",
            "bash",
            "-c",
            "exit 0",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "[longrun] started" in result.stdout
