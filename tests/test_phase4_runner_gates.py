from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "pipelines" / "run_phase4_single_branch_experiment.py"
    spec = importlib.util.spec_from_file_location("run_phase4_single_branch_experiment", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_phase4_runner_stops_before_relabel_when_theta_continuity_is_bad() -> None:
    mod = _load_module()
    gate = {
        "hard_gate_passed": True,
        "all10_theta_p95_deg": 7.1,
        "all10_tension_p95_n": 80.0,
        "beta_close_tension_p95_n": 40.0,
    }

    assert mod._theta_continuity_stop_failed(gate, stop_deg=5.0)
    assert not mod._training_acceptance_passed(
        gate,
        theta_target_deg=3.0,
        tension_target_n=100.0,
        beta_close_tension_target_n=50.0,
    )


def test_phase4_runner_accepts_only_full_selected_dataset_gate() -> None:
    mod = _load_module()
    gate = {
        "hard_gate_passed": True,
        "all10_theta_p95_deg": 2.9,
        "all10_tension_p95_n": 95.0,
        "beta_close_tension_p95_n": 49.0,
    }

    assert not mod._theta_continuity_stop_failed(gate, stop_deg=5.0)
    assert mod._training_acceptance_passed(
        gate,
        theta_target_deg=3.0,
        tension_target_n=100.0,
        beta_close_tension_target_n=50.0,
    )
