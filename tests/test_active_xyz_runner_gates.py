from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "generate_active_single_branch_dataset.py"
    spec = importlib.util.spec_from_file_location("generate_active_single_branch_dataset", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_active_gate_stops_before_relabel_when_theta_continuity_fails() -> None:
    mod = _load_module()
    gate = {
        "hard_gate_passed": True,
        "xyz_err_m_p95": 0.003,
        "all10_theta_p95_deg": 5.5,
        "all10_tension_p95_n": 80.0,
        "beta_close_tension_p95_n": 40.0,
        "multi_branch_ball_ratio": 0.1,
    }

    assert mod.active_acceptance_gate_passed(gate) is False
    assert mod.should_stop_before_relabel(gate) is True


def test_active_gate_passes_only_when_all_thresholds_are_met() -> None:
    mod = _load_module()
    gate = {
        "hard_gate_passed": True,
        "xyz_err_m_p95": 0.003,
        "all10_theta_p95_deg": 2.8,
        "all10_tension_p95_n": 95.0,
        "beta_close_tension_p95_n": 48.0,
        "multi_branch_ball_ratio": 0.2,
    }

    assert mod.active_acceptance_gate_passed(gate) is True
    assert mod.should_stop_before_relabel(gate) is False
