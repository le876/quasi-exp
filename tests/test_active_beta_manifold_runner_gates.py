from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts" / "pipelines" / "run_active_beta_manifold_overnight.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_active_beta_manifold_overnight", PIPELINE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_strict_gate_blocks_20k_when_theta_continuity_fails():
    mod = load_module()
    gate = {
        "hard_gate_passed": True,
        "all10_theta_p95_deg": 5.1,
        "all10_tension_p95_n": 80.0,
        "beta_close_tension_p95_n": 40.0,
        "multi_branch_ball_ratio": 0.1,
    }
    assert mod.strict_gate_passed(gate, all10_pairs=200, min_all10_pairs=100) is False


def test_strict_gate_requires_enough_workspace_neighbor_pairs():
    mod = load_module()
    gate = {
        "hard_gate_passed": True,
        "all10_theta_p95_deg": 2.0,
        "all10_tension_p95_n": 80.0,
        "beta_close_tension_p95_n": 40.0,
        "multi_branch_ball_ratio": 0.1,
    }
    assert mod.strict_gate_passed(gate, all10_pairs=10, min_all10_pairs=100) is False


def test_select_best_variant_prefers_passing_low_score_variant():
    mod = load_module()
    reports = [
        {
            "variant": "bad",
            "gate": {"hard_gate_passed": True, "all10_theta_p95_deg": 4.0, "all10_tension_p95_n": 80.0, "beta_close_tension_p95_n": 40.0, "multi_branch_ball_ratio": 0.1},
            "diagnostics": {"continuity": {"groups": {"all_xyz_<=10mm": {"pairs": 200}}}},
        },
        {
            "variant": "good_b",
            "gate": {"hard_gate_passed": True, "all10_theta_p95_deg": 2.5, "all10_tension_p95_n": 95.0, "beta_close_tension_p95_n": 45.0, "multi_branch_ball_ratio": 0.2},
            "diagnostics": {"continuity": {"groups": {"all_xyz_<=10mm": {"pairs": 200}}}},
        },
        {
            "variant": "good_a",
            "gate": {"hard_gate_passed": True, "all10_theta_p95_deg": 1.5, "all10_tension_p95_n": 90.0, "beta_close_tension_p95_n": 40.0, "multi_branch_ball_ratio": 0.1},
            "diagnostics": {"continuity": {"groups": {"all_xyz_<=10mm": {"pairs": 200}}}},
        },
    ]

    selected = mod.select_best_passing_variant(reports, min_all10_pairs=100)

    assert selected["variant"] == "good_a"
