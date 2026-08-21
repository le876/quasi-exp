from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_utils():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "true_ellipse_atlas_utils.py"
    spec = importlib.util.spec_from_file_location("true_ellipse_atlas_utils", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_sobol_reachability_pool_stays_in_beta_bounds_and_maps_theta() -> None:
    mod = _load_utils()
    bounds = mod.beta_bounds_rad("current")

    beta = mod.sobol_beta_samples(16, bounds=bounds, seed=7)
    theta = mod.theta_from_beta_batch(beta, theta_sign=-1.0)

    assert beta.shape == (16, 6)
    assert theta.shape == (16, 30)
    assert np.all(beta >= bounds[:, 0] - 1.0e-12)
    assert np.all(beta <= bounds[:, 1] + 1.0e-12)
    assert np.allclose(theta[:, 0], -beta[:, 0])
    assert np.allclose(theta[:, 1], -beta[:, 1])
    assert np.allclose(theta[:, 20], -beta[:, 4])
    assert np.allclose(theta[:, 21], -beta[:, 5])


def test_true_ellipse_geometry_rejects_rank1_same_phase() -> None:
    mod = _load_utils()

    ellipse = mod.make_axis_phase_ellipse(
        candidate_id="rank2",
        center=(1.1, 0.02, 0.03),
        amp_xy_mm=50.0,
        phase_y_rad=0.0,
        phase_z_rad=math.pi / 2.0,
        n_points=72,
    )
    same_phase = mod.make_axis_phase_ellipse(
        candidate_id="rank1",
        center=(1.1, 0.02, 0.03),
        amp_xy_mm=50.0,
        phase_y_rad=0.0,
        phase_z_rad=0.0,
        n_points=72,
    )

    rank2 = mod.trajectory_geometry_metrics(ellipse[mod.TARGET_XYZ_COLS].to_numpy())
    rank1 = mod.trajectory_geometry_metrics(same_phase[mod.TARGET_XYZ_COLS].to_numpy())

    assert rank2["rank2_gate_pass"] is True
    assert rank2["svd2_over_svd1"] > 0.3
    assert rank1["rank2_gate_pass"] is False
    assert rank1["svd2_over_svd1"] < 1.0e-6


def test_pointwise_ik_recovers_known_beta_target_with_small_residual() -> None:
    mod = _load_utils()

    true_beta = np.deg2rad(np.array([1.0, -1.5, 2.0, -2.5, 7.0, -6.0]))
    lengths_m = np.full(31, 0.04, dtype=float)
    p_end_local_m = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
    target = mod.fk_from_beta_batch(
        true_beta.reshape(1, 6),
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=-1.0,
    )[0]
    result = mod.solve_beta_ik(
        target,
        init_betas=np.vstack([np.zeros(6), true_beta + np.deg2rad(0.2)]),
        bounds=mod.beta_bounds_rad("current"),
        lengths_m=lengths_m,
        p_end_local_m=p_end_local_m,
        theta_sign=-1.0,
        max_nfev=120,
    )

    assert result.success
    assert result.residual_mm <= 2.0
    assert np.all(np.isfinite(result.beta_rad))


def test_cluster_and_cyclic_linking_choose_smooth_closed_branch() -> None:
    mod = _load_utils()

    rows = []
    smooth = np.deg2rad(np.array([[0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 2.2, 0], [0, 0, 0, 0, 2.1, 0]]))
    jump = np.deg2rad(np.array([[0, 0, 0, 0, -8, 0], [0, 0, 0, 0, 8, 0], [0, 0, 0, 0, -8, 0]]))
    for angle_idx in range(3):
        for candidate_id, beta in enumerate((smooth[angle_idx], jump[angle_idx])):
            row = {
                "candidate_id": candidate_id,
                "angle_idx": angle_idx,
                "xyz_residual_mm": 0.1,
                "kappa": 10.0,
            }
            for i, col in enumerate(mod.BETA_COLS):
                row[col] = beta[i]
            rows.append(row)
    candidates = pd.DataFrame(rows)

    selected, report = mod.link_cyclic_branch(candidates, max_edge_deg=3.0, closure_weight=2.0)

    assert len(selected) == 3
    assert report["zero_reconfiguration"] is True
    assert report["seam_beta_rms_deg"] <= 1.0
    assert selected["candidate_id"].tolist() == [0, 0, 0]


def test_tube_target_generation_uses_5x5_normal_grid() -> None:
    mod = _load_utils()
    centerline = pd.DataFrame(
        {
            "candidate_id": ["c"] * 4,
            "angle_idx": [0, 1, 2, 3],
            "x_target_m": [1.0, 1.01, 1.02, 1.03],
            "y_target_m": [0.0, 0.01, 0.0, -0.01],
            "z_target_m": [0.0, 0.01, 0.02, 0.01],
        }
    )

    tube = mod.make_normal_tube_targets(centerline, offsets_mm=[-5.0, -2.5, 0.0, 2.5, 5.0])

    assert len(tube) == 4 * 25
    assert {"delta_n1_mm", "delta_n2_mm", "tube_offset_id"}.issubset(tube.columns)
    assert int(tube["is_centerline"].sum()) == 4


def test_centerline_gate_enforces_planned_seam_limit() -> None:
    mod = _load_utils()
    report = {
        "residual_p95_mm": 0.1,
        "residual_max_mm": 0.2,
        "delta_beta_p95_deg": 0.1,
        "delta_beta_max_deg": 0.2,
        "delta2_beta_p95_deg": 0.01,
        "seam_beta_rms_deg": 0.76,
        "sigma3_p05_m": 0.2,
        "kappa_p95": 20.0,
    }

    gates = mod.evaluate_centerline_gates(report)

    assert gates["branch_gate_pass"] is False
    assert gates["canonical_gate_pass"] is False
    assert gates["centerline_gate_pass"] is False


def test_centerline_gate_separates_legacy_strict_conditioning_from_downstream_admission() -> None:
    mod = _load_utils()
    report = {
        "residual_p95_mm": 0.1,
        "residual_max_mm": 0.2,
        "delta_beta_p95_deg": 0.1,
        "delta_beta_max_deg": 0.2,
        "delta2_beta_p95_deg": 0.01,
        "seam_beta_rms_deg": 0.03,
        "sigma3_p05_m": 0.02,
        "kappa_p95": 151.0,
    }

    gates = mod.evaluate_centerline_gates(report)

    assert gates["conditioning_gate_pass"] is False
    assert gates["strict_conditioning_gate_pass"] is False
    assert gates["centerline_gate_pass"] is False
    assert gates["downstream_admission_conditioning_gate_pass"] is True
    assert gates["downstream_admission_gate_pass"] is True


def test_conditioning_policy_sweep_changes_only_candidate_strict_decision() -> None:
    mod = _load_utils()
    report = {"sigma3_p05_m": 0.02, "kappa_p95": 234.0}

    rejected = mod.evaluate_conditioning_policy(report, kappa_threshold=200.0)
    admitted = mod.evaluate_conditioning_policy(report, kappa_threshold=250.0)

    assert rejected["conditioning_policy_gate_pass"] is False
    assert admitted["conditioning_policy_gate_pass"] is True
    assert rejected["sigma3_gate_pass"] is True
    assert admitted["sigma3_gate_pass"] is True


def test_stage_selector_materializes_candidate_accepted_stage_not_last_tracking_stage() -> None:
    mod = _load_utils()
    first = pd.DataFrame({"stage_marker": [1]})
    last = pd.DataFrame({"stage_marker": [2]})
    outputs = [
        (
            "conservative_1",
            first,
            {
                "centerline_gate_pass": False,
                "branch_gate_pass": True,
                "stage_acceptance_gate_pass": True,
                "kappa_p95": 338.0,
            },
            np.zeros((1, 6)),
        ),
        (
            "conservative_3",
            last,
            {
                "centerline_gate_pass": False,
                "branch_gate_pass": True,
                "stage_acceptance_gate_pass": False,
                "kappa_p95": 415.0,
            },
            np.ones((1, 6)),
        ),
    ]

    selected = mod.select_trajectory_stage(outputs)

    assert selected[0] == "conservative_1"
    assert selected[1]["stage_marker"].tolist() == [1]
    assert selected[2]["kappa_p95"] == 338.0
