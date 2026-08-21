from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "run_true_ellipse_beta6_training_v4.py"
    spec = importlib.util.spec_from_file_location("run_true_ellipse_beta6_training_v4", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _toy_tube(n_angles: int = 4) -> pd.DataFrame:
    rows = []
    offsets = [-5.0, -2.5, 0.0, 2.5, 5.0]
    for angle_idx in range(n_angles):
        angle = 2.0 * math.pi * angle_idx / n_angles
        for n1 in offsets:
            for n2 in offsets:
                row = {
                    "angle_idx": angle_idx,
                    "angle_rad": angle,
                    "delta_n1_mm": n1,
                    "delta_n2_mm": n2,
                    "tube_offset_id": f"n1_{n1:g}_n2_{n2:g}",
                    "is_centerline": bool(n1 == 0.0 and n2 == 0.0),
                    "x_target_m": angle_idx * 0.01 + n1 * 1.0e-4,
                    "y_target_m": n2 * 1.0e-4,
                    "z_target_m": 0.0,
                    "x_m": angle_idx * 0.01 + n1 * 1.0e-4,
                    "y_m": n2 * 1.0e-4,
                    "z_m": 0.0,
                }
                for i in range(1, 7):
                    row[f"beta{i}_rad"] = 0.001 * (angle_idx + i + n1 + n2)
                rows.append(row)
    return pd.DataFrame(rows)


def test_exact_free_phase_ellipse_preserves_v3_phases() -> None:
    mod = _load_module()
    metadata = {
        "center_x_m": 1.1,
        "center_y_m": 0.2,
        "center_z_m": -0.3,
        "phase_y_rad": math.radians(120.0),
        "phase_z_rad": math.radians(30.0),
    }

    xyz, angle = mod.generate_exact_ellipse(metadata, amp_xy_mm=75.0, n_points=4)

    assert np.isclose(angle[0], 0.0)
    assert np.isclose(xyz[0, 0], 1.1)
    assert np.isclose(xyz[0, 1], 0.2 + 0.075 * math.sin(math.radians(120.0)))
    assert np.isclose(xyz[0, 2], -0.3 + 0.1125 * math.sin(math.radians(30.0)))


def test_tube_split_holds_out_centerline_without_key_leakage() -> None:
    mod = _load_module()
    df = _toy_tube()

    split = mod.make_tube_split(df)

    assert len(split.train_idx) == 80
    assert len(split.val_idx) == 16
    assert len(split.test_idx) == 4
    assert set(split.train_idx).isdisjoint(split.val_idx)
    assert set(split.train_idx).isdisjoint(split.test_idx)
    assert set(split.val_idx).isdisjoint(split.test_idx)
    assert df.iloc[split.test_idx]["is_centerline"].all()
    val_offsets = set(map(tuple, df.iloc[split.val_idx][["delta_n1_mm", "delta_n2_mm"]].drop_duplicates().to_numpy()))
    assert val_offsets == {(-2.5, 0.0), (2.5, 0.0), (0.0, -2.5), (0.0, 2.5)}


def test_support_metrics_only_see_explicit_training_pool() -> None:
    mod = _load_module()
    target = np.array([[0.0, 0.0, 0.0]])
    training = np.array([[0.020, 0.0, 0.0]])
    held_out = np.array([[0.0, 0.0, 0.0]])

    report = mod.compute_support_metrics(training, target, radius_mm=15.0)
    leaked = mod.compute_support_metrics(np.vstack([training, held_out]), target, radius_mm=15.0)

    assert np.isclose(report["nn_p95_mm"], 20.0)
    assert report["tube_count_p10"] == 0.0
    assert leaked["nn_p95_mm"] == 0.0
    assert leaked["tube_count_p10"] == 1.0


def test_input_audit_accepts_one_current_selected_branch_without_hardcoded_run_id() -> None:
    mod = _load_module()
    df = _toy_tube(n_angles=360)
    df["candidate_id"] = "candidate"
    df["ellipse_id"] = "candidate"
    df["center_x_m"] = 1.0
    df["center_y_m"] = 0.0
    df["center_z_m"] = 0.0
    df["amp_xy_mm"] = 75.0
    df["amp_z_mm"] = 112.5
    df["phase_y_rad"] = math.radians(120.0)
    df["phase_z_rad"] = math.radians(30.0)
    df["ik_success"] = True
    df["tube_success"] = True
    df["xyz_residual_mm"] = 0.0
    df["continuation_run_id"] = "v2_02_forward"
    theta = mod.theta_from_beta_batch(
        df[[f"beta{i}_rad" for i in range(1, 7)]].to_numpy(dtype=float),
        theta_sign=-1.0,
    )
    for i in range(30):
        df[f"theta_{i + 1}_rad"] = theta[:, i]

    report = mod.audit_tube_dataset(df, theta_sign=-1.0)

    assert report["continuation_run_ids"] == ["v2_02_forward"]
    assert report["checks"]["selected_branch_only"] is True
    assert report["audit_gate_pass"] is True


def test_periodic_smoothness_includes_closure_edge() -> None:
    mod = _load_module()
    beta = np.zeros((4, 6), dtype=float)
    beta[:, 0] = np.deg2rad([0.0, 0.1, 0.2, 0.3])

    report = mod.periodic_beta_metrics(beta)

    assert report["delta_beta_max_deg"] > 0.1
    assert np.isclose(report["seam_beta_rms_deg"], 0.3 / math.sqrt(6.0))
    assert report["delta2_beta_p95_deg"] > 0.0


def test_open_sector_smoothness_does_not_create_artificial_seam() -> None:
    mod = _load_module()
    beta = np.zeros((4, 6), dtype=float)
    beta[:, 0] = np.deg2rad([0.0, 0.1, 0.2, 0.3])

    report = mod.linear_beta_metrics(beta)

    assert np.isclose(report["delta_beta_max_deg"], 0.1 / math.sqrt(6.0))
    assert report["seam_beta_rms_deg"] is None


def test_four_of_five_seed_aggregation() -> None:
    mod = _load_module()
    rows = pd.DataFrame({"seed": [1, 2, 3, 4, 5], "model_gate_pass": [True, True, True, True, False]})

    report = mod.aggregate_seed_gate(rows, required_fraction=0.8)

    assert report["passed_seed_count"] == 4
    assert report["total_seed_count"] == 5
    assert report["stable_gate_pass"] is True


def test_max_radius_must_be_contiguous_from_anchor() -> None:
    mod = _load_module()
    table = pd.DataFrame(
        {
            "amp_xy_mm": [75.0, 75.25, 75.5, 75.75, 76.0],
            "strict_gate_pass": [True, True, False, True, True],
            "relaxed_gate_pass": [True, True, True, True, False],
            "model_only_gate_pass": [True, True, True, True, True],
        }
    )

    assert mod.contiguous_max_radius(table, "strict_gate_pass", anchor_mm=75.0) == 75.25
    assert mod.contiguous_max_radius(table, "relaxed_gate_pass", anchor_mm=75.0) == 75.75
    assert mod.contiguous_max_radius(table, "model_only_gate_pass", anchor_mm=75.0) == 76.0


def test_no_official_radius_when_e75_seed_gate_fails() -> None:
    mod = _load_module()
    table = pd.DataFrame(
        {
            "amp_xy_mm": [75.0, 75.25],
            "strict_gate_pass": [False, True],
            "relaxed_gate_pass": [False, True],
            "model_only_gate_pass": [False, True],
        }
    )

    report = mod.select_stable_radii(table, anchor_mm=75.0)

    assert report["strict_supported_rmax_mm"] is None
    assert report["relaxed_supported_rmax_mm"] is None
    assert report["model_only_rmax_mm"] is None


def test_strict_limit_identifies_support_boundary() -> None:
    mod = _load_module()
    table = pd.DataFrame(
        {
            "amp_xy_mm": [75.0, 75.25],
            "stable_gate_pass": [True, True],
            "strict_support_gate_pass": [True, False],
            "strict_gate_pass": [True, False],
        }
    )

    assert mod.classify_strict_limit(table, strict_rmax_mm=75.0) == "data_support"


def test_prediction_bound_metrics_use_the_explicit_joint_domain() -> None:
    mod = _load_module()
    from true_ellipse_radial_bundle_engine import registered_joint_domain

    beta = np.zeros((3, 6), dtype=float)
    beta[:, 2] = np.deg2rad(7.0)
    cfg = mod.load_config(str(mod.DEFAULT_CONFIG))
    robot = mod.load_robot_inputs(cfg)
    theta_sign = float(cfg["kinematics"]["theta_sign"])
    target = mod.fk_dh_batch(
        mod.theta_from_beta_batch(beta, theta_sign=theta_sign),
        lengths_m=robot.lengths_m,
        p_end_local_m=robot.p_end_local_m,
    )

    current, _xyz, _theta = mod.evaluate_beta_prediction(
        beta_pred=beta,
        target_xyz=target,
        lengths_m=robot.lengths_m,
        p_end_local_m=robot.p_end_local_m,
        theta_sign=theta_sign,
        joint_domain=registered_joint_domain("current_v6"),
    )
    standard, _xyz, _theta = mod.evaluate_beta_prediction(
        beta_pred=beta,
        target_xyz=target,
        lengths_m=robot.lengths_m,
        p_end_local_m=robot.p_end_local_m,
        theta_sign=theta_sign,
        joint_domain=registered_joint_domain("standard_beta34_10deg_v1"),
    )

    assert current["joint_domain_id"] == "current_v6"
    assert current["beta_bound_violation_count"] == 3
    assert np.isclose(current["prediction_min_joint_margin_deg"], -2.0)
    assert standard["joint_domain_id"] == "standard_beta34_10deg_v1"
    assert standard["beta_bound_violation_count"] == 0
    assert np.isclose(standard["prediction_min_joint_margin_deg"], 3.0)
