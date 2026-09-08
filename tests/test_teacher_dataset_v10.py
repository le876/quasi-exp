from __future__ import annotations

import numpy as np

from quasi_exp.teacher.canonical import TeacherTrajectory
from quasi_exp.teacher.dataset import (
    CENTERLINE_GATE_V10,
    evaluate_centerline_gate,
    trajectory_frame,
    evaluate_tube_gate,
)


class _LinearEnvironment:
    bounds = np.deg2rad(np.asarray([[-5.0, 5.0]] * 6))

    def jacobian(self, beta: np.ndarray) -> np.ndarray:
        del beta
        return np.concatenate([np.eye(3), np.zeros((3, 3))], axis=1)


def _trajectory() -> TeacherTrajectory:
    beta = np.zeros((4, 6), dtype=float)
    beta[:, 0] = np.deg2rad([0.0, 0.1, 0.2, 0.1])
    target = beta[:, :3].copy()
    return TeacherTrajectory(
        beta_rad=beta,
        theta_rad=np.repeat(beta, 5, axis=1),
        achieved_xyz_m=target.copy(),
        target_xyz_m=target,
        chart_id=np.asarray([0, 0, 1, 1]),
        branch_id=np.zeros(4, dtype=np.int64),
        metrics={
            "residual_p95_mm": 0.0,
            "residual_max_mm": 0.0,
            "delta_beta_rms_p95_deg": 0.1,
            "delta_beta_rms_max_deg": 0.1,
            "acceleration_beta_rms_p95_deg": 0.1,
            "seam_beta_rms_deg": 0.1,
            "joint_margin_min_deg": 4.8,
        },
        provenance={
            "trajectory_id": "curve",
            "family_id": "family",
            "radius_mm": 92.5,
            "teacher_policy_id": "policy-sha",
            "solver_seed": 7,
            "traversal_direction": "forward",
            "cyclic_cut": 0,
        },
        success=True,
    )


def test_trajectory_frame_contains_history_geometry_conditioning_and_lineage() -> None:
    frame = trajectory_frame(_trajectory(), _LinearEnvironment())

    required = {
        "target_x_m", "target_y_m", "target_z_m",
        "teacher_beta1_rad", "teacher_beta6_rad",
        "previous_beta1_rad", "next_beta6_rad",
        "theta_1_rad", "theta_30_rad",
        "trajectory_id", "family_id", "radius_mm", "phase_rad",
        "tube_n1_mm", "tube_n2_mm", "chart_id", "branch_id",
        "teacher_fk_residual_mm", "sigma1_m", "sigma2_m", "sigma3_m",
        "kappa", "joint_margin_min_deg", "teacher_policy_id",
        "solver_seed", "traversal_direction", "cyclic_cut",
    }
    assert required <= set(frame.columns)
    np.testing.assert_allclose(frame.loc[0, "previous_beta1_rad"], frame.loc[3, "teacher_beta1_rad"])
    np.testing.assert_allclose(frame.loc[3, "next_beta1_rad"], frame.loc[0, "teacher_beta1_rad"])
    assert frame["sample_id"].is_unique


def test_centerline_gate_uses_the_registered_physical_thresholds() -> None:
    passed = evaluate_centerline_gate(_trajectory())
    failed_trajectory = _trajectory()
    failed_trajectory.metrics["joint_margin_min_deg"] = 1.49
    failed = evaluate_centerline_gate(failed_trajectory)

    assert CENTERLINE_GATE_V10["joint_margin_min_deg"] == 1.5
    assert passed["centerline_gate_pass"] is True
    assert failed["centerline_gate_pass"] is False
    assert failed["checks"]["joint_margin_min"] is False


def test_tube_gate_returns_a_complete_boolean_report() -> None:
    report = evaluate_tube_gate(
        {"success_rate": 1.0, "residual_p95_mm": 1.0, "residual_max_mm": 2.0,
         "local_beta_rms_p95_deg": 0.5, "multi_branch_ratio": 0.0}
    )
    assert report["tube_gate_pass"] is True
    assert all(report["checks"].values())
