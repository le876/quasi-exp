"""Auditable dataset serialization and physical gates for V10 teachers."""

from __future__ import annotations

import math
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd

from .canonical import TeacherTrajectory


CENTERLINE_GATE_V10: Mapping[str, float] = {
    "residual_p95_mm": 1.0,
    "residual_max_mm": 3.0,
    "delta_beta_rms_p95_deg": 1.0,
    "delta_beta_rms_max_deg": 2.0,
    "acceleration_beta_rms_p95_deg": 0.5,
    "seam_beta_rms_deg": 1.0,
    "joint_margin_min_deg": 1.5,
}

TUBE_GATE_V10: Mapping[str, float] = {
    "success_rate": 0.99,
    "residual_p95_mm": 1.5,
    "residual_max_mm": 3.0,
    "local_beta_rms_p95_deg": 1.0,
    "multi_branch_ratio": 0.0,
}


class _JacobianEnvironment(Protocol):
    bounds: np.ndarray

    def jacobian(self, beta: np.ndarray) -> np.ndarray: ...


def evaluate_centerline_gate(
    trajectory: TeacherTrajectory,
    *,
    thresholds: Mapping[str, float] = CENTERLINE_GATE_V10,
) -> dict[str, Any]:
    metrics = trajectory.metrics
    checks = {
        "solver_success": bool(trajectory.success),
        "residual_p95": float(metrics.get("residual_p95_mm", math.inf))
        <= float(thresholds["residual_p95_mm"]),
        "residual_max": float(metrics.get("residual_max_mm", math.inf))
        <= float(thresholds["residual_max_mm"]),
        "delta_beta_p95": float(metrics.get("delta_beta_rms_p95_deg", math.inf))
        <= float(thresholds["delta_beta_rms_p95_deg"]),
        "delta_beta_max": float(metrics.get("delta_beta_rms_max_deg", math.inf))
        <= float(thresholds["delta_beta_rms_max_deg"]),
        "acceleration_p95": float(
            metrics.get("acceleration_beta_rms_p95_deg", math.inf)
        )
        <= float(thresholds["acceleration_beta_rms_p95_deg"]),
        "seam": float(metrics.get("seam_beta_rms_deg", math.inf))
        <= float(thresholds["seam_beta_rms_deg"]),
        "joint_margin_min": float(metrics.get("joint_margin_min_deg", -math.inf))
        >= float(thresholds["joint_margin_min_deg"]),
    }
    return {
        "gate_id": "trajectory-centerline-v10.1",
        "thresholds": dict(thresholds),
        "checks": checks,
        "centerline_gate_pass": bool(all(checks.values())),
    }


def evaluate_tube_gate(
    metrics: Mapping[str, float],
    *,
    thresholds: Mapping[str, float] = TUBE_GATE_V10,
) -> dict[str, Any]:
    checks = {
        "success_rate": float(metrics.get("success_rate", -math.inf))
        >= float(thresholds["success_rate"]),
        "residual_p95": float(metrics.get("residual_p95_mm", math.inf))
        <= float(thresholds["residual_p95_mm"]),
        "residual_max": float(metrics.get("residual_max_mm", math.inf))
        <= float(thresholds["residual_max_mm"]),
        "local_beta_p95": float(metrics.get("local_beta_rms_p95_deg", math.inf))
        <= float(thresholds["local_beta_rms_p95_deg"]),
        "multi_branch_zero": float(metrics.get("multi_branch_ratio", math.inf))
        <= float(thresholds["multi_branch_ratio"]),
    }
    return {
        "gate_id": "trajectory-tube-v10.1",
        "thresholds": dict(thresholds),
        "checks": checks,
        "tube_gate_pass": bool(all(checks.values())),
    }


def trajectory_frame(
    trajectory: TeacherTrajectory,
    environment: _JacobianEnvironment,
    *,
    tube_n1_mm: float = 0.0,
    tube_n2_mm: float = 0.0,
) -> pd.DataFrame:
    """Serialize all state and lineage needed for static/stateful/chart audits."""

    beta = np.asarray(trajectory.beta_rad, dtype=float).reshape(-1, 6)
    theta = np.asarray(trajectory.theta_rad, dtype=float).reshape(len(beta), 30)
    target = np.asarray(trajectory.target_xyz_m, dtype=float).reshape(len(beta), 3)
    achieved = np.asarray(trajectory.achieved_xyz_m, dtype=float).reshape(len(beta), 3)
    if not all(np.isfinite(values).all() for values in (beta, theta, target, achieved)):
        raise ValueError("teacher trajectory contains non-finite numerical values")
    provenance = trajectory.provenance
    frame = pd.DataFrame(
        {
            "target_x_m": target[:, 0],
            "target_y_m": target[:, 1],
            "target_z_m": target[:, 2],
            "achieved_x_m": achieved[:, 0],
            "achieved_y_m": achieved[:, 1],
            "achieved_z_m": achieved[:, 2],
            "teacher_fk_residual_mm": np.linalg.norm(achieved - target, axis=1) * 1000.0,
            "trajectory_id": str(provenance["trajectory_id"]),
            "family_id": str(provenance["family_id"]),
            "radius_mm": float(provenance["radius_mm"]),
            "phase_idx": np.arange(len(beta), dtype=np.int64),
            "phase_rad": np.arange(len(beta), dtype=float) * (2.0 * math.pi / len(beta)),
            "tube_n1_mm": float(tube_n1_mm),
            "tube_n2_mm": float(tube_n2_mm),
            "chart_id": np.asarray(trajectory.chart_id, dtype=np.int64),
            "branch_id": np.asarray(trajectory.branch_id, dtype=np.int64),
            "teacher_policy_id": str(provenance["teacher_policy_id"]),
            "solver_seed": int(provenance["solver_seed"]),
            "traversal_direction": str(provenance["traversal_direction"]),
            "cyclic_cut": int(provenance["cyclic_cut"]),
        }
    )
    previous = np.roll(beta, 1, axis=0)
    following = np.roll(beta, -1, axis=0)
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    margin_deg = np.rad2deg(
        np.minimum(beta - bounds[:, 0][None, :], bounds[:, 1][None, :] - beta)
    )
    frame["joint_margin_min_deg"] = np.min(margin_deg, axis=1)
    for index in range(6):
        ordinal = index + 1
        frame[f"teacher_beta{ordinal}_rad"] = beta[:, index]
        frame[f"previous_beta{ordinal}_rad"] = previous[:, index]
        frame[f"next_beta{ordinal}_rad"] = following[:, index]
    for index in range(30):
        frame[f"theta_{index + 1}_rad"] = theta[:, index]

    jacobians = [np.asarray(environment.jacobian(row), dtype=float) for row in beta]
    singular = np.asarray([np.linalg.svd(jac, compute_uv=False) for jac in jacobians])
    frame["sigma1_m"] = singular[:, 0]
    frame["sigma2_m"] = singular[:, 1]
    frame["sigma3_m"] = singular[:, 2]
    frame["kappa"] = np.divide(
        singular[:, 0],
        singular[:, 2],
        out=np.full(len(beta), np.inf),
        where=singular[:, 2] > 0.0,
    )
    frame["teacher_posture_cost"] = np.mean(
        np.square(beta) * np.asarray([4.0, 4.0, 2.0, 2.0, 1.0, 1.0])[None, :],
        axis=1,
    )
    frame["teacher_velocity_cost"] = np.mean(np.square(following - beta), axis=1)
    frame["teacher_acceleration_cost"] = np.mean(
        np.square(following - 2.0 * beta + previous), axis=1
    )
    frame["teacher_condition_cost"] = frame["kappa"]
    frame["root_configuration_id"] = str(provenance.get("root_configuration_id", "root0"))
    frame["sample_id"] = [
        f"{provenance['trajectory_id']}:{index:04d}:{float(tube_n1_mm):+.3f}:{float(tube_n2_mm):+.3f}"
        for index in range(len(beta))
    ]
    if not frame["sample_id"].is_unique:
        raise ValueError("trajectory serialization produced duplicate sample ids")
    return frame
