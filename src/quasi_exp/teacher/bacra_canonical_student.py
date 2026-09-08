"""Canonical-beta diagnostics and deterministic pilot selection for BACRA."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from quasi_exp.model.kinematics import forward_kinematics
from quasi_exp.model.sampling import beta_to_theta

from .dense_chart_sampling import BETA_COLUMNS, XYZ_COLUMNS
from .student_tracking_tf import StudentGeometry


def _fk(beta_rad: np.ndarray, geometry: StudentGeometry) -> np.ndarray:
    return forward_kinematics(
        beta_to_theta(beta_rad),
        geometry.lengths_m,
        geometry.p_end_local_m,
        geometry.theta_sign,
    )[0]


def evaluate_canonical_predictions(
    frame: pd.DataFrame,
    prediction_rad: np.ndarray,
    *,
    geometry: StudentGeometry,
    jacobian_eps_rad: float = 1.0e-4,
) -> dict[str, Any]:
    """Measure actuator-label error and its local FK row/null decomposition."""

    prediction = np.asarray(prediction_rad, dtype=float)
    truth = frame.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
    if prediction.shape != truth.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} does not match {truth.shape}"
        )
    target_xyz = frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    delta = prediction - truth
    row_delta = np.empty_like(delta)
    sigma3 = np.empty(len(frame), dtype=float)
    for row, beta in enumerate(truth):
        jacobian = np.empty((3, 6), dtype=float)
        for joint in range(6):
            step = np.zeros(6, dtype=float)
            step[joint] = float(jacobian_eps_rad)
            jacobian[:, joint] = (
                _fk(beta + step, geometry) - _fk(beta - step, geometry)
            ) / (2.0 * float(jacobian_eps_rad))
        _u, singular, vt = np.linalg.svd(jacobian, full_matrices=True)
        rank = int(
            np.sum(
                singular
                > max(float(singular[0]) * 1.0e-8, 1.0e-10)
            )
        )
        row_basis = vt[:rank]
        row_delta[row] = row_basis.T @ (row_basis @ delta[row])
        sigma3[row] = float(singular[2])
    null_delta = delta - row_delta

    beta_abs_deg = np.abs(np.rad2deg(delta))
    beta_rms_deg = np.rad2deg(
        np.sqrt(np.mean(np.square(delta), axis=1))
    )
    row_rms_deg = np.rad2deg(
        np.sqrt(np.mean(np.square(row_delta), axis=1))
    )
    null_rms_deg = np.rad2deg(
        np.sqrt(np.mean(np.square(null_delta), axis=1))
    )
    null_energy_fraction = np.sum(
        np.square(null_delta), axis=1
    ) / np.maximum(np.sum(np.square(delta), axis=1), 1.0e-20)

    predicted_xyz = np.vstack(
        [_fk(beta, geometry) for beta in prediction]
    )
    fk_error_mm = (
        np.linalg.norm(predicted_xyz - target_xyz, axis=1) * 1000.0
    )
    bounds = np.asarray(geometry.beta_bounds_rad, dtype=float)
    midpoint = 0.5 * (bounds[:, 0] + bounds[:, 1])
    halfspan = 0.5 * (bounds[:, 1] - bounds[:, 0])
    normalized = (prediction - midpoint[None, :]) / halfspan[None, :]
    margins_deg = np.rad2deg(
        np.minimum(
            prediction - bounds[:, 0][None, :],
            bounds[:, 1][None, :] - prediction,
        )
    )
    return {
        "row_count": int(len(frame)),
        "validation_beta_rms_p50_deg": float(
            np.percentile(beta_rms_deg, 50)
        ),
        "validation_beta_rms_p95_deg": float(
            np.percentile(beta_rms_deg, 95)
        ),
        "validation_beta_rms_max_deg": float(np.max(beta_rms_deg)),
        "validation_beta_abs_p50_by_joint_deg": np.percentile(
            beta_abs_deg, 50, axis=0
        ).tolist(),
        "validation_beta_abs_p95_by_joint_deg": np.percentile(
            beta_abs_deg, 95, axis=0
        ).tolist(),
        "validation_beta_abs_max_by_joint_deg": np.max(
            beta_abs_deg, axis=0
        ).tolist(),
        "validation_beta_signed_bias_by_joint_deg": np.mean(
            np.rad2deg(delta), axis=0
        ).tolist(),
        "validation_rowspace_beta_rms_p50_deg": float(
            np.percentile(row_rms_deg, 50)
        ),
        "validation_rowspace_beta_rms_p95_deg": float(
            np.percentile(row_rms_deg, 95)
        ),
        "validation_nullspace_beta_rms_p50_deg": float(
            np.percentile(null_rms_deg, 50)
        ),
        "validation_nullspace_beta_rms_p95_deg": float(
            np.percentile(null_rms_deg, 95)
        ),
        "validation_null_energy_fraction_p50": float(
            np.percentile(null_energy_fraction, 50)
        ),
        "validation_null_energy_fraction_p95": float(
            np.percentile(null_energy_fraction, 95)
        ),
        "validation_jacobian_sigma3_p05_m": float(
            np.percentile(sigma3, 5)
        ),
        "validation_fk_p50_mm": float(np.percentile(fk_error_mm, 50)),
        "validation_fk_p95_mm": float(np.percentile(fk_error_mm, 95)),
        "validation_fk_max_mm": float(np.max(fk_error_mm)),
        "prediction_near_saturation_fraction": float(
            np.mean(np.abs(normalized) >= 0.95)
        ),
        "predicted_minimum_joint_margin_deg": float(np.min(margins_deg)),
        "predicted_margin_lt_1p5_fraction": float(
            np.mean(np.min(margins_deg, axis=1) < 1.5)
        ),
    }


def project_toward_canonical_beta(
    base_beta_rad: np.ndarray,
    canonical_beta_rad: np.ndarray,
    *,
    geometry: StudentGeometry,
    alpha: float,
    jacobian_eps_rad: float = 1.0e-4,
) -> np.ndarray:
    """Move toward a canonical estimate only along the local FK null-space."""

    base = np.asarray(base_beta_rad, dtype=float)
    canonical = np.asarray(canonical_beta_rad, dtype=float)
    if base.shape != canonical.shape or base.ndim != 2 or base.shape[1] != 6:
        raise ValueError("base and canonical beta must both have shape (N, 6)")
    projected = np.empty_like(base)
    identity = np.eye(6)
    for row, beta in enumerate(base):
        jacobian = np.empty((3, 6), dtype=float)
        for joint in range(6):
            step = np.zeros(6, dtype=float)
            step[joint] = float(jacobian_eps_rad)
            jacobian[:, joint] = (
                _fk(beta + step, geometry) - _fk(beta - step, geometry)
            ) / (2.0 * float(jacobian_eps_rad))
        null_projector = identity - np.linalg.pinv(jacobian) @ jacobian
        projected[row] = (
            beta
            + float(alpha)
            * null_projector
            @ (canonical[row] - beta)
        )
    return projected


def exploratory_student_admission_checks(
    metrics: Mapping[str, Any],
    *,
    joint_abs_p95_max_deg: float,
    fk_p95_max_mm: float,
    fk_max_mm: float,
    minimum_joint_margin_min_deg: float,
) -> dict[str, bool]:
    """Evaluate the four frozen exploratory Student admission criteria."""

    joint_p95 = np.asarray(
        metrics["validation_beta_abs_p95_by_joint_deg"], dtype=float
    )
    return {
        "all_joint_abs_p95_below_limit": bool(
            np.all(joint_p95 < float(joint_abs_p95_max_deg))
        ),
        "fk_p95_below_limit": bool(
            float(metrics["validation_fk_p95_mm"])
            < float(fk_p95_max_mm)
        ),
        "fk_max_below_limit": bool(
            float(metrics["validation_fk_max_mm"]) < float(fk_max_mm)
        ),
        "minimum_joint_margin_above_limit": bool(
            float(metrics["predicted_minimum_joint_margin_deg"])
            > float(minimum_joint_margin_min_deg)
        ),
    }


def select_pilot_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    fk_p95_max_mm: float,
    fk_max_mm: float,
    beta_rms_p95_max_deg: float,
    joint_abs_p95_max_deg: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Apply the frozen V12.7 eligibility and ranking rules."""

    evaluated: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        worst_joint = max(
            map(
                float,
                row["validation_beta_abs_p95_by_joint_deg"],
            )
        )
        row["worst_joint_abs_p95_deg"] = worst_joint
        row["fk_gate_pass"] = bool(
            float(row["validation_fk_p95_mm"]) <= float(fk_p95_max_mm)
            and float(row["validation_fk_max_mm"]) <= float(fk_max_mm)
        )
        row["canonical_beta_target_pass"] = bool(
            float(row["validation_beta_rms_p95_deg"])
            <= float(beta_rms_p95_max_deg)
            and worst_joint <= float(joint_abs_p95_max_deg)
        )
        evaluated.append(row)
    eligible = [row for row in evaluated if row["fk_gate_pass"]]
    eligible.sort(
        key=lambda row: (
            float(row["validation_beta_rms_p95_deg"]),
            float(row["worst_joint_abs_p95_deg"]),
            float(row["validation_beta_rms_max_deg"]),
            float(row["validation_fk_p95_mm"]),
            float(row["lambda_fk"]),
        )
    )
    return (eligible[0] if eligible else None), evaluated
