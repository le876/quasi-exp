"""Independent audits and hard gates for V11 region artifacts."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


BETA_COLUMNS = tuple(f"teacher_beta{index}_rad" for index in range(1, 7))
XYZ_COLUMNS = ("target_x_m", "target_y_m", "target_z_m")


DEFAULT_SURFACE_THRESHOLDS: Mapping[str, float] = {
    "success_rate": 1.0,
    "residual_p95_mm": 1.0,
    "residual_max_mm": 3.0,
    "joint_margin_min_deg": 1.5,
    "phase_beta_rms_p95_deg": 1.0,
    "acceleration_beta_rms_p95_deg": 0.5,
    "seam_beta_rms_max_deg": 1.0,
    "surface_edge_beta_rms_p95_deg": 1.0,
    "surface_laplacian_beta_rms_p95_deg": 1.0,
    "surface_block_update_rms_max_deg": 0.05,
}


def evaluate_teacher_surface_gate(
    metrics: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float] = DEFAULT_SURFACE_THRESHOLDS,
) -> dict[str, Any]:
    required = set(DEFAULT_SURFACE_THRESHOLDS)
    missing_metrics = sorted(required - set(metrics))
    missing_thresholds = sorted(required - set(thresholds))
    if missing_metrics:
        raise ValueError(f"surface metrics missing fields: {missing_metrics}")
    if missing_thresholds:
        raise ValueError(f"surface thresholds missing fields: {missing_thresholds}")
    checks = {
        "all_rows_success": bool(
            float(metrics["success_rate"])
            >= float(thresholds["success_rate"]) - 1.0e-12
        ),
        "residual_p95": bool(
            float(metrics["residual_p95_mm"])
            <= float(thresholds["residual_p95_mm"])
        ),
        "residual_max": bool(
            float(metrics["residual_max_mm"])
            <= float(thresholds["residual_max_mm"])
        ),
        "joint_margin": bool(
            float(metrics["joint_margin_min_deg"])
            >= float(thresholds["joint_margin_min_deg"])
        ),
        "phase_smoothness": bool(
            float(metrics["phase_beta_rms_p95_deg"])
            <= float(thresholds["phase_beta_rms_p95_deg"])
        ),
        "phase_acceleration": bool(
            float(metrics["acceleration_beta_rms_p95_deg"])
            <= float(thresholds["acceleration_beta_rms_p95_deg"])
        ),
        "cyclic_seam": bool(
            float(metrics["seam_beta_rms_max_deg"])
            <= float(thresholds["seam_beta_rms_max_deg"])
        ),
        "normal_smoothness": bool(
            float(metrics["surface_edge_beta_rms_p95_deg"])
            <= float(thresholds["surface_edge_beta_rms_p95_deg"])
        ),
        "normal_laplacian_smoothness": bool(
            float(metrics["surface_laplacian_beta_rms_p95_deg"])
            <= float(thresholds["surface_laplacian_beta_rms_p95_deg"])
        ),
        "whole_surface_block_convergence": bool(
            float(metrics["surface_block_update_rms_max_deg"])
            <= float(thresholds["surface_block_update_rms_max_deg"])
        ),
    }
    return {
        "thresholds": {key: float(value) for key, value in thresholds.items()},
        "metrics": {key: float(metrics[key]) for key in required},
        "checks": checks,
        "gate_pass": bool(all(checks.values())),
    }


def audit_cartesian_coverage(
    reference_xyz_m: np.ndarray,
    probe_xyz_m: np.ndarray,
    *,
    p95_limit_mm: float = 3.0,
    max_limit_mm: float = 5.0,
) -> dict[str, Any]:
    reference = np.asarray(reference_xyz_m, dtype=float).reshape(-1, 3)
    probe = np.asarray(probe_xyz_m, dtype=float).reshape(-1, 3)
    if len(reference) == 0 or len(probe) == 0:
        raise ValueError("coverage audit needs non-empty reference and probe sets")
    if not np.isfinite(reference).all() or not np.isfinite(probe).all():
        raise ValueError("coverage audit coordinates must be finite")
    distance_mm = np.asarray(cKDTree(reference).query(probe, k=1)[0]) * 1000.0
    p95 = float(np.percentile(distance_mm, 95))
    maximum = float(np.max(distance_mm))
    checks = {
        "nearest_p95": bool(p95 <= float(p95_limit_mm)),
        "nearest_max": bool(maximum <= float(max_limit_mm)),
    }
    return {
        "reference_count": int(len(reference)),
        "probe_count": int(len(probe)),
        "nearest_distance_p95_mm": p95,
        "nearest_distance_max_mm": maximum,
        "checks": checks,
        "coverage_gate_pass": bool(all(checks.values())),
    }


def _beta_rms_gap_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.sqrt(np.mean(np.square(left - right), axis=1)))


def audit_cross_family_conflicts(
    frame: pd.DataFrame,
    *,
    xyz_radius_mm: float = 2.0,
    beta_gap_threshold_deg: float = 1.0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {"family_id", "sample_id", *XYZ_COLUMNS, *BETA_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"conflict audit missing columns: {missing}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("conflict audit sample_id must be unique")
    if float(xyz_radius_mm) <= 0.0 or float(beta_gap_threshold_deg) <= 0.0:
        raise ValueError("conflict audit thresholds must be positive")
    xyz = frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    beta = frame.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
    families = frame["family_id"].astype(str).to_numpy()
    samples = frame["sample_id"].astype(str).to_numpy()
    tree = cKDTree(xyz)
    pairs: list[tuple[int, int]] = []
    for left, neighbours in enumerate(
        tree.query_ball_point(xyz, r=float(xyz_radius_mm) / 1000.0)
    ):
        for right in neighbours:
            if right <= left or families[left] == families[right]:
                continue
            pairs.append((left, int(right)))
    columns = (
        "left_sample_id",
        "right_sample_id",
        "left_family_id",
        "right_family_id",
        "xyz_distance_mm",
        "beta_gap_rms_deg",
    )
    if not pairs:
        report = {
            "xyz_radius_mm": float(xyz_radius_mm),
            "beta_gap_threshold_deg": float(beta_gap_threshold_deg),
            "cross_family_pair_count": 0,
            "conflict_pair_count": 0,
            "conflict_pair_ratio": 0.0,
            "beta_gap_p95_deg": 0.0,
            "beta_gap_max_deg": 0.0,
            "static_representation_gate_pass": True,
        }
        return report, pd.DataFrame(columns=columns)
    left = np.asarray([pair[0] for pair in pairs], dtype=int)
    right = np.asarray([pair[1] for pair in pairs], dtype=int)
    gap = _beta_rms_gap_deg(beta[left], beta[right])
    distance = np.linalg.norm(xyz[left] - xyz[right], axis=1) * 1000.0
    conflict = gap > float(beta_gap_threshold_deg)
    conflicts = pd.DataFrame(
        {
            "left_sample_id": samples[left][conflict],
            "right_sample_id": samples[right][conflict],
            "left_family_id": families[left][conflict],
            "right_family_id": families[right][conflict],
            "xyz_distance_mm": distance[conflict],
            "beta_gap_rms_deg": gap[conflict],
        }
    )
    return (
        {
            "xyz_radius_mm": float(xyz_radius_mm),
            "beta_gap_threshold_deg": float(beta_gap_threshold_deg),
            "cross_family_pair_count": int(len(pairs)),
            "conflict_pair_count": int(np.count_nonzero(conflict)),
            "conflict_pair_ratio": float(np.mean(conflict)),
            "beta_gap_p95_deg": float(np.percentile(gap, 95)),
            "beta_gap_max_deg": float(np.max(gap)),
            "static_representation_gate_pass": bool(not np.any(conflict)),
        },
        conflicts,
    )


def select_student_representation(
    report: Mapping[str, Any], *, thresholds: Mapping[str, Any] | None = None
) -> str:
    """Apply the frozen static → chart → stateful → stop decision ladder."""

    if int(report.get("conflict_pair_count", 1)) == 0:
        return "static"
    limits = thresholds or {
        "chart_count_range": (2, 4),
        "chart_repeat_ari_min": 0.99,
        "chart_xyz_macro_f1_min": 0.98,
        "ambiguous_voxel_max": 0,
    }
    chart_min, chart_max = (int(value) for value in limits["chart_count_range"])
    chart_pass = bool(
        chart_min <= int(report.get("stable_chart_count", 0)) <= chart_max
        and float(report.get("chart_repeat_ari", -math.inf))
        >= float(limits["chart_repeat_ari_min"])
        and float(report.get("chart_xyz_macro_f1", -math.inf))
        >= float(limits["chart_xyz_macro_f1_min"])
        and int(report.get("ambiguous_voxel_count", 1))
        <= int(limits["ambiguous_voxel_max"])
    )
    if chart_pass:
        return "chart_expert"
    if bool(report.get("previous_beta_resolves_conflicts", False)):
        return "stateful"
    return "stop"
