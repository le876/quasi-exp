"""Consistency and invariance audits that select the student representation."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


BETA_COLUMNS = tuple(f"teacher_beta{index}_rad" for index in range(1, 7))
XYZ_COLUMNS = ("target_x_m", "target_y_m", "target_z_m")


def _beta_gap_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.sqrt(np.mean(np.square(left - right), axis=1)))


def compare_aligned_trajectories(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    key: str = "sample_id",
) -> dict[str, float | int]:
    required = {key, *BETA_COLUMNS}
    for name, frame in (("reference", reference), ("candidate", candidate)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if frame[key].duplicated().any():
            raise ValueError(f"{name} alignment key must be unique")
    merged = reference[[key, *BETA_COLUMNS]].merge(
        candidate[[key, *BETA_COLUMNS]], on=key, suffixes=("_left", "_right"), validate="one_to_one"
    )
    if merged.empty:
        raise ValueError("aligned trajectories have no shared samples")
    left = merged[[f"{column}_left" for column in BETA_COLUMNS]].to_numpy(dtype=float)
    right = merged[[f"{column}_right" for column in BETA_COLUMNS]].to_numpy(dtype=float)
    gap = _beta_gap_deg(left, right)
    return {
        "aligned_count": len(merged),
        "beta_gap_rms_p95_deg": float(np.percentile(gap, 95)),
        "beta_gap_rms_max_deg": float(np.max(gap)),
    }


def audit_label_consistency(
    frame: pd.DataFrame,
    *,
    xyz_radius_mm: float = 5.0,
) -> dict[str, Any]:
    required = {"trajectory_id", "chart_id", *XYZ_COLUMNS, *BETA_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"teacher dataset is missing columns: {sorted(missing)}")
    if xyz_radius_mm <= 0.0:
        raise ValueError("xyz_radius_mm must be positive")
    xyz = frame.loc[:, XYZ_COLUMNS].to_numpy(dtype=float)
    beta = frame.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
    trajectories = frame["trajectory_id"].astype(str).to_numpy()
    charts = frame["chart_id"].astype(str).to_numpy()
    tree = cKDTree(xyz)
    pairs: list[tuple[int, int]] = []
    for left, neighbours in enumerate(tree.query_ball_point(xyz, r=float(xyz_radius_mm) / 1000.0)):
        for right in neighbours:
            if right <= left or trajectories[left] == trajectories[right]:
                continue
            pairs.append((left, right))
    if not pairs:
        return {
            "xyz_radius_mm": float(xyz_radius_mm),
            "cross_trajectory_pair_count": 0,
            "cross_trajectory_beta_gap_p95_deg": math.inf,
            "cross_trajectory_beta_gap_max_deg": math.inf,
            "stable_chart_cluster_count": 0,
            "audit_pass": False,
        }
    left_indices = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    right_indices = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    gaps = _beta_gap_deg(beta[left_indices], beta[right_indices])
    conflict = gaps > 1.0
    stable_chart_pairs = {
        tuple(sorted((charts[left], charts[right])))
        for left, right, is_conflict in zip(left_indices, right_indices, conflict)
        if is_conflict and charts[left] != charts[right]
    }
    return {
        "xyz_radius_mm": float(xyz_radius_mm),
        "cross_trajectory_pair_count": len(pairs),
        "cross_trajectory_beta_gap_p95_deg": float(np.percentile(gaps, 95)),
        "cross_trajectory_beta_gap_max_deg": float(np.max(gaps)),
        "branch_conflict_ratio": float(np.mean(conflict)),
        "stable_chart_cluster_count": len(stable_chart_pairs),
        "audit_pass": bool(float(np.percentile(gaps, 95)) <= 1.0),
    }


def recommend_student_representation(report: Mapping[str, Any]) -> str:
    gap = float(report.get("cross_trajectory_beta_gap_p95_deg", math.inf))
    if gap <= 1.0:
        return "static"
    if int(report.get("stable_chart_cluster_count", 0)) > 0:
        return "chart_expert"
    return "stateful"
