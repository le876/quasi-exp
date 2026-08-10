from __future__ import annotations

import numpy as np
import pandas as pd

from quasi_exp.teacher.audit import (
    audit_label_consistency,
    compare_aligned_trajectories,
    recommend_student_representation,
)


def _frame(offset_deg: float = 0.0) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "target_x_m": [0.0, 0.01, 0.02],
            "target_y_m": [0.0, 0.0, 0.0],
            "target_z_m": [0.0, 0.0, 0.0],
            "trajectory_id": ["t1", "t1", "t1"],
            "chart_id": [0, 0, 1],
        }
    )
    for index in range(1, 7):
        frame[f"teacher_beta{index}_rad"] = np.deg2rad(
            np.asarray([0.0, 0.1, 0.2]) + offset_deg
        )
    return frame


def test_aligned_comparison_reports_beta_rms_p95_and_max() -> None:
    report = compare_aligned_trajectories(_frame(), _frame(0.2))
    assert np.isclose(report["beta_gap_rms_p95_deg"], 0.2)
    assert np.isclose(report["beta_gap_rms_max_deg"], 0.2)
    assert report["aligned_count"] == 3


def test_consistency_audit_excludes_same_trajectory_pairs() -> None:
    first = _frame()
    second = _frame(0.5)
    second["trajectory_id"] = "t2"
    combined = pd.concat([first, second], ignore_index=True)

    report = audit_label_consistency(combined, xyz_radius_mm=5.0)

    assert report["cross_trajectory_pair_count"] == 3
    assert np.isclose(report["cross_trajectory_beta_gap_p95_deg"], 0.5)
    assert recommend_student_representation(report) == "static"


def test_representation_recommendation_escalates_from_static_to_chart_or_stateful() -> None:
    assert recommend_student_representation(
        {"cross_trajectory_beta_gap_p95_deg": 2.0, "stable_chart_cluster_count": 2}
    ) == "chart_expert"
    assert recommend_student_representation(
        {"cross_trajectory_beta_gap_p95_deg": 2.0, "stable_chart_cluster_count": 0}
    ) == "stateful"
