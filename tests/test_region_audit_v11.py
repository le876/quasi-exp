from __future__ import annotations

import numpy as np
import pandas as pd

from quasi_exp.teacher.region_audit import (
    audit_cartesian_coverage,
    audit_cross_family_conflicts,
    evaluate_teacher_surface_gate,
    select_student_representation,
)


def _frame(*, conflicting: bool) -> pd.DataFrame:
    rows = []
    xyz = np.asarray([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
    for family_index, family_id in enumerate(("F0", "F1")):
        for phase, point in enumerate(xyz):
            beta = np.zeros(6)
            if conflicting and family_index == 1:
                beta[0] = np.deg2rad(3.0)
            row = {
                "family_id": family_id,
                "trajectory_id": f"{family_id}:node0",
                "sample_id": f"{family_id}:{phase}",
                "phase_idx": phase,
                "target_x_m": point[0],
                "target_y_m": point[1],
                "target_z_m": point[2],
                "chart_id": family_index if conflicting else 0,
            }
            row.update({f"teacher_beta{i + 1}_rad": beta[i] for i in range(6)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_cross_family_conflict_audit_distinguishes_close_xyz_from_label_conflict() -> None:
    clean_report, clean_pairs = audit_cross_family_conflicts(_frame(conflicting=False))
    conflict_report, conflict_pairs = audit_cross_family_conflicts(_frame(conflicting=True))

    assert clean_report["cross_family_pair_count"] > 0
    assert clean_report["conflict_pair_count"] == 0
    assert clean_pairs.empty
    assert conflict_report["conflict_pair_count"] > 0
    assert not conflict_pairs.empty
    assert conflict_pairs["beta_gap_rms_deg"].min() > 1.0


def test_representation_selection_obeys_static_chart_stateful_stop_order() -> None:
    assert select_student_representation({"conflict_pair_count": 0}) == "static"
    assert select_student_representation(
        {
            "conflict_pair_count": 10,
            "stable_chart_count": 3,
            "chart_repeat_ari": 0.995,
            "chart_xyz_macro_f1": 0.99,
            "ambiguous_voxel_count": 0,
        }
    ) == "chart_expert"
    assert select_student_representation(
        {"conflict_pair_count": 10, "previous_beta_resolves_conflicts": True}
    ) == "stateful"
    assert select_student_representation({"conflict_pair_count": 10}) == "stop"


def test_coverage_uses_an_independent_probe_set() -> None:
    reference = np.asarray([[0.0, 0.0, 0.0], [0.004, 0.0, 0.0]])
    probe = np.asarray([[0.001, 0.0, 0.0], [0.003, 0.0, 0.0]])

    report = audit_cartesian_coverage(reference, probe, p95_limit_mm=3.0, max_limit_mm=5.0)

    assert report["nearest_distance_p95_mm"] == 1.0
    assert report["nearest_distance_max_mm"] == 1.0
    assert report["coverage_gate_pass"] is True


def test_surface_gate_is_strict_and_exposes_raw_boolean_checks() -> None:
    metrics = {
        "success_rate": 1.0,
        "residual_p95_mm": 0.9,
        "residual_max_mm": 2.9,
        "joint_margin_min_deg": 1.5,
        "phase_beta_rms_p95_deg": 0.9,
        "acceleration_beta_rms_p95_deg": 0.4,
        "seam_beta_rms_max_deg": 0.8,
        "surface_edge_beta_rms_p95_deg": 0.9,
        "surface_laplacian_beta_rms_p95_deg": 0.8,
        "surface_block_update_rms_max_deg": 0.04,
    }

    report = evaluate_teacher_surface_gate(metrics)

    assert report["gate_pass"] is True
    assert all(type(value) is bool for value in report["checks"].values())
    failed = evaluate_teacher_surface_gate({**metrics, "success_rate": 0.999})
    assert failed["gate_pass"] is False
    assert failed["checks"]["all_rows_success"] is False


def test_surface_success_rate_can_be_relaxed_by_a_frozen_minimum_threshold() -> None:
    metrics = {
        "success_rate": 0.75,
        "residual_p95_mm": 1.8,
        "residual_max_mm": 5.8,
        "joint_margin_min_deg": 0.8,
        "phase_beta_rms_p95_deg": 1.8,
        "acceleration_beta_rms_p95_deg": 0.8,
        "seam_beta_rms_max_deg": 1.8,
        "surface_edge_beta_rms_p95_deg": 1.8,
        "surface_laplacian_beta_rms_p95_deg": 1.8,
        "surface_block_update_rms_max_deg": 0.08,
    }
    thresholds = {
        "success_rate": 0.5,
        "residual_p95_mm": 2.0,
        "residual_max_mm": 6.0,
        "joint_margin_min_deg": 0.75,
        "phase_beta_rms_p95_deg": 2.0,
        "acceleration_beta_rms_p95_deg": 1.0,
        "seam_beta_rms_max_deg": 2.0,
        "surface_edge_beta_rms_p95_deg": 2.0,
        "surface_laplacian_beta_rms_p95_deg": 2.0,
        "surface_block_update_rms_max_deg": 0.1,
    }

    report = evaluate_teacher_surface_gate(metrics, thresholds=thresholds)

    assert report["gate_pass"] is True
    assert report["checks"]["all_rows_success"] is True
