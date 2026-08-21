from __future__ import annotations

import numpy as np
import pandas as pd

from quasi_exp.teacher.bacra_geometry_holdout import (
    LoopReferencePolicy,
    cyclic_beta_metrics,
    evaluate_student_loop,
    generate_geometry_holdout,
    solve_loop_reference,
)
from quasi_exp.teacher.canonical import TeacherPolicy
from quasi_exp.teacher.dense_chart_sampling import BETA_COLUMNS, XYZ_COLUMNS
from quasi_exp.teacher.region import EllipseFamilySpec


class LinearEnvironment:
    bounds = np.asarray([[-2.0, 2.0]] * 6, dtype=float)

    def fk(self, beta_rad: np.ndarray) -> np.ndarray:
        beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
        return beta[:, :3]

    def jacobian(self, beta_rad: np.ndarray) -> np.ndarray:
        del beta_rad
        return np.column_stack([np.eye(3), np.zeros((3, 3))])


def _anchor() -> EllipseFamilySpec:
    return EllipseFamilySpec(
        family_id="anchor",
        center_m=np.zeros(3),
        major_direction=np.asarray([1.0, 0.0, 0.0]),
        minor_direction=np.asarray([0.0, 1.0, 0.0]),
        major_semiaxis_m=0.10,
        minor_semiaxis_m=0.05,
    )


def _groups() -> list[dict[str, object]]:
    return [
        {
            "group_id": "core_interpolation",
            "count": 4,
            "seed": 20260744,
            "major_semiaxis_m": [0.09, 0.10],
            "axis_ratio": [0.45, 0.55],
            "center_q1_offset_mm": [-1.0, 1.0],
            "center_q2_offset_mm": [-1.0, 1.0],
            "center_normal_offset_mm": [-1.0, 1.0],
            "tilt_deg": [-1.0, 1.0],
        },
        {
            "group_id": "historical_range_stress",
            "count": 4,
            "seed": 20260745,
            "major_semiaxis_m": [0.08, 0.10],
            "axis_ratio": [0.40, 0.60],
            "center_q1_offset_mm": [-2.0, 2.0],
            "center_q2_offset_mm": [-2.0, 2.0],
            "center_normal_offset_mm": [-2.0, 2.0],
            "tilt_deg": [-2.0, 2.0],
        },
    ]


def _chart() -> pd.DataFrame:
    axis = np.linspace(-0.16, 0.16, 7)
    xyz = np.asarray(
        [(x, y, z) for x in axis for y in axis for z in axis],
        dtype=float,
    )
    beta = np.column_stack([xyz, np.zeros((len(xyz), 3))])
    return pd.DataFrame(
        {
            **{
                name: xyz[:, index]
                for index, name in enumerate(XYZ_COLUMNS)
            },
            **{
                name: beta[:, index]
                for index, name in enumerate(BETA_COLUMNS)
            },
        }
    )


def test_geometry_holdout_inventory_is_deterministic_and_complete() -> None:
    first = generate_geometry_holdout(_anchor(), _groups())
    second = generate_geometry_holdout(_anchor(), _groups())

    pd.testing.assert_frame_equal(first, second)
    assert list(first["family_id"]) == [
        *(f"core_interpolation_{index:02d}" for index in range(4)),
        *(
            f"historical_range_stress_{index:02d}"
            for index in range(4)
        ),
    ]
    assert first.groupby("group_id").size().to_dict() == {
        "core_interpolation": 4,
        "historical_range_stress": 4,
    }


def test_cyclic_metrics_detect_internal_and_seam_jumps() -> None:
    beta = np.zeros((40, 6), dtype=float)
    beta[20:, 3] = np.deg2rad(3.0)
    metrics = cyclic_beta_metrics(beta)

    assert metrics["phase_beta_rms_max_deg"] > 1.0
    assert metrics["acceleration_beta_rms_p95_deg"] > 0.0
    assert metrics["seam_beta_rms_deg"] > 1.0


def test_loop_reference_uses_complete_chart_and_bidirectional_paths() -> None:
    policy = LoopReferencePolicy(
        phase_count=36,
        max_tetrahedron_edge_mm=200.0,
        target_support_max_mm=100.0,
        residual_p95_mm=1.0,
        residual_max_mm=3.0,
        joint_margin_min_deg=1.5,
        phase_beta_rms_p95_deg=10.0,
        phase_beta_rms_max_deg=10.0,
        acceleration_beta_rms_p95_deg=10.0,
        seam_beta_rms_deg=10.0,
        direction_gap_p95_deg=1.0,
        direction_gap_max_deg=1.0,
        teacher_policy=TeacherPolicy(
            tracking_tolerance_mm=1.0,
            safe_joint_margin_deg=1.5,
            max_corrector_iterations=20,
        ),
    )
    frame, report = solve_loop_reference(
        LinearEnvironment(),
        _chart(),
        _anchor(),
        group_id="core_interpolation",
        policy=policy,
    )

    assert len(frame) == 36
    assert frame["chart_supported"].all()
    assert frame["reference_corrector_success"].all()
    assert frame["forward_success"].all()
    assert frame["reverse_success"].all()
    assert report["gate_pass"] is True


def test_loop_reference_connects_from_chart_without_full_tetrahedron_support() -> None:
    chart = _chart()
    for column in XYZ_COLUMNS:
        chart[column] *= 0.25
    for column in BETA_COLUMNS[:3]:
        chart[column] *= 0.25
    policy = LoopReferencePolicy(
        phase_count=36,
        max_tetrahedron_edge_mm=40.0,
        target_support_max_mm=20.0,
        continuation_step_mm=5.0,
        residual_p95_mm=1.0,
        residual_max_mm=3.0,
        joint_margin_min_deg=1.5,
        phase_beta_rms_p95_deg=10.0,
        phase_beta_rms_max_deg=10.0,
        acceleration_beta_rms_p95_deg=10.0,
        seam_beta_rms_deg=10.0,
        direction_gap_p95_deg=1.0,
        direction_gap_max_deg=1.0,
        teacher_policy=TeacherPolicy(
            tracking_tolerance_mm=1.0,
            safe_joint_margin_deg=1.5,
            max_corrector_iterations=20,
        ),
    )

    frame, report = solve_loop_reference(
        LinearEnvironment(),
        chart,
        _anchor(),
        group_id="core_interpolation",
        policy=policy,
    )

    assert not frame["chart_supported"].all()
    assert report["chart_support_is_diagnostic_only"] is True
    assert report["root_connection_success"] is True
    assert frame["reference_corrector_success"].all()
    assert report["gate_pass"] is True


def test_student_admission_keeps_strict_two_degree_boundary() -> None:
    count = 40
    reference = pd.DataFrame(
        {
            "family_id": ["core_interpolation_00"] * count,
            "group_id": ["core_interpolation"] * count,
            "phase_idx": np.arange(count),
            **{name: np.zeros(count) for name in XYZ_COLUMNS},
            **{name: np.zeros(count) for name in BETA_COLUMNS},
        }
    )
    prediction = np.zeros((count, 6), dtype=float)
    prediction[:, 3] = np.deg2rad(2.0)
    report, diagnostics = evaluate_student_loop(
        LinearEnvironment(),
        reference,
        prediction,
        teacher_reference_gate_pass=True,
        admission={
            "joint_abs_p95_max_deg": 2.0,
            "fk_p95_max_mm": 5.0,
            "fk_max_mm": 10.0,
            "minimum_joint_margin_min_deg": 1.5,
        },
        trajectory_limits={
            "phase_beta_rms_p95_deg": 1.0,
            "phase_beta_rms_max_deg": 2.0,
            "acceleration_beta_rms_p95_deg": 0.5,
            "seam_beta_rms_deg": 1.0,
        },
    )

    assert report["checks"]["all_joint_abs_p95_below_limit"] is False
    assert report["checks"]["fk_p95_below_limit"] is True
    assert report["checks"]["trajectory_continuity"] is True
    assert report["student_family_gate_pass"] is False
    assert len(diagnostics) == count
