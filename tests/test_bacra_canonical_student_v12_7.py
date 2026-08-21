from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import run_bacra_v12 as v12
import run_bacra_v12_7_canonical_student as runner
from quasi_exp.teacher.bacra_canonical_student import (
    exploratory_student_admission_checks,
    project_toward_canonical_beta,
    select_pilot_candidate,
)
from quasi_exp.teacher.student_tracking_tf import (
    StudentGeometry,
    build_static_model,
    student_loss_terms,
)


def _geometry() -> StudentGeometry:
    bounds = np.deg2rad(
        np.asarray(
            [
                [-5.0, 5.0],
                [-5.0, 5.0],
                [-10.0, 10.0],
                [-10.0, 10.0],
                [-15.0, 15.0],
                [-15.0, 15.0],
            ]
        )
    )
    return StudentGeometry(
        lengths_m=np.ones(31),
        p_end_local_m=np.asarray([0.0, 0.0, 0.0, 1.0]),
        theta_sign=-1.0,
        beta_bounds_rad=bounds,
    )


def _single_joint_loss(
    joint: int, scale: float | list[float] | None
) -> float:
    import tensorflow as tf

    prediction = np.zeros((1, 6), dtype=np.float32)
    prediction[0, joint] = math.radians(1.0)
    packed = np.zeros((1, 9), dtype=np.float32)
    return float(
        student_loss_terms(
            packed,
            tf.constant(prediction),
            geometry=_geometry(),
            lambda_fk=0.0,
            beta_loss_scale_deg=scale,
        )["beta"].numpy()
    )


def test_uniform_degree_loss_penalizes_equal_joint_errors_equally() -> None:
    losses = [_single_joint_loss(joint, 1.0) for joint in range(6)]
    np.testing.assert_allclose(losses, losses[0], rtol=1.0e-6)


def test_legacy_loss_keeps_span_normalization() -> None:
    beta1 = _single_joint_loss(0, None)
    beta6 = _single_joint_loss(5, None)
    assert beta1 > beta6
    assert math.isclose(beta1 / beta6, 9.0, rel_tol=1.0e-5)


def test_beta5_specific_scale_increases_only_beta5_penalty() -> None:
    scales = [1.0, 1.0, 1.0, 1.0, 0.75, 1.0]
    beta1 = _single_joint_loss(0, scales)
    beta5 = _single_joint_loss(4, scales)
    beta6 = _single_joint_loss(5, scales)

    assert math.isclose(beta1, beta6, rel_tol=1.0e-6)
    assert beta5 > beta1


def test_beta5_specific_head_preserves_six_joint_output() -> None:
    model = build_static_model(
        np.zeros((8, 3), dtype=np.float32),
        geometry=_geometry(),
        output_mode="tanh",
        beta5_head_units=(16, 8),
    )

    assert model.output_shape == (None, 6)
    assert model.get_layer("beta5_dense_0").units == 16
    assert model.get_layer("beta5_dense_1").units == 8
    assert model.get_layer("beta5_latent").units == 1


def test_canonical_projection_keeps_only_local_fk_nullspace_delta() -> None:
    geometry = _geometry()
    base = np.zeros((1, 6), dtype=float)
    canonical = np.deg2rad(
        np.asarray([[1.0, -0.5, 0.8, -0.7, 1.2, -0.9]])
    )

    projected = project_toward_canonical_beta(
        base,
        canonical,
        geometry=geometry,
        alpha=0.5,
    )

    assert projected.shape == (1, 6)
    np.testing.assert_allclose(
        project_toward_canonical_beta(
            base,
            canonical,
            geometry=geometry,
            alpha=0.0,
        ),
        base,
    )
    eps = 1.0e-4
    jacobian = np.empty((3, 6), dtype=float)
    from quasi_exp.teacher.bacra_canonical_student import _fk

    for joint in range(6):
        step = np.zeros(6)
        step[joint] = eps
        jacobian[:, joint] = (
            _fk(step, geometry) - _fk(-step, geometry)
        ) / (2.0 * eps)
    np.testing.assert_allclose(
        jacobian @ (projected[0] - base[0]),
        0.0,
        atol=1.0e-10,
    )


def test_exploratory_admission_uses_only_the_four_frozen_limits() -> None:
    passing = {
        "validation_beta_abs_p95_by_joint_deg": [1.99] * 6,
        "validation_fk_p95_mm": 4.99,
        "validation_fk_max_mm": 9.99,
        "predicted_minimum_joint_margin_deg": 1.51,
        "validation_beta_rms_p95_deg": 99.0,
    }
    limits = {
        "joint_abs_p95_max_deg": 2.0,
        "fk_p95_max_mm": 5.0,
        "fk_max_mm": 10.0,
        "minimum_joint_margin_min_deg": 1.5,
    }

    assert all(
        exploratory_student_admission_checks(passing, **limits).values()
    )
    for key, value in (
        ("validation_beta_abs_p95_by_joint_deg", [2.0] * 6),
        ("validation_fk_p95_mm", 5.0),
        ("validation_fk_max_mm", 10.0),
        ("predicted_minimum_joint_margin_deg", 1.5),
    ):
        failed = {**passing, key: value}
        assert not all(
            exploratory_student_admission_checks(
                failed, **limits
            ).values()
        )


def test_v12_7_config_and_pilot_selection_are_deterministic() -> None:
    config = v12.load_protocol_config(
        ROOT / "configs/bacra_v12_7_canonical_student.yaml", "pilot"
    )
    assert config["protocol_id"] == runner.PROTOCOL_ID
    assert config["canonical_student"]["lambda_fk_candidates"] == [
        0.0,
        0.1,
        1.0,
    ]
    assert [
        runner._lambda_label(value)
        for value in config["canonical_student"]["lambda_fk_candidates"]
    ] == ["0", "0p1", "1"]

    common = {
        "validation_beta_rms_max_deg": 1.8,
        "validation_beta_abs_p95_by_joint_deg": [1.0] * 6,
        "validation_fk_p95_mm": 2.0,
        "validation_fk_max_mm": 4.0,
    }
    selected, evaluated = select_pilot_candidate(
        [
            {
                **common,
                "lambda_fk": 1.0,
                "validation_beta_rms_p95_deg": 1.2,
            },
            {
                **common,
                "lambda_fk": 0.1,
                "validation_beta_rms_p95_deg": 0.9,
            },
            {
                **common,
                "lambda_fk": 0.0,
                "validation_beta_rms_p95_deg": 0.9,
            },
        ],
        fk_p95_max_mm=5.0,
        fk_max_mm=10.0,
        beta_rms_p95_max_deg=1.5,
        joint_abs_p95_max_deg=2.0,
    )
    assert len(evaluated) == 3
    assert selected is not None
    assert selected["lambda_fk"] == 0.0
    assert selected["canonical_beta_target_pass"] is True
