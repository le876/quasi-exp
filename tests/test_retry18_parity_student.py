from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quasi_exp.teacher.region_growth import JACOBIAN_COLUMNS
from quasi_exp.teacher.retry12_symmetry import BETA_COLUMNS, XYZ_COLUMNS
from quasi_exp.teacher.retry18_parity_student import (
    ParityStudentConfig,
    build_parity_smooth_student,
    data_driven_radial_scale_mm,
    load_parity_student,
    mirror_beta,
    parity_loss_terms,
    save_parity_student,
    seam_boundary_metrics,
)
from quasi_exp.teacher.student_tracking_tf import StudentGeometry


def _geometry() -> StudentGeometry:
    bounds = np.deg2rad(np.asarray([[-5, 5], [-5, 5], [-10, 10], [-10, 10], [-15, 15], [-15, 15]], dtype=np.float32))
    return StudentGeometry(lengths_m=np.ones(31, dtype=np.float32), p_end_local_m=np.asarray([0, 0, 0, 1], dtype=np.float32), theta_sign=-1.0, beta_bounds_rad=bounds)


def _points() -> np.ndarray:
    return np.asarray([[1.19, 0.04, 0.06], [1.17, 0.09, 0.02], [1.15, 0.01, 0.11], [1.20, 0.00, 0.08]], dtype=np.float32)


def test_data_driven_radial_scale_uses_frozen_supervision() -> None:
    frame = pd.DataFrame({"y_m": [0.0, 0.3], "z_m": [0.0, 0.4]})
    assert data_driven_radial_scale_mm(frame, quantile=1.0, safety_factor=1.05) == pytest.approx(525.0)


def test_parity_model_is_exactly_equivariant_smooth_and_bounded() -> None:
    geometry = _geometry()
    model = build_parity_smooth_student(train_xyz_m=_points(), zero_x_m=1.2, geometry=geometry, config=ParityStudentConfig(hidden_units=(8, 8), maximum_steps=2))
    points = _points()
    base = np.asarray(model(points, training=False))
    mirror_y_points = points.copy(); mirror_y_points[:, 1] *= -1
    mirror_z_points = points.copy(); mirror_z_points[:, 2] *= -1
    assert np.allclose(model(mirror_y_points, training=False), mirror_beta(base, mirror_y=True), atol=2e-7)
    assert np.allclose(model(mirror_z_points, training=False), mirror_beta(base, mirror_z=True), atol=2e-7)
    seam = seam_boundary_metrics(model, points, seam="y", epsilon_mm=1.0)
    assert seam["odd_zero_maximum_deg"] <= 1.0e-7
    assert seam["even_normal_slope_maximum_deg_per_mm"] <= 1.0e-6
    bounds = geometry.beta_bounds_rad
    assert np.logical_and(base >= bounds[:, 0], base <= bounds[:, 1]).all()


def test_parity_model_round_trips_through_keras(tmp_path) -> None:
    geometry = _geometry()
    model = build_parity_smooth_student(train_xyz_m=_points(), zero_x_m=1.2, geometry=geometry, config=ParityStudentConfig(hidden_units=(8,), maximum_steps=2))
    expected = np.asarray(model(_points(), training=False))
    path = tmp_path / "parity.keras"
    save_parity_student(model, path)
    loaded = load_parity_student(path)
    assert np.allclose(loaded(_points(), training=False), expected, atol=1.0e-7)


def test_zero_jacobian_lambda_reduces_to_weighted_beta_loss() -> None:
    geometry = _geometry()
    model = build_parity_smooth_student(train_xyz_m=_points(), zero_x_m=1.2, geometry=geometry, config=ParityStudentConfig(hidden_units=(8,), maximum_steps=2))
    xyz = _points()
    beta = np.zeros((len(xyz), 6), dtype=np.float32)
    jacobian = np.ones((len(xyz), 3, 6), dtype=np.float32)
    terms = parity_loss_terms(model=model, xyz_m=xyz, beta_true=beta, jacobian_true=jacobian, sample_weight=np.ones(len(xyz), dtype=np.float32), beta_coordinate_weights=(4, 4, 2, 2, 1, 1), jacobian_lambda=0.0, training=False)
    assert np.isclose(float(terms["objective"].numpy()), float(terms["beta"].numpy()))
