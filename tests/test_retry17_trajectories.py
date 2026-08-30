from __future__ import annotations

import numpy as np

from quasi_exp.teacher.retry17_trajectories import (
    PLANE_SPECS,
    frozen_shape_objective_registry,
    plane_basis,
    unit_shape,
)


def test_shape_objective_registry_is_frozen_three_by_three() -> None:
    registry = frozen_shape_objective_registry()
    assert len(registry) == 9
    assert registry.groupby("shape_class").size().to_dict() == {
        "ellipse": 3,
        "rounded_rectangle": 3,
        "rounded_star": 3,
    }
    assert registry["minimum_axial_span_mm"].eq(150.0).all()
    assert registry["maximum_support_distance_mm"].eq(30.0).all()


def test_registered_planes_are_orthonormal_and_tilted() -> None:
    assert len(PLANE_SPECS) == 12
    for spec in PLANE_SPECS:
        normal, e1, e2 = plane_basis(spec["alpha_deg"], spec["psi_deg"])
        matrix = np.stack([normal, e1, e2])
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
        assert 0.0 < abs(normal[0]) < 1.0


def test_ellipse_rectangle_star_are_closed_finite_unit_curves() -> None:
    for shape in ("ellipse", "rounded_rectangle", "rounded_star"):
        points = unit_shape(shape, 720)
        assert points.shape == (720, 2)
        assert np.isfinite(points).all()
        assert float(np.max(np.linalg.norm(np.diff(np.vstack([points, points[:1]]), axis=0), axis=1))) < 0.1
    star = unit_shape("rounded_star", 720)
    assert float(np.max(np.linalg.norm(star, axis=1))) > 0.9
    assert float(np.min(np.linalg.norm(star, axis=1))) < 0.55

