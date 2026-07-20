from __future__ import annotations

import numpy as np

from quasi_exp.teacher.canonical import TeacherPolicy, TeacherVariant
from quasi_exp.teacher.region import (
    CanonicalRegionTeacher,
    EllipseFamilySpec,
    TubeCrossSection,
)


class _AffineEnvironment:
    bounds = np.tile(np.asarray([-1.0, 1.0]), (6, 1))

    def fk(self, beta: np.ndarray) -> np.ndarray:
        return np.asarray(beta, dtype=float).reshape(-1, 6)[:, :3]

    def theta(self, beta: np.ndarray) -> np.ndarray:
        return np.zeros((len(np.asarray(beta).reshape(-1, 6)), 30), dtype=float)

    def jacobian(self, _beta: np.ndarray) -> np.ndarray:
        jacobian = np.zeros((3, 6), dtype=float)
        jacobian[:, :3] = np.eye(3)
        return jacobian


class _RedundantEnvironment:
    bounds = np.deg2rad(np.tile(np.asarray([-5.0, 5.0]), (6, 1)))

    def fk(self, beta: np.ndarray) -> np.ndarray:
        values = np.asarray(beta, dtype=float).reshape(-1, 6)
        return values[:, :3] + values[:, 3:]

    def theta(self, beta: np.ndarray) -> np.ndarray:
        return np.zeros((len(np.asarray(beta).reshape(-1, 6)), 30), dtype=float)

    def jacobian(self, _beta: np.ndarray) -> np.ndarray:
        return np.hstack([np.eye(3), np.eye(3)])


def _family() -> EllipseFamilySpec:
    return EllipseFamilySpec(
        family_id="F000",
        center_m=np.asarray([0.2, -0.1, 0.3]),
        major_direction=np.asarray([1.0, 0.0, 0.0]),
        minor_direction=np.asarray([0.0, 1.0, 0.0]),
        major_semiaxis_m=0.1,
        minor_semiaxis_m=0.05,
    )


def test_family_generates_exact_orthogonal_ellipse_and_tube_targets() -> None:
    family = _family()

    centerline = family.centerline(phase_count=4)
    target = family.tube_targets(
        phase_count=4,
        normalized_offsets=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, -1.0]]),
        radial_radius_mm=2.0,
        plane_radius_mm=3.0,
    )

    assert np.allclose(
        centerline,
        np.asarray(
            [
                [0.3, -0.1, 0.3],
                [0.2, -0.05, 0.3],
                [0.1, -0.1, 0.3],
                [0.2, -0.15, 0.3],
            ]
        ),
    )
    assert target.shape == (3, 4, 3)
    assert np.allclose(target[0], centerline)
    assert np.isclose(np.linalg.norm(target[1, 0] - centerline[0]), 0.002)
    assert np.allclose(target[2] - centerline, np.asarray([0.0, 0.0, -0.003]))
    assert np.allclose(family.plane_normal, np.asarray([0.0, 0.0, 1.0]))


def test_cross_section_master_design_is_deterministic_nested_and_connected() -> None:
    design = TubeCrossSection.master(seed=20260722)
    repeated = TubeCrossSection.master(seed=20260722)

    assert len(design.points) == 81
    assert design.fingerprint == repeated.fingerprint
    assert np.allclose(design.points, repeated.points)
    assert np.allclose(design.points[0], 0.0)
    assert np.count_nonzero(np.isclose(np.linalg.norm(design.points, axis=1), 1.0)) == 16
    for count in (9, 25, 49, 81):
        prefix = design.prefix(count)
        assert len(prefix.points) == count
        assert prefix.parent_fingerprint == design.fingerprint
        assert len(prefix.edges) >= count - 1
        touched = set(np.asarray(prefix.edges).reshape(-1).tolist())
        assert touched == set(range(count))


def test_surface_solver_returns_complete_coupled_surface_on_affine_environment() -> None:
    environment = _AffineEnvironment()
    teacher = CanonicalRegionTeacher(environment)
    cross_section = TubeCrossSection.master(seed=20260722).prefix(9)
    policy = TeacherPolicy(
        variant=TeacherVariant.T1,
        candidate_budget=2,
        max_corrector_iterations=20,
        tracking_tolerance_mm=0.01,
    )

    surface = teacher.solve_surface(
        family=_family(),
        cross_section=cross_section,
        phase_count=12,
        radial_radius_mm=1.0,
        plane_radius_mm=1.0,
        policy=policy,
        root_beta=np.zeros(6),
        sweep_directions=("outward", "inward"),
    )

    assert surface.complete
    assert surface.beta_rad.shape == (9, 12, 6)
    assert surface.target_xyz_m.shape == (9, 12, 3)
    assert np.max(np.linalg.norm(surface.achieved_xyz_m - surface.target_xyz_m, axis=2)) <= 1e-5
    assert surface.provenance["surface_strategy"] == "delaunay_neighbor_anchor_block_coordinate_v1"
    assert surface.provenance["cross_section_fingerprint"] == cross_section.fingerprint
    assert surface.metrics["success_rate"] == 1.0
    assert surface.metrics["surface_edge_beta_rms_p95_deg"] < 0.1


def test_v11_corrector_can_use_nullspace_to_recover_safe_joint_margin() -> None:
    environment = _RedundantEnvironment()
    family = EllipseFamilySpec(
        family_id="redundant",
        center_m=np.zeros(3),
        major_direction=np.asarray([1.0, 0.0, 0.0]),
        minor_direction=np.asarray([0.0, 1.0, 0.0]),
        major_semiaxis_m=0.01,
        minor_semiaxis_m=0.005,
    )
    seed = np.zeros((12, 6))
    seed[:, :3] = np.deg2rad(4.9)
    seed[:, 3:] = family.centerline(phase_count=12) - seed[:, :3]
    policy = TeacherPolicy(
        variant=TeacherVariant.T1,
        candidate_budget=1,
        tracking_tolerance_mm=0.01,
        safe_joint_margin_deg=2.0,
        safe_margin_repulsion_step_deg=0.5,
        max_corrector_iterations=80,
    )

    surface = CanonicalRegionTeacher(environment).solve_surface(
        family=family,
        cross_section=TubeCrossSection.master(seed=3).prefix(9),
        phase_count=12,
        radial_radius_mm=0.0,
        plane_radius_mm=0.0,
        policy=policy,
        root_beta=seed[0],
        centerline_seed=seed,
        sweep_directions=("outward",),
    )

    assert surface.complete
    assert surface.metrics["joint_margin_min_deg"] >= 1.9
