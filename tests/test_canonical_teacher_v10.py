from __future__ import annotations

import numpy as np

from quasi_exp.teacher.canonical import (
    CanonicalTeacher,
    TeacherPolicy,
    TeacherVariant,
    TrajectorySpec,
    link_cyclic_candidates,
    weighted_damped_pinv,
)


class LinearForwardEnvironment:
    """Independent exact adapter used to exercise the public teacher seam."""

    bounds = np.deg2rad(
        np.asarray([[-30.0, 30.0], [-30.0, 30.0], [-30.0, 30.0]] * 2)
    )

    def fk(self, beta: np.ndarray) -> np.ndarray:
        values = np.asarray(beta, dtype=float).reshape(-1, 6)
        return values[:, :3]

    def jacobian(self, beta: np.ndarray) -> np.ndarray:
        del beta
        return np.concatenate([np.eye(3), np.zeros((3, 3))], axis=1)

    def theta(self, beta: np.ndarray) -> np.ndarray:
        return np.repeat(np.asarray(beta, dtype=float).reshape(-1, 6), 5, axis=1)


def test_weighted_damped_pinv_solves_full_rank_task_without_moving_null_joints() -> None:
    jacobian = np.concatenate([np.eye(3), np.zeros((3, 3))], axis=1)

    pinv = weighted_damped_pinv(
        jacobian,
        damping=0.0,
        weights=np.asarray([4.0, 4.0, 2.0, 2.0, 1.0, 1.0]),
    )
    step = pinv @ np.asarray([0.1, -0.2, 0.3])

    assert np.allclose(jacobian @ step, [0.1, -0.2, 0.3], atol=1.0e-12)
    assert np.allclose(step[3:], 0.0, atol=1.0e-12)


def test_cyclic_linker_prefers_one_smooth_branch_over_pointwise_switches() -> None:
    smooth = np.deg2rad(
        np.asarray(
            [
                [0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        )
    )
    jumping = smooth.copy()
    jumping[1::2, 0] += np.deg2rad(12.0)
    candidates = [
        np.vstack([smooth[index], jumping[index]]) for index in range(len(smooth))
    ]
    residual_mm = [np.asarray([0.2, 0.0]) for _ in candidates]

    selected, report = link_cyclic_candidates(
        candidates,
        residual_mm,
        lambda_velocity=1.0,
        closure_weight=5.0,
    )

    assert np.allclose(selected, smooth)
    assert report["success"] is True
    assert report["delta_beta_rms_max_deg"] < 1.0


def test_cyclic_linker_removes_edges_above_hard_transition_limit() -> None:
    layers = [
        np.deg2rad(np.asarray([[0.0] * 6])),
        np.deg2rad(np.asarray([[3.0] * 6])),
        np.deg2rad(np.asarray([[0.0] * 6])),
    ]

    selected, report = link_cyclic_candidates(
        layers,
        [np.asarray([0.0])] * 3,
        lambda_velocity=1.0,
        closure_weight=5.0,
        max_transition_deg=2.0,
    )

    assert selected.shape == (0, 6)
    assert report == {"success": False, "reason": "no_closed_path"}


def test_teacher_policy_fingerprint_binds_solver_semantics() -> None:
    baseline = TeacherPolicy(variant=TeacherVariant.T3, candidate_budget=16)
    same = TeacherPolicy(variant=TeacherVariant.T3, candidate_budget=16)
    changed = TeacherPolicy(variant=TeacherVariant.T3, candidate_budget=32)

    assert baseline.fingerprint == same.fingerprint
    assert baseline.fingerprint != changed.fingerprint


def test_t3_teacher_returns_closed_exact_trajectory_with_auditable_provenance() -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    targets = np.column_stack(
        [0.1 * np.cos(angle), 0.1 * np.sin(angle), np.full_like(angle, 0.05)]
    )
    spec = TrajectorySpec(
        trajectory_id="linear-circle",
        family_id="linear-family",
        radius_mm=100.0,
        target_xyz_m=targets,
        cyclic_cut=0,
        traversal_direction="forward",
    )
    policy = TeacherPolicy(
        variant=TeacherVariant.T3,
        candidate_budget=16,
        max_corrector_iterations=20,
        tracking_tolerance_mm=0.01,
        solver_seed=20260720,
    )

    trajectory = CanonicalTeacher(LinearForwardEnvironment()).solve(
        spec,
        policy,
        root_beta=np.asarray([0.1, 0.0, 0.05, 0.0, 0.0, 0.0]),
    )

    assert trajectory.success is True
    assert trajectory.beta_rad.shape == (24, 6)
    assert trajectory.theta_rad.shape == (24, 30)
    assert trajectory.metrics["residual_max_mm"] <= 0.01
    assert trajectory.metrics["seam_beta_rms_deg"] <= 1.0
    assert trajectory.provenance["teacher_policy_id"] == policy.fingerprint
    assert trajectory.provenance["trajectory_id"] == "linear-circle"
