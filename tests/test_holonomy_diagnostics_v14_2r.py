from __future__ import annotations

import numpy as np

from quasi_exp.teacher.canonical_atlas import AtlasCandidate, AtlasTaskNode
from quasi_exp.teacher.canonical_gauge import (
    GaugeCorrectorPolicy,
    gauge_locked_predictor_corrector,
)
from quasi_exp.teacher.holonomy_diagnostics import (
    canonical_physical_path,
    physical_audit_direction,
    physical_audit_entity_id,
    physical_repeat_perturbation,
)


def test_physical_repeat_identity_survives_path_rotation_reversal_and_phase_rename() -> None:
    path = (7, 9, 11, 7)
    rotated = (9, 11, 7, 9)
    reversed_path = tuple(reversed(path))
    entity = physical_audit_entity_id("patch_07", "fundamental_cycle", path)

    assert entity == physical_audit_entity_id(
        "patch_07", "fundamental_cycle", rotated
    )
    assert entity == physical_audit_entity_id(
        "patch_07", "fundamental_cycle", reversed_path
    )
    assert canonical_physical_path(path, cycle=True) == (7, 9, 11)

    forward = physical_audit_direction(path, "forward", cycle=True)
    reversed_runtime = physical_audit_direction(
        reversed_path, "reverse", cycle=True
    )
    assert forward == reversed_runtime

    left = physical_repeat_perturbation(entity, forward, 2, 1.0e-8)
    right = physical_repeat_perturbation(entity, reversed_runtime, 2, 1.0e-8)
    np.testing.assert_array_equal(left, right)
    assert np.linalg.norm(left) > 0.0


class _RedundantAffineEnvironment:
    bounds = np.asarray([[-1.0, 1.0]] * 6, dtype=float)

    def fk_and_jacobian(self, beta_rad: np.ndarray):
        beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
        xyz = beta[:, :3].copy()
        jac = np.zeros((len(beta), 3, 6), dtype=float)
        jac[:, :, :3] = np.eye(3)[None, :, :]
        return xyz, jac

    def fk(self, beta_rad: np.ndarray):
        return np.asarray(beta_rad, dtype=float).reshape(-1, 6)[:, :3].copy()


def test_gauge_locked_corrector_moves_only_redundant_posture_toward_anchor() -> None:
    environment = _RedundantAffineEnvironment()
    source_beta = np.asarray([0.0, 0.0, 0.0, 0.12, -0.09, 0.06])
    source = AtlasCandidate(
        node_id=0,
        candidate_id="source",
        beta_rad=source_beta,
        residual_mm=0.0,
        min_margin_deg=5.0,
        normalized_min_margin=0.5,
        condition_number=1.0,
    )
    target = AtlasTaskNode(1, np.asarray([0.2, -0.1, 0.3]), ())
    anchor = np.zeros(6, dtype=float)
    corrector = gauge_locked_predictor_corrector(
        environment,
        policy=GaugeCorrectorPolicy(
            mode="anchor_potential",
            gauge_gain=0.3,
            maximum_gauge_step_deg=0.25,
            anchor_weight=0.1,
            cartesian_step_mm=5.0,
            maximum_iterations=100,
        ),
        anchor_beta_rad=anchor,
    )

    outcome = corrector(source, target)

    assert outcome.success is True
    np.testing.assert_allclose(outcome.beta_rad[:3], target.xyz_m, atol=1.0e-10)
    assert np.linalg.norm(outcome.beta_rad[3:]) < np.linalg.norm(source_beta[3:])
    assert outcome.residual_mm <= 3.0


def test_proximal_slsqp_is_registered_target_blind_fallback() -> None:
    environment = _RedundantAffineEnvironment()
    source_beta = np.asarray([0.0, 0.0, 0.0, 0.12, -0.09, 0.06])
    source = AtlasCandidate(
        node_id=0,
        candidate_id="source",
        beta_rad=source_beta,
        residual_mm=0.0,
        min_margin_deg=5.0,
        normalized_min_margin=0.5,
        condition_number=1.0,
    )
    target = AtlasTaskNode(1, np.asarray([0.2, -0.1, 0.3]), ())
    corrector = gauge_locked_predictor_corrector(
        environment,
        policy=GaugeCorrectorPolicy(
            mode="proximal_slsqp",
            gauge_gain=0.1,
            maximum_gauge_step_deg=0.25,
            anchor_weight=0.1,
            cartesian_step_mm=1.25,
            maximum_iterations=100,
        ),
        anchor_beta_rad=np.zeros(6, dtype=float),
    )

    outcome = corrector(source, target)

    assert outcome.success is True
    assert outcome.residual_mm <= 3.0
    assert "proximal_slsqp" in outcome.status
